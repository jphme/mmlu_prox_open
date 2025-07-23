#!/usr/bin/env python3
"""
Curator pipeline to classify MMLU-ProX-Lite questions for free-form answerability.
"""

import os
from typing import List
from pydantic import BaseModel
from datasets import load_dataset
import pandas as pd
from bespokelabs import curator
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class QuestionClassifier(curator.LLM):
    """Curator LLM for classifying question answerability."""
    
    def __init__(self):
        # Azure OpenAI configuration
        from bespokelabs.curator.request_processor.config import OnlineBackendParams
        
        backend_params = OnlineBackendParams(
            base_url="https://aiendpointsswe5672148277.cognitiveservices.azure.com/",
            max_requests_per_minute=25000,  # 25k requests per minute (25M TPM)
            max_retries=3,
            require_all_responses=True
        )
        
        # Set up the API key as environment variable for OpenAI backend
        os.environ["OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
        
        super().__init__(
            model_name="gpt-4.1-mini-gobal",
            backend="openai",
            backend_params=backend_params,
            generation_params={"api_version": "2025-01-01-preview"}
            # Pricing: $0.4/M input tokens, $1.6/M output tokens
        )
    
    def prompt(self, question: str) -> str:
        return f"""You are analyzing a multiple choice question to determine if it can be answered in free-form format without seeing the answer options.

A question is answerable in free-form if:
1. It is self-contained and doesn't reference answer choices (e.g., "Which of the following...", "Select the best option...")
2. It can be answered with specific knowledge without needing to compare given options
3. It doesn't require choosing between provided alternatives

A question is NOT answerable in free-form if:
1. It references answer options (e.g., "Which of the following statements is true?")
2. It asks to select from given choices
3. The answer requires comparing multiple provided options
4. It contains phrases like "Select all that apply", "Choose the best answer", etc.

Question to analyze:
"{question}"

Please respond with a JSON object containing:
- "is_answerable_freeform": true or false
- "reasoning": your explanation
- "confidence": a number between 0.0 and 1.0"""

    def parse(self, question: str, response: str) -> dict:
        """Parse the LLM response into the desired output format."""
        import json
        import re
        
        # Clean the response - remove markdown code blocks if present
        cleaned_response = response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = re.sub(r"```json\s*", "", cleaned_response)
            cleaned_response = re.sub(r"```\s*$", "", cleaned_response)
        elif cleaned_response.startswith("```"):
            cleaned_response = re.sub(r"```\s*", "", cleaned_response)
            cleaned_response = re.sub(r"```\s*$", "", cleaned_response)
        
        try:
            result = json.loads(cleaned_response)
            return {
                "question": question,
                "is_answerable_freeform": result.get("is_answerable_freeform", False),
                "reasoning": result.get("reasoning", ""),
                "confidence": result.get("confidence", 0.0)
            }
        except json.JSONDecodeError:
            # Fallback: try to extract information from text
            return {
                "question": question,
                "is_answerable_freeform": "true" in response.lower(),
                "reasoning": f"Could not parse JSON response: {response}",
                "confidence": 0.0
            }


def process_dataset_split(dataset_split, split_name: str) -> List[dict]:
    """Process a dataset split and classify all questions."""
    print(f"Processing {split_name} split with {len(dataset_split)} questions...")
    
    # Create dataset with just the questions we need to classify
    questions_dataset = [{"question": example["question"], "question_id": example["question_id"], 
                         "category": example["category"], "src": example["src"], "split": split_name} 
                        for example in dataset_split]
    
    classifier = QuestionClassifier()
    
    # Process the entire dataset at once
    curator_response = classifier(dataset=questions_dataset)
    
    # Extract results
    results = []
    for item in curator_response:
        results.append(item)
        
    print(f"Processed {len(results)} questions from {split_name} split")
    return results


def main():
    """Main function to run the classification pipeline."""
    # Set up Azure OpenAI API key (make sure it's in your environment)
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        print("Warning: AZURE_OPENAI_API_KEY not found in environment variables")
        print("Please set your Azure OpenAI API key in the .env file")
        return
    
    # Load the dataset
    print("Loading MMLU-ProX-Lite dataset...")
    dataset = load_dataset("li-lab/MMLU-ProX-Lite", "en")
    
    all_results = []
    
    # Process validation split
    validation_results = process_dataset_split(dataset["validation"], "validation")
    all_results.extend(validation_results)
    
    # Process test split
    test_results = process_dataset_split(dataset["test"], "test")
    all_results.extend(test_results)
    
    # Convert to DataFrame and save results
    df = pd.DataFrame(all_results)
    df.to_csv("mmlu_prox_question_classification.csv", index=False)
    df.to_json("mmlu_prox_question_classification.json", orient="records", indent=2)
    
    # Print summary statistics
    print("\n=== CLASSIFICATION RESULTS SUMMARY ===")
    print(f"Total questions processed: {len(all_results)}")
    print(f"Questions answerable in free-form: {df['is_answerable_freeform'].sum()}")
    print(f"Questions NOT answerable in free-form: {(~df['is_answerable_freeform']).sum()}")
    print(f"Percentage answerable in free-form: {df['is_answerable_freeform'].mean()*100:.1f}%")
    
    print("\nBreakdown by split:")
    split_summary = df.groupby('split')['is_answerable_freeform'].agg(['count', 'sum', 'mean'])
    print(split_summary)
    
    print("\nBreakdown by category:")
    category_summary = df.groupby('category')['is_answerable_freeform'].agg(['count', 'sum', 'mean'])
    print(category_summary)
    
    print(f"\nResults saved to:")
    print(f"- mmlu_prox_question_classification.csv")
    print(f"- mmlu_prox_question_classification.json")


if __name__ == "__main__":
    main()