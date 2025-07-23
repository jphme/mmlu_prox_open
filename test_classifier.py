#!/usr/bin/env python3
"""
Test script to validate question classification logic on sample questions.
"""

import os
from question_classifier import QuestionClassifier
from datasets import load_dataset
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def test_sample_questions():
    """Test the classifier on a few sample questions to validate logic."""
    
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        print("Warning: AZURE_OPENAI_API_KEY not found. Please set it in the .env file to test the classifier.")
        return
    
    print("Loading dataset for sample questions...")
    dataset = load_dataset("li-lab/MMLU-ProX-Lite", "en")
    
    # Get a few sample questions from validation set
    sample_questions = [
        dataset["validation"][0]["question"],  # Math question about ring characteristic
        dataset["validation"][1]["question"],  # Math question about transformations
        dataset["test"][0]["question"],        # Business question with fill-in-the-blanks
        dataset["test"][1]["question"],        # Business question about discount series comparison
    ]
    
    print("Sample questions to classify:")
    for i, q in enumerate(sample_questions, 1):
        print(f"\n{i}. {q[:100]}..." if len(q) > 100 else f"\n{i}. {q}")
    
    print("\n" + "="*80)
    print("RUNNING CLASSIFICATION...")
    print("="*80)
    
    # Create a dataset for the classifier
    test_dataset = [{"question": q} for q in sample_questions]
    
    classifier = QuestionClassifier()
    
    try:
        print("Running classifier on sample questions...")
        curator_response = classifier(dataset=test_dataset)
        
        for i, result in enumerate(curator_response, 1):
            print(f"\n--- QUESTION {i} ---")
            print(f"Question: {sample_questions[i-1][:100]}..." if len(sample_questions[i-1]) > 100 else f"Question: {sample_questions[i-1]}")
            print(f"Answerable in free-form: {result['is_answerable_freeform']}")
            print(f"Confidence: {result['confidence']}")
            print(f"Reasoning: {result['reasoning']}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error running classifier: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_sample_questions()