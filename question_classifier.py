#!/usr/bin/env python3
"""
Curator pipeline to classify MMLU-ProX-Lite questions for free-form answerability.
"""

import os

from bespokelabs import curator
from bespokelabs.curator.request_processor.config import OnlineBackendParams
from datasets import load_dataset
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

os.environ["CURATOR_VIEWER"] = "1"
# os.environ["OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
#
#    level=logging.DEBUG,
# )


class AnswerableQuestion(BaseModel):
    is_answerable_freeform: bool


backend_params = OnlineBackendParams(
    # base_url="https://aiendpointsswe5672148277.cognitiveservices.azure.com/",
    max_requests_per_minute=25000,  # 25k requests per minute (25M TPM)
    max_tokens_per_minute=25_000_000,
    max_retries=3,
    require_all_responses=True,
)


class QuestionClassifier(curator.LLM):
    """A classifier for question answerability."""

    response_format = AnswerableQuestion

    def prompt(self, input: dict) -> str:
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
"{input["question"]}"

Please respond with a JSON object containing:
- "is_answerable_freeform": true or false
- "reasoning": your explanation
- "confidence": a number between 0.0 and 1.0"""

    def parse(self, input: dict, response: AnswerableQuestion) -> dict:
        """Parse the LLM response into the desired output format."""
        input["is_answerable"] = response.is_answerable_freeform
        return input


def test_sample_questions():
    """Test the classifier on a few sample questions to validate logic."""

    print("Loading dataset for sample questions...")
    dataset = load_dataset("li-lab/MMLU-ProX-Lite", "en")

    # Get a few sample questions from validation set
    sample_questions = [
        dataset["validation"][0]["question"],  # Math question about ring characteristic
        dataset["validation"][1]["question"],  # Math question about transformations
        dataset["test"][0]["question"],  # Business question with fill-in-the-blanks
        dataset["test"][1][
            "question"
        ],  # Business question about discount series comparison
    ]

    print("Sample questions to classify:")
    for i, q in enumerate(sample_questions, 1):
        print(f"\n{i}. {q[:100]}..." if len(q) > 100 else f"\n{i}. {q}")

    print("\n" + "=" * 80)
    print("RUNNING CLASSIFICATION...")
    print("=" * 80)

    # Create a dataset for the classifier
    test_dataset = [{"question": q} for q in sample_questions]
    qclassifier = QuestionClassifier(
        # model_name="azure/gpt-4.1-mini-gobal",
        model_name="azure/gpt-4.1-mini",
        backend="litellm",
        backend_params=backend_params,
    )

    try:
        print("Running classifier on sample questions...")
        curator_response = qclassifier(dataset=test_dataset)

        for i, result in enumerate(curator_response, 1):
            print(f"\n--- QUESTION {i} ---")
            print(
                f"Question: {sample_questions[i - 1][:100]}..."
                if len(sample_questions[i - 1]) > 100
                else f"Question: {sample_questions[i - 1]}"
            )
            print(f"Answerable in free-form: {result['is_answerable_freeform']}")
            print(f"Confidence: {result['confidence']}")
            print(f"Reasoning: {result['reasoning']}")
            print("-" * 50)

    except Exception as e:
        print(f"Error running classifier: {e}")
        import traceback

        traceback.print_exc()


'''
def process_dataset_split(dataset_split, split_name: str) -> List[dict]:
    """Process a dataset split and classify all questions."""
    print(f"Processing {split_name} split with {len(dataset_split)} questions...")

    # Create dataset with just the questions we need to classify
    questions_dataset = [
        {
            "question": example["question"],
            "question_id": example["question_id"],
            "category": example["category"],
            "src": example["src"],
            "split": split_name,
        }
        for example in dataset_split
    ]

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
    print(
        f"Questions NOT answerable in free-form: {(~df['is_answerable_freeform']).sum()}"
    )
    print(
        f"Percentage answerable in free-form: {df['is_answerable_freeform'].mean() * 100:.1f}%"
    )

    print("\nBreakdown by split:")
    split_summary = df.groupby("split")["is_answerable_freeform"].agg([
        "count",
        "sum",
        "mean",
    ])
    print(split_summary)

    print("\nBreakdown by category:")
    category_summary = df.groupby("category")["is_answerable_freeform"].agg([
        "count",
        "sum",
        "mean",
    ])
    print(category_summary)

    print("\nResults saved to:")
    print("- mmlu_prox_question_classification.csv")
    print("- mmlu_prox_question_classification.json")
'''

if __name__ == "__main__":
    test_sample_questions()
    # main()
