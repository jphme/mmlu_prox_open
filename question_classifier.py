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

os.environ["CURATOR_VIEWER"] = "0"
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

A question is NOT answerable in free-form if:
1. It references answer options (e.g., "Which of the following statements is true?")
2. It contains phrases like "Select all that apply", "Choose the best answer", etc.
3. It contains multiple placeholders that need to be filled in a specifc order.

Question to analyze:
"{input["question"]}"

Please respond with a JSON object containing:
- "is_answerable_freeform": true or false
"""

    def parse(self, input: dict, response: AnswerableQuestion) -> dict:
        """Parse the LLM response into the desired output format."""
        input["is_answerable"] = response.is_answerable_freeform
        return input


def test_sample_questions():
    """Test the classifier on a few sample questions to validate logic."""

    print("Loading dataset for sample questions...")
    dataset = load_dataset("li-lab/MMLU-ProX-Lite", "en")

    # Get a few sample questions from validation set
    """sample_questions = [
        dataset["validation"][0]["question"],  # Math question about ring characteristic
        dataset["validation"][1]["question"],  # Math question about transformations
        dataset["test"][0]["question"],  # Business question with fill-in-the-blanks
        dataset["test"][1][
            "question"
        ],  # Business question about discount series comparison
    ]"""
    sample_questions = dataset["test"]  # .take(10)

    # print("Sample questions to classify:")
    # for i, q in enumerate(sample_questions, 1):
    #    print(f"\n{i}. {q[:100]}..." if len(q) > 100 else f"\n{i}. {q}")

    print("\n" + "=" * 80)
    print("RUNNING CLASSIFICATION...")
    print("=" * 80)

    # Create a dataset for the classifier
    # test_dataset = [{"question": q} for q in sample_questions]
    qclassifier = QuestionClassifier(
        # model_name="azure/gpt-4.1-mini-gobal",
        model_name="azure/gpt-4.1-mini",
        backend="litellm",
        backend_params=backend_params,
    )

    print("Running classifier on sample questions...")
    curator_response = qclassifier(dataset=sample_questions)  # test_dataset)

    print(curator_response.dataset)
    curator_response.dataset.save_to_disk("mmlu_prox_classified")


if __name__ == "__main__":
    test_sample_questions()
    # main()
