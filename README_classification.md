# MMLU-ProX-Lite Question Classification Pipeline

This pipeline uses Bespoke Curator and GPT-4.1-mini-global (via Azure OpenAI) to classify questions from the MMLU-ProX-Lite dataset to determine whether they can be answered in free-form format without knowing the multiple choice answer options.

## Setup

1. Install dependencies:
```bash
uv add datasets huggingface_hub bespokelabs-curator python-dotenv
```

2. Set your Azure OpenAI API key in the `.env` file:
```
AZURE_OPENAI_API_KEY=your-azure-api-key-here
```

## Configuration

The pipeline is configured to use:
- **Azure OpenAI Endpoint**: `https://aiendpointsswe5672148277.cognitiveservices.azure.com/`
- **Model**: `gpt-4.1-mini-gobal`
- **API Version**: `2025-01-01-preview`
- **Rate Limit**: 25,000 requests per minute (25M TPM)
- **Pricing**: $0.4/M input tokens, $1.6/M output tokens

## Usage

### Test the classifier on sample questions:
```bash
uv run python test_classifier.py
```

### Run the full classification pipeline:
```bash
uv run python question_classifier.py
```

This will:
- Load the MMLU-ProX-Lite dataset (English version)
- Process all questions in validation (70) and test (588) splits
- Classify each question for free-form answerability
- Save results to CSV and JSON files
- Print summary statistics

## Classification Criteria

A question is considered **answerable in free-form** if:
1. It is self-contained and doesn't reference answer choices
2. It can be answered with specific knowledge without needing to compare given options  
3. It doesn't require choosing between provided alternatives

A question is **NOT answerable in free-form** if:
1. It references answer options (e.g., "Which of the following...")
2. It asks to select from given choices
3. The answer requires comparing multiple provided options
4. It contains phrases like "Select all that apply", "Choose the best answer", etc.

## Output Files

- `mmlu_prox_question_classification.csv` - Results in CSV format
- `mmlu_prox_question_classification.json` - Results in JSON format

Each result includes:
- `question_id`: Original question ID
- `question`: The question text
- `is_answerable_freeform`: Boolean classification result
- `reasoning`: GPT-4o-mini's reasoning for the classification
- `confidence`: Confidence score (0.0 to 1.0)
- `category`: Question category (math, business, etc.)
- `src`: Source dataset
- `split`: Dataset split (validation/test)

## Dataset Information

The MMLU-ProX-Lite dataset contains questions from various domains including:
- Mathematics
- Business 
- Science
- And many others

Each question has up to 10 multiple choice options (option_0 through option_9) and includes chain-of-thought reasoning in the `cot_content` field.