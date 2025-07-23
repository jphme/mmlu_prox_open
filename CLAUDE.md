# CLAUDE.md - Assistant Instructions

This file contains instructions and context for Claude Code when working with this MMLU-ProX open-ended processing repository.

## Repository Overview

This repo processes MMLU-ProX-Lite dataset to create an open-ended version by filtering questions answerable without multiple choice options.

## Key Components

### 1. Question Classification (`question_classifier.py`)
- **Purpose**: Classifies questions as answerable in free-form format using LLM
- **Framework**: Uses Bespoke Labs Curator with Azure OpenAI backend
- **Input**: MMLU-ProX-Lite English dataset (`li-lab/MMLU-ProX-Lite`)
- **Output**: `mmlu_prox_classified/` directory with classification results
- **Key criteria**: Questions must be self-contained and not reference answer choices

### 2. Dataset Processing (`mmlu_analysis.ipynb`)
- **Purpose**: Process all 29 language configs and upload filtered dataset
- **Steps**:
  1. Load classified results from `mmlu_prox_classified/`
  2. Create answerable questions ID set (questions where `is_answerable==1`)
  3. Process each language config:
     - Filter by answerable question IDs
     - Replace answer column with actual answer text from `option_{answer_index}`
     - Keep only columns: `question_id`, `question`, `answer`, `cot_content`, `category`, `src`
  4. Upload as DatasetDict to `jphme/MMLU-ProX-Lite-open`

### 3. Data Files
- `mmlu_prox_classified/`: Local cache of classified dataset
- `is_answerable.csv`: CSV mapping of question_id to is_answerable flag
- `.env`: Contains HF_TOKEN for HuggingFace authentication

## Dataset Format

### Input (MMLU-ProX-Lite)
```python
{
    'question_id': int,
    'question': str,
    'option_0': str, 'option_1': str, ..., 'option_9': str,
    'answer': str,  # Letter like 'A', 'B', etc.
    'answer_index': int,  # Index of correct option
    'cot_content': str,
    'category': str,
    'src': str,
    'question_id_src': int
}
```

### Output (MMLU-ProX-Lite-Open)
```python
{
    'question_id': int,
    'question': str,
    'answer': str,  # Actual answer text, not letter
    'cot_content': str,
    'category': str,
    'src': str
}
```

## Language Configurations
29 total: af, ar, bn, cs, de, en, es, fr, hi, hu, id, it, ja, ko, mr, ne, pt, ru, sr, sw, te, th, uk, ur, vi, wo, yo, zh, zu

## Important Commands

### Run Classification
```bash
python question_classifier.py
```

### Process and Upload (via notebook)
Execute all cells in `mmlu_analysis.ipynb`

### Check Dataset Stats
- Total questions per language: 588
- Answerable questions per language: ~470 (79.93%)
- Total processed questions: ~13,630 across all languages

## Environment Setup
- Requires HF_TOKEN in `.env` file
- Dependencies in `pyproject.toml`
- Azure OpenAI or OpenAI API access for classification

## Key Links
- Original: https://huggingface.co/datasets/li-lab/MMLU-ProX-Lite
- Processed: https://huggingface.co/datasets/jphme/MMLU-ProX-Lite-open
- Curator: https://github.com/bespokelabsai/curator

## Troubleshooting
- If HuggingFace upload fails, check HF_TOKEN in `.env`
- If classification fails, verify Azure OpenAI credentials
- Dataset processing expects exactly 588 questions per language config