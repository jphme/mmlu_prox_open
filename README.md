# MMLU-ProX Open-Ended Processing

This repository processes the MMLU-ProX-Lite dataset to create an open-ended version by filtering questions that can be answered without seeing multiple choice options.

## Overview

The project consists of four main steps:
1. **Question Classification**: Uses LLM to classify questions as answerable in free-form format
2. **Dataset Processing**: Filters and transforms the multilingual MMLU-ProX-Lite dataset
3. **Dataset Upload**: Creates a new HuggingFace dataset with processed results
4. **Elluminate Integration**: Uploads processed data to Elluminate for analysis and evaluation

## Key Files

- `question_classifier.py` - LLM-based classifier using Bespoke Labs Curator
- `mmlu_analysis.ipynb` - Jupyter notebook for dataset processing and upload
- `elluminate_upload.ipynb` - Jupyter notebook for Elluminate integration
- `mmlu_prox_classified/` - Local classified dataset cache
- `is_answerable.csv` - Classification results mapping

## Datasets

### Input Dataset
- **MMLU-ProX-Lite**: `li-lab/MMLU-ProX-Lite` 
- 29 language configurations (af, ar, bn, cs, de, en, es, fr, hi, hu, id, it, ja, ko, mr, ne, pt, ru, sr, sw, te, th, uk, ur, vi, wo, yo, zh, zu)
- Original format: Multiple choice questions with options

### Output Dataset  
- **MMLU-ProX-Lite-Open**: `jphme/MMLU-ProX-Lite-open`
- Filtered to ~470 answerable questions per language (from 588 total)
- Columns: `question_id`, `question`, `answer`, `cot_content`, `category`, `src`
- Answer column contains actual answer text (not multiple choice letter)

### Elluminate Collections
- 29 separate collections (one per language)
- Collection naming: `MMLU-ProX-Lite-Open-{LANG}`
- Columns: `question_id`, `question`, `answer`, `category`, `src` (cot_content removed as empty)
- 470 questions per collection

## Usage

### Prerequisites
- Python environment with dependencies from `pyproject.toml`
- HuggingFace token in `.env` file as `HF_TOKEN`
- Elluminate API credentials in `.env` file as `ELLUMINATE_API_KEY` and `ELLUMINATE_BASE_URL`
- Access to Azure OpenAI or OpenAI API (for classification step)

### Running Classification
```bash
python question_classifier.py
```

### Processing and Upload
Run the Jupyter notebook `mmlu_analysis.ipynb` to:
- Load classified results
- Process all 29 language configs
- Filter by answerability 
- Transform answer format
- Upload to HuggingFace

### Elluminate Integration
Run the Jupyter notebook `elluminate_upload.ipynb` to:
- Load processed dataset from HuggingFace
- Remove empty `cot_content` column
- Create 29 language-specific collections in Elluminate
- Prepare data for upload (collections only, data upload to be implemented)

## Links

- **Original Dataset**: [li-lab/MMLU-ProX-Lite](https://huggingface.co/datasets/li-lab/MMLU-ProX-Lite)
- **Processed Dataset**: [jphme/MMLU-ProX-Lite-open](https://huggingface.co/datasets/jphme/MMLU-ProX-Lite-open)
- **Bespoke Labs Curator**: [GitHub](https://github.com/bespokelabsai/curator)
- **Elluminate Documentation**: [API Docs](https://docs.elluminate.de/llms-full.txt)

## Dataset Statistics

- **Total original questions**: 588 per language
- **Answerable questions**: ~470 per language (79.93%)
- **Languages supported**: 29
- **Total processed questions**: ~13,630 across all languages

## License

Follows the licensing of the original MMLU-ProX-Lite dataset.