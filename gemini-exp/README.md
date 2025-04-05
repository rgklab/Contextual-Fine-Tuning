# Medical Question Pipeline

A comprehensive pipeline for generating educational datasets from hard medical questions and evaluating models on MedQA.

## Overview

This pipeline consists of two main components:

1. **Dataset Generation** - Creates a dataset from incorrectly answered questions by:
   - Loading questions that were incorrectly answered from a text file
   - Generating educational content for these questions using Claude
   - Creating contextual prompts for effective learning
   - Formatting the data into a CSV for model training

2. **Model Evaluation** - Evaluates models on the MedQA dataset by:
   - Running inference on test questions
   - Processing model responses with Claude to extract predicted answers
   - Calculating and reporting accuracy

## Requirements

- Python 3.7+
- Anthropic API key (for Claude)
- Google AI API key (for Gemini)

Install dependencies:
```bash
pip install anthropic google-generativeai datasets tqdm argparse
```

## Usage

### Generate Dataset

To generate a dataset from hard questions:

```bash
python pipeline.py generate-dataset \
  --claude-api-key YOUR_CLAUDE_API_KEY \
  --indices-file path/to/incorrect_prediction_indices.txt \
  --output-csv path/to/output.csv \
  --batch-size 50
```

Arguments:
- `--claude-api-key`: Your Anthropic API key
- `--indices-file`: Text file containing incorrect prediction indices
- `--output-csv`: Path to save the generated CSV dataset
- `--batch-size`: Number of questions to process in each batch (default: 50)

### Evaluate Model

To evaluate a model on MedQA:

```bash
python pipeline.py evaluate \
  --gemini-api-key YOUR_GEMINI_API_KEY \
  --claude-api-key YOUR_CLAUDE_API_KEY \
  --model-name MODEL_NAME \
  --output-eval-file path/to/eval_results.json \
  --output-interp-file path/to/interp_results.json
```

Arguments:
- `--gemini-api-key`: Your Google AI API key
- `--claude-api-key`: Your Anthropic API key
- `--model-name`: Name of the Gemini model to evaluate
- `--output-eval-file`: Path to save the evaluation results
- `--output-interp-file`: Path to save the interpretation results

## Example Workflow

1. Generate dataset from the indices file:
```bash
python pipeline.py generate-dataset \
  --claude-api-key YOUR_CLAUDE_API_KEY \
  --indices-file incorrect_prediction_indices.txt \
  --output-csv datasets/training_data.csv
```

2. Fine-tune model with the generated dataset (using Google AI platform)

3. Evaluate fine-tuned model:
```bash
python pipeline.py evaluate \
  --gemini-api-key YOUR_GEMINI_API_KEY \
  --claude-api-key YOUR_CLAUDE_API_KEY \
  --model-name tunedModels/your-tuned-model \
  --output-eval-file results/tuned_eval.json \
  --output-interp-file results/tuned_interp.json
```