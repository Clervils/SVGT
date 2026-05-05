# Data Preparation

This directory contains scripts for processing training data. You need to download the following datasets before running the processing scripts.

## Required Datasets

### 1. WildGuardMix (Required for Stage 1 and Stage 2)

**Source**: [HuggingFace - allenai/wildguardmix](https://huggingface.co/datasets/allenai/wildguardmix)


**Required files**:
- `wildguard_train.parquet` (or equivalent format)
- Should contain columns: `prompt`, `response`, `prompt_harm_label`, `response_harm_label`


### 2. BeaverTails (Optional for Stage 2)

**Source**: [HuggingFace - PKU-Alignment/BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails)

**Note**: BeaverTails is automatically downloaded from HuggingFace if not found locally.


**Note**: This is optional. Stage 1 can work with only WildGuardMix prompts.

## Data Processing

### Stage 1: Unconditional Value Learning

Process prompts for harmful/safe classification:

```bash
python data/process_stage1.py \
    --input_dir data/raw/wildguardmix \
    --output_dir data/processed/stage1 \
    --test_size 0.1
```

**Input**: WildGuardMix parquet files with `prompt` and `prompt_harm_label` columns

**Output**: 
- `data/processed/stage1/train.json`
- `data/processed/stage1/val.json`

**Format**:
```json
{
  "text": "User prompt",
  "label": 1.0  // 1.0 = harmful, 0.0 = safe
}
```

### Stage 2: Conditional Value Learning

Process prompt-response pairs for response harmful/safe classification:

```bash
python data/process_stage2.py \
    --input_dir data/raw/wildguardmix \
    --output_dir data/processed/stage2 \
    --test_size 0.1 \
    --use_beavertails  # Optional: include BeaverTails dataset
```

**Input**: 
- WildGuardMix parquet files with `prompt`, `response`, and `response_harm_label` columns
- BeaverTails dataset from HuggingFace when `--use_beavertails` is enabled

**Output**:
- `data/processed/stage2/train.json`
- `data/processed/stage2/val.json`

**Format**:
```json
{
  "prompt": "User question",
  "response": "Model response",
  "response_is_harmful": true  // true = harmful, false = safe
}
```

## Directory Structure

After downloading and processing, your data directory should look like:

```
data/
├── raw/                            
│   ├── wildguardmix/
│   │   └── train.parquet
│   └── beavertails/
│       └── train.parquet
│
├── processed/                      
│   ├── stage1/
│   │   ├── train.json
│   │   └── val.json
│   │
│   └── stage2/
│       ├── train.json
│       └── val.json
│
└── README.md                      
```

## Notes

- All paths in the scripts are relative to the project root directory
- The processing scripts automatically handle data deduplication
- WildGuardMix requires HuggingFace access approval
- BeaverTails is automatically downloaded from HuggingFace if not found locally
