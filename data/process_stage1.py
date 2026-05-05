#!/usr/bin/env python3
import os
import json
import argparse
import pandas as pd
from datasets import load_from_disk
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def process_jigsaw(input_dir: str):
    """Process the Jigsaw Toxic Comments dataset"""
    print("\nProcessing Jigsaw Toxic Comments...")
    
    train_csv = os.path.join(input_dir, "train.csv")
    
    if not os.path.exists(train_csv):
        print(f"  [Warn] train.csv not found: {train_csv}")
        return []
    
    try:
        df = pd.read_csv(train_csv)
    except Exception as e:
        print(f"  [Warn] Failed to read CSV: {e}")
        return []
    
    processed = []
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Processing"):
        text = row.get('comment_text', '')
        if pd.isna(text) or not text.strip():
            continue
        
        is_toxic = any([row.get(col, 0) == 1 for col in label_cols])
        label = 1.0 if is_toxic else 0.0
        
        processed.append({
            "text": str(text).strip(),
            "label": label,
        })
    
    print(f"  Processed {len(processed)} samples")
    print(f"  Toxic: {sum(1 for x in processed if x['label'] == 1.0):,}")
    print(f"  Safe: {sum(1 for x in processed if x['label'] == 0.0):,}")
    
    return processed


def process_toxigen(input_dir: str):
    """Process the ToxiGen dataset"""
    print("\nProcessing ToxiGen...")
    
    try:
        dataset = load_from_disk(input_dir)
        if hasattr(dataset, 'keys'):
            train_data = dataset.get('train', dataset)
        else:
            train_data = dataset
    except Exception as e:
        print(f"  [Warn] Failed to load: {input_dir}, error: {e}")
        return []
    
    processed = []
    
    for item in tqdm(train_data, desc="  Processing"):
        # ToxiGen uses the 'generation' field as text to classify
        text = item.get('generation', item.get('text', item.get('prompt', '')))
        if not text:
            continue
        
        # ToxiGen uses prompt_label: 1 = toxic prompt, 0 = safe prompt
        prompt_label = item.get('prompt_label', item.get('label', 0))
        
        # Convert to binary label: 1.0 = harmful, 0.0 = safe
        if isinstance(prompt_label, (int, float)):
            label = 1.0 if prompt_label > 0.5 else 0.0
        elif isinstance(prompt_label, str):
            label = 1.0 if prompt_label.lower() in ['toxic', 'true', '1'] else 0.0
        else:
            label = 0.0
        
        processed.append({
            "text": text.strip(),
            "label": label,
        })
    
    print(f"  Processed {len(processed)} samples")
    print(f"  Harmful: {sum(1 for x in processed if x['label'] == 1.0):,}")
    print(f"  Safe: {sum(1 for x in processed if x['label'] == 0.0):,}")
    
    return processed


def process_wildguardmix_prompts(input_dir: str):
    """
    Extract prompts from WildGuardMix and annotate as harmful or not.

    Steps:
    1. Load the parquet file.
    2. Extract all unique prompts.
    3. Use the prompt_harm_label field to annotate harmfulness.
    """
    print("\nProcessing WildGuardMix prompts...")
    
    # Search for the train parquet file
    train_file = None
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.parquet') and 'train' in file.lower():
                train_file = os.path.join(root, file)
                break
        if train_file:
            break
    
    if not train_file:
        print(f"  [Warn] Could not find training parquet file")
        return []
    
    print(f"  Found training file: {train_file}")
    
    try:
        df = pd.read_parquet(train_file)
    except Exception as e:
        print(f"  [Warn] Failed to read parquet: {e}")
        return []
    
    print(f"  Total samples: {len(df):,}")
    
    # Extract unique prompts and their corresponding prompt_harm_label
    prompt_to_label = {}
    skipped_no_prompt = 0
    skipped_no_label = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Processing data"):
        prompt = row.get('prompt', '')
        if pd.isna(prompt) or not str(prompt).strip():
            skipped_no_prompt += 1
            continue
        
        prompt = str(prompt).strip()
        
        # Only process the first occurrence of each prompt
        if prompt in prompt_to_label:
            continue
        
        prompt_harm_label = row.get('prompt_harm_label', None)
        if pd.isna(prompt_harm_label) or prompt_harm_label is None:
            skipped_no_label += 1
            # Default to unharmful if label missing
            prompt_to_label[prompt] = 0.0
        else:
            # prompt_harm_label: 'harmful', 'unharmful', 'benign', or similar
            if isinstance(prompt_harm_label, str):
                label = 1.0 if prompt_harm_label.lower() in ['harmful', 'true', '1'] else 0.0
            else:
                label = 1.0 if prompt_harm_label > 0.5 else 0.0
            prompt_to_label[prompt] = label
    
    processed = []
    for prompt, label in prompt_to_label.items():
        processed.append({
            "text": prompt,
            "label": label,
        })
    
    print(f"  Processed {len(processed)} unique prompts")
    print(f"  Harmful: {sum(1 for x in processed if x['label'] == 1.0):,}")
    print(f"  Unharmful: {sum(1 for x in processed if x['label'] == 0.0):,}")
    print(f"  Skipped (no prompt): {skipped_no_prompt:,}")
    print(f"  Skipped (no label): {skipped_no_label:,}")
    
    return processed


def first_existing_path(paths):
    """Return the first path that exists, or the first candidate if none exist."""
    for path in paths:
        if os.path.exists(path):
            return path
    return paths[0]


def main():
    parser = argparse.ArgumentParser(description="Stage 1 Harmfulness Classification Data Processing")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/raw",
        help="Input data directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/stage1",
        help="Output data directory"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
        help="Proportion of validation set"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Stage 1 Harmfulness Classification Data Processing")
    print("=" * 60)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_data = []
    
    jigsaw_dir = first_existing_path([
        os.path.join(args.input_dir, "discriminative", "jigsaw"),
        os.path.join(args.input_dir, "jigsaw"),
    ])
    if os.path.exists(jigsaw_dir):
        jigsaw_data = process_jigsaw(jigsaw_dir)
        all_data.extend(jigsaw_data)
    else:
        print(f"  [Warn] Jigsaw data not found: {jigsaw_dir}")
    
    toxigen_dir = first_existing_path([
        os.path.join(args.input_dir, "discriminative", "toxigen"),
        os.path.join(args.input_dir, "toxigen"),
    ])
    if os.path.exists(toxigen_dir):
        toxigen_data = process_toxigen(toxigen_dir)
        all_data.extend(toxigen_data)
    else:
        print(f"  [Warn] ToxiGen data not found: {toxigen_dir}")
    
    wgmix_dir = first_existing_path([
        os.path.join(args.input_dir, "generative", "wildguardmix"),
        os.path.join(args.input_dir, "wildguardmix"),
        args.input_dir,
    ])
    if os.path.exists(wgmix_dir):
        wgmix_data = process_wildguardmix_prompts(wgmix_dir)
        all_data.extend(wgmix_data)
    else:
        print(f"  [Warn] WildGuardMix data not found: {wgmix_dir}")
        print("  Hint: place WildGuardMix parquet files under data/raw/wildguardmix")
    
    if not all_data:
        print("\n[Error] No data processed, please download the datasets first.")
        return 1
    
    # Split train and validation sets
    print(f"\nSplitting training and validation sets...")
    train_data, val_data = train_test_split(
        all_data,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=[x['label'] for x in all_data]  # Preserve class ratio
    )
    
    # Save
    train_path = os.path.join(args.output_dir, "train.json")
    val_path = os.path.join(args.output_dir, "val.json")
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nProcessing complete!")
    print(f"  Train set: {len(train_data):,} samples")
    print(f"  Validation set: {len(val_data):,} samples")
    print(f"  Train set saved to: {train_path}")
    print(f"  Validation set saved to: {val_path}")
    
    # Statistics
    train_harmful = sum(1 for x in train_data if x['label'] == 1.0)
    train_unharmful = sum(1 for x in train_data if x['label'] == 0.0)
    val_harmful = sum(1 for x in val_data if x['label'] == 1.0)
    val_unharmful = sum(1 for x in val_data if x['label'] == 0.0)
    
    print(f"\n[Stats]")
    print(f"  Train set: Harmful={train_harmful:,} ({train_harmful/len(train_data)*100:.1f}%), Unharmful={train_unharmful:,} ({train_unharmful/len(train_data)*100:.1f}%)")
    print(f"  Validation set: Harmful={val_harmful:,} ({val_harmful/len(val_data)*100:.1f}%), Unharmful={val_unharmful:,} ({val_unharmful/len(val_data)*100:.1f}%)")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
