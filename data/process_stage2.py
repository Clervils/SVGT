#!/usr/bin/env python3
import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import sys

# Setup project path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def load_beavertails(split: str = 'train'):
    """
    Load BeaverTails dataset from HuggingFace.
    
    Args:
        split: Dataset split ('train' or 'test')
    
    Returns:
        List of data items with 'prompt', 'response', and 'label' fields
    """
    try:
        from datasets import load_dataset
        dataset = load_dataset("PKU-Alignment/BeaverTails", split=split)
        data = []
        for item in dataset:
            data.append({
                'prompt': item.get('prompt', ''),
                'response': item.get('response', ''),
                'label': item.get('label', 0),  # 0 = safe, 1 = unsafe
            })
        return data
    except Exception as e:
        print(f"  [Warning] Failed to load BeaverTails from HuggingFace: {e}")
        print(f"  Hint: Make sure you have internet connection and datasets library installed")
        return []


def process_wildguardtrain(input_dir: str):
    """
    Process the WildGuardMix dataset, extracting prompt+response pairs.

    Strategy:
    1. Read the parquet file
    2. Extract all prompt+response pairs
    3. response_is_harmful: determined by response_harm_label
       - response_harm_label="harmful" -> response_is_harmful=True
       - response_harm_label="unharmful" -> response_is_harmful=False
    Note: prompt_is_harmful is not needed for Stage 2 training
    """
    print("\nProcessing WildGuardMix (prompt+response pairs)...")
    
    train_file = None
    if os.path.isdir(input_dir):
        for file in os.listdir(input_dir):
            if file.endswith('.parquet') and 'train' in file.lower():
                train_file = os.path.join(input_dir, file)
                break
        if train_file is None:
            train_dir = os.path.join(input_dir, 'train')
            if os.path.isdir(train_dir):
                for file in os.listdir(train_dir):
                    if file.endswith('.parquet'):
                        train_file = os.path.join(train_dir, file)
                        break
    
    if train_file is None:
        print("  [Warning] No training parquet file found.")
        return []
    
    try:
        df = pd.read_parquet(train_file)
        print(f"  [OK] Read parquet file: {train_file}")
        print(f"  [Stats] Number of samples: {len(df)}")
    except Exception as e:
        print(f"  [Warning] Failed to read parquet: {e}")
        return []
    
    processed = []
    skipped_no_prompt = 0
    skipped_no_response = 0
    skipped_no_label = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Processing"):
        prompt = row.get('prompt', '')
        response = row.get('response', '')
        response_harm_label = row.get('response_harm_label', None)
        
        # Process prompt
        if pd.isna(prompt) or not str(prompt).strip():
            skipped_no_prompt += 1
            continue
        prompt = str(prompt).strip()
        
        # Process response
        if pd.isna(response) or not str(response).strip():
            skipped_no_response += 1
            continue
        response = str(response).strip()
        
        # Process response_harm_label
        if pd.isna(response_harm_label):
            skipped_no_label += 1
            continue
        
        # response_harm_label: "harmful" = True, "unharmful" = False
        if isinstance(response_harm_label, str):
            response_is_harmful = response_harm_label.lower() == 'harmful'
        else:
            response_is_harmful = bool(response_harm_label)
        
        processed.append({
            "prompt": prompt,
            "response": response,
            "response_is_harmful": response_is_harmful,
        })
    
    print(f"  [OK] Processed {len(processed)} samples.")
    print(f"  [Warning] Skipped missing prompt: {skipped_no_prompt:,}")
    print(f"  [Warning] Skipped missing response: {skipped_no_response:,}")
    print(f"  [Warning] Skipped missing label: {skipped_no_label:,}")
    
    response_harmful = sum(1 for x in processed if x['response_is_harmful'])
    response_safe = sum(1 for x in processed if not x['response_is_harmful'])
    
    print(f"\n[Stats]")
    print(f"  Response: Harmful={response_harmful:,} ({response_harmful/len(processed)*100:.1f}%), Safe={response_safe:,} ({response_safe/len(processed)*100:.1f}%)")
    
    return processed


def process_beavertails(split: str = 'train'):
    """
    Process the BeaverTails dataset, extracting prompt+response pairs.
    
    Args:
        split: dataset split ('train' or 'test')
    
    Returns:
        List of processed data in the same format as WildGuardMix.
    """
    print(f"\nProcessing BeaverTails (prompt+response pairs, split={split})...")
    
    try:
        beavertails_data = load_beavertails(split=split)
        
        if not beavertails_data:
            print("  [Warning] BeaverTails data is empty.")
            return []
        
        print(f"  [OK] Loaded BeaverTails data: {len(beavertails_data)} samples.")
        
        processed = []
        skipped_no_prompt = 0
        skipped_no_response = 0
        
        for item in tqdm(beavertails_data, desc="  Processing"):
            prompt = item.get('prompt', '')
            response = item.get('response', '')
            label = item.get('label', 0)  # 0 = safe, 1 = unsafe
            
            # Process prompt
            if not prompt or not str(prompt).strip():
                skipped_no_prompt += 1
                continue
            prompt = str(prompt).strip()
            
            # Process response
            if not response or not str(response).strip():
                skipped_no_response += 1
                continue
            response = str(response).strip()
            
            # BeaverTails: label 0 = safe, 1 = unsafe
            response_is_harmful = (label == 1)
            
            processed.append({
                "prompt": prompt,
                "response": response,
                "response_is_harmful": response_is_harmful,
            })
        
        print(f"  [OK] Processed {len(processed)} samples.")
        print(f"  [Warning] Skipped missing prompt: {skipped_no_prompt:,}")
        print(f"  [Warning] Skipped missing response: {skipped_no_response:,}")
        
        response_harmful = sum(1 for x in processed if x['response_is_harmful'])
        response_safe = sum(1 for x in processed if not x['response_is_harmful'])
        
        print(f"\n[BeaverTails Stats]")
        print(f"  Response: Harmful={response_harmful:,} ({response_harmful/len(processed)*100:.1f}%), Safe={response_safe:,} ({response_safe/len(processed)*100:.1f}%)")
        
        return processed
    except Exception as e:
        print(f"  [Error] Failed to process BeaverTails data: {e}")
        import traceback
        traceback.print_exc()
        return []


def main():
    parser = argparse.ArgumentParser(description="Stage 2 Data Processing: Conditional Response Harmfulness Classification")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/raw/wildguardmix",
        help="WildGuardMix raw data directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/stage2",
        help="Directory for processed data"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
        help="Validation split ratio"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--use_beavertails",
        action="store_true",
        help="Include BeaverTails dataset (merged into training data)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Stage 2 Data Processing: Conditional Response Harmfulness Classification")
    if args.use_beavertails:
        print("Data sources: WildGuardMix + BeaverTails")
    else:
        print("Data sources: WildGuardMix")
    print("=" * 60)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_data = []
    
    # Process WildGuardMix
    if not os.path.exists(args.input_dir):
        print(f"  [Warning] WildGuardMix data not found: {args.input_dir}")
        print(f"  Please run python data/download.py to download the data.")
    else:
        wildguard_data = process_wildguardtrain(args.input_dir)
        if wildguard_data:
            all_data.extend(wildguard_data)
            print(f"\n  [OK] WildGuardMix: {len(wildguard_data):,} samples")
    
    # Optionally process BeaverTails
    if args.use_beavertails:
        beavertails_data = process_beavertails(split='train')
        if beavertails_data:
            all_data.extend(beavertails_data)
            print(f"\n  [OK] BeaverTails: {len(beavertails_data):,} samples")
    
    if not all_data:
        print("\n[Error] No data processed.")
        return 1
    
    # Dataset statistics before deduplication
    print("\n" + "=" * 60)
    print("Merged dataset statistics (before deduplication)")
    print("=" * 60)
    total_samples_before = len(all_data)
    print(f"  Total samples before deduplication: {total_samples_before:,}")

    # Each dataset contribution
    if args.use_beavertails:
        wildguard_count = len(wildguard_data) if wildguard_data else 0
        beavertails_count = len(beavertails_data) if beavertails_data else 0
        print(f"  - WildGuardMix: {wildguard_count:,} samples")
        print(f"  - BeaverTails: {beavertails_count:,} samples")
        print(f"  - Total (should match sum above): {wildguard_count + beavertails_count:,} samples")
        if total_samples_before != (wildguard_count + beavertails_count):
            print(f"  [Warning] Actual merged sample count ({total_samples_before:,}) does not match expected sum.")

    # Deduplication based on (prompt, response) pair
    print(f"\n  Deduplicating based on prompt+response pairs...")
    seen = set()
    deduplicated_data = []
    duplicates = 0

    for item in all_data:
        key = (item['prompt'].strip(), item['response'].strip())
        if key not in seen:
            seen.add(key)
            deduplicated_data.append(item)
        else:
            duplicates += 1

    all_data = deduplicated_data
    total_samples_after = len(all_data)

    print(f"  Total samples after deduplication: {total_samples_after:,}")
    print(f"  Removed {duplicates:,} duplicate samples ({duplicates/total_samples_before*100:.2f}%)")

    # Annotation distribution after deduplication
    response_harmful = sum(1 for x in all_data if x['response_is_harmful'])
    response_safe = sum(1 for x in all_data if not x['response_is_harmful'])
    print(f"\n  Label distribution after deduplication:")
    print(f"  Response Harmful: {response_harmful:,} ({response_harmful/total_samples_after*100:.1f}%)")
    print(f"  Response Safe: {response_safe:,} ({response_safe/total_samples_after*100:.1f}%)")
    
    # Train/val split
    print(f"\nSplitting train and validation sets (test_size={args.test_size})...")
    train_data, val_data = train_test_split(
        all_data,
        test_size=args.test_size,
        random_state=args.seed,
    )
    
    # Save
    train_path = os.path.join(args.output_dir, "train.json")
    val_path = os.path.join(args.output_dir, "val.json")
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n[Success] Processing complete.")
    print(f"  [OK] Total samples after deduplication: {total_samples_after:,}")
    print(f"  [OK] Training set: {len(train_data):,} samples ({len(train_data)/total_samples_after*100:.1f}%)")
    print(f"  [OK] Validation set: {len(val_data):,} samples ({len(val_data)/total_samples_after*100:.1f}%)")
    print(f"  [OK] Training set saved to: {train_path}")
    print(f"  [OK] Validation set saved to: {val_path}")
    
    # Final statistics
    train_harmful = sum(1 for x in train_data if x['response_is_harmful'])
    train_safe = sum(1 for x in train_data if not x['response_is_harmful'])
    val_harmful = sum(1 for x in val_data if x['response_is_harmful'])
    val_safe = sum(1 for x in val_data if not x['response_is_harmful'])
    
    print(f"\n[Final Distribution]")
    print(f"  Training set: Harmful={train_harmful:,} ({train_harmful/len(train_data)*100:.1f}%), Safe={train_safe:,} ({train_safe/len(train_data)*100:.1f}%)")
    print(f"  Validation set: Harmful={val_harmful:,} ({val_harmful/len(val_data)*100:.1f}%), Safe={val_safe:,} ({val_safe/len(val_data)*100:.1f}%)")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
