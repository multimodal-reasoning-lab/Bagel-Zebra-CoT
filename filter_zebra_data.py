#!/usr/bin/env python3
"""
Filter Zebra-CoT dataset to remove problematic samples that cause training issues.

Filters out:
1. Samples with no problem images (all problem_image_* are null)
2. Samples with no images at all (both problem and reasoning images are null)
3. Samples that are too long (exceed max token length)
4. Samples where referenced images don't exist on disk
"""

import json
import os
import re
from pathlib import Path
import argparse
from typing import Dict, List, Optional, Set
from tqdm import tqdm
from collections import defaultdict


def extract_image_references(text: str) -> List[str]:
    """Extract image references from text like <image_start>[problem_image_1]<image_end>"""
    pattern = r'<image_start>\[([^\]]+)\]<image_end>'
    matches = re.findall(pattern, text)
    return matches


def get_non_null_images(data_item: Dict) -> Dict[str, List[str]]:
    """Get all non-null image fields from the data item."""
    problem_images = []
    reasoning_images = []
    
    # Extract problem images
    for key, value in data_item.items():
        if key.startswith('problem_image_') and value is not None:
            problem_images.append(value)
    
    # Extract reasoning images
    for key, value in data_item.items():
        if key.startswith('reasoning_image_') and value is not None:
            reasoning_images.append(value)
    
    return {
        'problem_images': problem_images,
        'reasoning_images': reasoning_images
    }


def get_referenced_images(data_item: Dict) -> Set[str]:
    """Get all image references mentioned in Question and Text Reasoning Trace."""
    referenced_images = set()
    
    # Extract from Question
    question = data_item.get('Question', '')
    question_refs = extract_image_references(question)
    referenced_images.update(question_refs)
    
    # Extract from Text Reasoning Trace
    reasoning_trace = data_item.get('Text Reasoning Trace', '')
    reasoning_refs = extract_image_references(reasoning_trace)
    referenced_images.update(reasoning_refs)
    
    return referenced_images


def check_images_exist(data_item: Dict, image_base_dir: str) -> Dict[str, bool]:
    """Check if referenced images actually exist on disk."""
    referenced_images = get_referenced_images(data_item)
    results = {}
    
    for image_ref in referenced_images:
        if image_ref in data_item and data_item[image_ref] is not None:
            image_path = data_item[image_ref]
            full_path = os.path.join(image_base_dir, image_path)
            results[image_ref] = os.path.exists(full_path)
        else:
            results[image_ref] = False
    
    return results


def calculate_text_length(data_item: Dict, tokenizer=None) -> int:
    """Calculate total text length for the sample."""
    question = data_item.get('Question', '')
    reasoning_trace = data_item.get('Text Reasoning Trace', '')
    final_answer = data_item.get('Final Answer', '')
    
    # Remove image references for length calculation
    clean_question = re.sub(r'<image_start>\[[^\]]+\]<image_end>', '', question)
    clean_reasoning = re.sub(r'<image_start>\[[^\]]+\]<image_end>', '', reasoning_trace)
    
    if tokenizer is not None:
        # Use tokenized length (more accurate for training)
        total_tokens = 0
        for text in [clean_question, clean_reasoning, final_answer]:
            if text.strip():
                tokens = tokenizer.encode(text, add_special_tokens=False)
                total_tokens += len(tokens)
        return total_tokens
    else:
        # Fallback to character length with rough token estimation
        # Rough estimate: ~4 characters per token for English text
        char_length = len(clean_question) + len(clean_reasoning) + len(final_answer)
        return char_length // 4


def should_filter_sample(data_item: Dict, image_base_dir: str, max_text_length: int = 2500, tokenizer=None) -> Dict[str, bool]:
    """
    Determine if a sample should be filtered out and why.
    
    Returns a dict of filter reasons and whether each applies.
    """
    filters = {
        'no_problem_images': False,
        'no_images_at_all': False,
        'too_long': False,
        'missing_referenced_images': False,
        'missing_required_fields': False
    }
    
    # Check for required fields
    required_fields = ['Question', 'Text Reasoning Trace', 'Final Answer']
    for field in required_fields:
        if not data_item.get(field, '').strip():
            filters['missing_required_fields'] = True
            return filters
    
    # Get all non-null images
    images = get_non_null_images(data_item)
    problem_images = images['problem_images']
    reasoning_images = images['reasoning_images']
    
    # Check for no problem images
    if not problem_images:
        filters['no_problem_images'] = True
    
    # Check for no images at all
    if not problem_images and not reasoning_images:
        filters['no_images_at_all'] = True
    
    # Check text length
    text_length = calculate_text_length(data_item, tokenizer)
    if text_length > max_text_length:
        filters['too_long'] = True
    
    # Check if referenced images exist
    if image_base_dir:
        referenced_images = get_referenced_images(data_item)
        if referenced_images:
            image_existence = check_images_exist(data_item, image_base_dir)
            # If any referenced image is missing, filter it out
            if not all(image_existence.values()):
                filters['missing_referenced_images'] = True
    
    return filters


def filter_jsonl_file(input_path: str, output_path: str, image_base_dir: str = None, 
                     max_text_length: int = 2500, verbose: bool = True, tokenizer=None,
                     filtered_output_path: str = None):
    """Filter a JSONL file and save the filtered results."""
    
    stats = {
        'total_samples': 0,
        'filtered_samples': 0,
        'kept_samples': 0,
        'filter_reasons': {
            'no_problem_images': 0,
            'no_images_at_all': 0,
            'too_long': 0,
            'missing_referenced_images': 0,
            'missing_required_fields': 0
        }
    }
    
    # Track stats per dataset
    dataset_stats = defaultdict(lambda: {
        'total': 0,
        'kept': 0,
        'filtered': 0,
        'filter_reasons': defaultdict(int)
    })
    
    # Open files for writing
    outfile = open(output_path, 'w', encoding='utf-8')
    filtered_outfile = None
    if filtered_output_path:
        filtered_outfile = open(filtered_output_path, 'w', encoding='utf-8')
    
    try:
        # First pass: count total lines and datasets for progress bar
        print("Analyzing datasets and counting total lines...")
        total_lines = 0
        dataset_counts = defaultdict(int)
        
        with open(input_path, 'r', encoding='utf-8') as infile:
            for line in infile:
                total_lines += 1
                try:
                    data_item = json.loads(line.strip())
                    dataset_name = data_item.get('dataset_name', 'Unknown')
                    dataset_counts[dataset_name] += 1
                except:
                    continue
        
        # Print dataset overview
        print(f"\n=== Dataset Overview (Before Filtering) ===")
        for dataset_name in sorted(dataset_counts.keys()):
            print(f"{dataset_name}: {dataset_counts[dataset_name]} samples")
        print(f"Total: {total_lines} samples")
        print()
        
        with open(input_path, 'r', encoding='utf-8') as infile:
            # Use tqdm for progress bar
            pbar = tqdm(infile, total=total_lines, desc="Filtering data")
            for line_num, line in enumerate(pbar, 1):
                try:
                    data_item = json.loads(line.strip())
                    stats['total_samples'] += 1
                    
                    # Get dataset name
                    dataset_name = data_item.get('dataset_name', 'Unknown')
                    dataset_stats[dataset_name]['total'] += 1
                    
                    # Check if sample should be filtered
                    filter_results = should_filter_sample(data_item, image_base_dir, max_text_length, tokenizer)
                    
                    # If any filter applies, skip this sample
                    should_filter = any(filter_results.values())
                    
                    if should_filter:
                        stats['filtered_samples'] += 1
                        dataset_stats[dataset_name]['filtered'] += 1
                        
                        # Count filter reasons
                        active_filters = []
                        for reason, applies in filter_results.items():
                            if applies:
                                stats['filter_reasons'][reason] += 1
                                dataset_stats[dataset_name]['filter_reasons'][reason] += 1
                                active_filters.append(reason)
                        
                        if verbose:
                            tqdm.write(f"Line {line_num} [{dataset_name}]: Filtered - {', '.join(active_filters)}")
                        
                        # Save filtered data with reasons if requested
                        if filtered_outfile:
                            filtered_item = data_item.copy()
                            filtered_item['_filter_reasons'] = active_filters
                            filtered_item['_line_number'] = line_num
                            filtered_item['_dataset_name'] = dataset_name
                            filtered_outfile.write(json.dumps(filtered_item, ensure_ascii=False) + '\n')
                    else:
                        stats['kept_samples'] += 1
                        dataset_stats[dataset_name]['kept'] += 1
                        outfile.write(line)
                    
                    # Update progress bar with current stats every 1000 lines
                    if line_num % 1000 == 0:
                        pbar.set_postfix({
                            'kept': stats['kept_samples'],
                            'filtered': stats['filtered_samples'],
                            'rate': f"{stats['filtered_samples']/(stats['total_samples'])*100:.1f}%"
                        })
                        
                except json.JSONDecodeError as e:
                    tqdm.write(f"Error parsing JSON on line {line_num}: {e}")
                    stats['filtered_samples'] += 1
                    stats['filter_reasons']['missing_required_fields'] += 1
                    # Save invalid JSON if requested
                    if filtered_outfile:
                        invalid_item = {
                            'original_line': line.strip(),
                            '_filter_reasons': ['json_decode_error'],
                            '_line_number': line_num,
                            '_dataset_name': 'Unknown',
                            '_error': str(e)
                        }
                        filtered_outfile.write(json.dumps(invalid_item, ensure_ascii=False) + '\n')
                except Exception as e:
                    tqdm.write(f"Error processing line {line_num}: {e}")
                    stats['filtered_samples'] += 1
                    # Save processing error if requested
                    if filtered_outfile:
                        error_item = {
                            'original_line': line.strip(),
                            '_filter_reasons': ['processing_error'],
                            '_line_number': line_num,
                            '_dataset_name': 'Unknown',
                            '_error': str(e)
                        }
                        filtered_outfile.write(json.dumps(error_item, ensure_ascii=False) + '\n')
    finally:
        outfile.close()
        if filtered_outfile:
            filtered_outfile.close()
    
    # Add dataset stats to return
    stats['dataset_stats'] = dict(dataset_stats)
    return stats


def create_separate_filter_files(filtered_output_path: str):
    """Create separate files for each filter reason from the main filtered file."""
    if not os.path.exists(filtered_output_path):
        tqdm.write(f"Filtered output file {filtered_output_path} does not exist")
        return
    
    # Create directory for separate files
    base_dir = os.path.dirname(filtered_output_path)
    base_name = os.path.splitext(os.path.basename(filtered_output_path))[0]
    separate_dir = os.path.join(base_dir, f"{base_name}_by_reason")
    os.makedirs(separate_dir, exist_ok=True)
    
    # Also create directory for dataset-specific files
    dataset_dir = os.path.join(base_dir, f"{base_name}_by_dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Track files and stats
    reason_files = {}
    reason_stats = {}
    dataset_files = {}
    dataset_stats = {}
    
    try:
        # Count total lines for progress bar
        total_lines = sum(1 for _ in open(filtered_output_path, 'r', encoding='utf-8'))
        
        with open(filtered_output_path, 'r', encoding='utf-8') as infile:
            for line in tqdm(infile, total=total_lines, desc="Separating by reason and dataset"):
                try:
                    data = json.loads(line.strip())
                    reasons = data.get('_filter_reasons', ['unknown'])
                    dataset_name = data.get('_dataset_name', 'Unknown')
                    
                    # Separate by filter reason
                    for reason in reasons:
                        if reason not in reason_files:
                            reason_path = os.path.join(separate_dir, f"{reason}.jsonl")
                            reason_files[reason] = open(reason_path, 'w', encoding='utf-8')
                            reason_stats[reason] = 0
                        
                        reason_files[reason].write(line)
                        reason_stats[reason] += 1
                    
                    # Separate by dataset
                    if dataset_name not in dataset_files:
                        dataset_path = os.path.join(dataset_dir, f"{dataset_name.replace('/', '_')}.jsonl")
                        dataset_files[dataset_name] = open(dataset_path, 'w', encoding='utf-8')
                        dataset_stats[dataset_name] = 0
                    
                    dataset_files[dataset_name].write(line)
                    dataset_stats[dataset_name] += 1
                        
                except json.JSONDecodeError:
                    continue
    finally:
        # Close all files
        for f in reason_files.values():
            f.close()
        for f in dataset_files.values():
            f.close()
    
    # Print stats
    print(f"\nSeparate filter files created in: {separate_dir}")
    for reason, count in sorted(reason_stats.items()):
        print(f"  {reason}.jsonl: {count} samples")
    
    print(f"\nDataset-specific filtered files created in: {dataset_dir}")
    for dataset, count in sorted(dataset_stats.items()):
        print(f"  {dataset.replace('/', '_')}.jsonl: {count} samples")


def main():
    parser = argparse.ArgumentParser(description='Filter Zebra-CoT dataset')
    parser.add_argument('input_file', nargs='?', help='Input JSONL file path', default='/dev/shm/data/Zebra-CoT/zebra_cot.jsonl')
    parser.add_argument('output_file', nargs='?', help='Output filtered JSONL file path (good data)', default='/dev/shm/data/Zebra-CoT/zebra_cot_filtered.jsonl')
    parser.add_argument('--filtered-output', help='Output file for filtered data with reasons', default='/dev/shm/data/Zebra-CoT/zebra_cot_filtered_with_reasons.jsonl')
    parser.add_argument('--separate-by-reason', action='store_true', 
                       help='Create separate files for each filter reason and dataset')
    parser.add_argument('--image-base-dir', help='Base directory for images (to check existence)', default='/dev/shm/data/Zebra-CoT')
    parser.add_argument('--max-text-length', type=int, default=2500, 
                       help='Maximum text length in tokens (default: 2500)')
    parser.add_argument('--tokenizer-path', help='Path to tokenizer for accurate token counting')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Load tokenizer if provided
    tokenizer = None
    if args.tokenizer_path:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
            print(f"Loaded tokenizer from: {args.tokenizer_path}")
        except Exception as e:
            print(f"Warning: Could not load tokenizer from {args.tokenizer_path}: {e}")
            print("Falling back to character-based estimation")
    
    print(f"Filtering {args.input_file} -> {args.output_file}")
    if args.filtered_output:
        print(f"Filtered data will be saved to: {args.filtered_output}")
    print(f"Max text length: {args.max_text_length} {'tokens' if tokenizer else 'estimated tokens'}")
    if args.image_base_dir:
        print(f"Image base directory: {args.image_base_dir}")
    
    stats = filter_jsonl_file(
        args.input_file, 
        args.output_file, 
        args.image_base_dir,
        args.max_text_length,
        args.verbose,
        tokenizer,
        args.filtered_output
    )
    
    print("\n=== Overall Filtering Statistics ===")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Kept samples: {stats['kept_samples']}")
    print(f"Filtered samples: {stats['filtered_samples']}")
    print(f"Filter rate: {stats['filtered_samples']/stats['total_samples']*100:.1f}%")
    
    print("\n=== Overall Filter Reasons ===")
    for reason, count in stats['filter_reasons'].items():
        if count > 0:
            print(f"{reason}: {count}")
    
    # Print per-dataset statistics
    print("\n=== Dataset Statistics (After Filtering) ===")
    dataset_stats = stats['dataset_stats']
    for dataset_name in sorted(dataset_stats.keys()):
        ds = dataset_stats[dataset_name]
        filter_rate = ds['filtered'] / ds['total'] * 100 if ds['total'] > 0 else 0
        print(f"\n{dataset_name}:")
        print(f"  Total: {ds['total']}")
        print(f"  Kept: {ds['kept']}")
        print(f"  Filtered: {ds['filtered']} ({filter_rate:.1f}%)")
        if ds['filter_reasons']:
            print("  Filter reasons:")
            for reason, count in sorted(ds['filter_reasons'].items()):
                if count > 0:
                    print(f"    {reason}: {count}")
    
    # Create separate files by reason if requested
    if args.separate_by_reason and args.filtered_output:
        create_separate_filter_files(args.filtered_output)


if __name__ == "__main__":
    main()
