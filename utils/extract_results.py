#!/usr/bin/env python3
"""
Script to extract results from debates subdirectory.
Extracts the first two and last three JSON lines from each "debate_main.log" file
and saves them to a new JSONL file called "all_debates_review.jsonl".
"""

import os
import json
import glob
from pathlib import Path
from typing import List, Dict, Any


def is_valid_json_line(line: str) -> bool:
    """Check if a line contains valid JSON."""
    try:
        json.loads(line.strip())
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def extract_json_lines_from_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract first two and last three JSON lines from a debate_main.log file.
    
    Args:
        file_path: Path to the debate_main.log file
        
    Returns:
        List of parsed JSON objects with metadata about their position
    """
    results = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Filter only valid JSON lines
        json_lines = []
        for i, line in enumerate(lines):
            line = line.strip()
            if line and is_valid_json_line(line):
                json_lines.append((i + 1, line))  # Store line number and content
        
        if not json_lines:
            print(f"Warning: No valid JSON lines found in {file_path}")
            return results
        
        # Extract first two JSON lines
        for i in range(min(2, len(json_lines))):
            line_num, line_content = json_lines[i]
            try:
                json_data = json.loads(line_content)
                json_data['_meta'] = {
                    'source_file': file_path,
                    'line_number': line_num,
                    'position': f'first_{i+1}'
                }
                results.append(json_data)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON in {file_path} at line {line_num}: {e}")
        
        # Extract last three JSON lines (if different from first ones)
        if len(json_lines) > 2:
            for i in range(max(0, len(json_lines) - 3), len(json_lines)):
                line_num, line_content = json_lines[i]
                try:
                    json_data = json.loads(line_content)
                    json_data['_meta'] = {
                        'source_file': file_path,
                        'line_number': line_num,
                        'position': f'last_{len(json_lines) - i}'
                    }
                    results.append(json_data)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON in {file_path} at line {line_num}: {e}")
        
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    
    return results


def find_debate_log_files(base_directory: str = "debates") -> List[str]:
    """
    Find all debate_main.log files in subdirectories.
    
    Args:
        base_directory: Base directory to search for debate logs
        
    Returns:
        List of paths to debate_main.log files
    """
    log_files = []
    
    if not os.path.exists(base_directory):
        print(f"Warning: Base directory '{base_directory}' does not exist")
        return log_files
    
    # Use glob to find all debate_main.log files recursively
    pattern = os.path.join(base_directory, "**/debate_main.log")
    log_files = glob.glob(pattern, recursive=True)
    
    print(f"Found {len(log_files)} debate_main.log files")
    return log_files


def main():
    """Main function to extract and consolidate debate results."""
    print("Starting debate results extraction...")
    
    # Find all debate log files
    log_files = find_debate_log_files()
    
    if not log_files:
        print("No debate_main.log files found. Exiting.")
        return
    
    all_results = []
    
    # Write consolidated results to JSONL file with separators
    output_file = "all_debates_review.jsonl"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            total_entries = 0
            
            # Process each log file
            for file_index, log_file in enumerate(log_files):
                print(f"Processing: {log_file}")
                extracted_data = extract_json_lines_from_file(log_file)
                
                if extracted_data:
                    # Extract chat_id from file path if possible
                    chat_id = "unknown"
                    try:
                        # Try to extract chat_id from path like: debates/topic_id/helper_type/chat_id/debate_main.log
                        path_parts = log_file.replace('\\', '/').split('/')
                        if len(path_parts) >= 2:
                            chat_id = path_parts[-2]  # The directory name before the filename
                    except:
                        pass
                    
                    # Add separator with file info (as a comment-style line)
                    f.write(f"# === CHAT SESSION {file_index + 1} | CHAT_ID: {chat_id} | FILE: {log_file} ===\n")
                    
                    # Write the JSON entries for this file
                    for result in extracted_data:
                        f.write(json.dumps(result) + '\n')
                        total_entries += 1
                    
                    # Add empty line for visual separation
                    f.write('\n')
                    
                    all_results.extend(extracted_data)
                else:
                    print(f"Warning: No valid JSON found in {log_file}")
        
        print(f"\nExtraction complete!")
        print(f"Extracted {len(all_results)} JSON entries from {len(log_files)} files")
        print(f"Results saved to: {output_file}")
        
        # Print summary statistics
        first_entries = sum(1 for r in all_results if r['_meta']['position'].startswith('first'))
        last_entries = sum(1 for r in all_results if r['_meta']['position'].startswith('last'))
        
        print(f"- First entries: {first_entries}")
        print(f"- Last entries: {last_entries}")
        print(f"- Each group of 2-5 JSON lines represents one debate session")
        print(f"- Groups are separated by comment lines with chat_id information")
        
    except Exception as e:
        print(f"Error writing output file: {e}")


if __name__ == "__main__":
    main()