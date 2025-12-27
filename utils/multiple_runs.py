#!/usr/bin/env python3
"""
Multiple Runs Script for LOGICOM

This script allows running multiple debates with different combinations of:
- Claim indexes 
- Helper types

It calls the existing main.py for each combination, making minimal changes to existing code.
"""

import subprocess
import sys
import argparse
from pathlib import Path
from typing import List, Optional
import time
import yaml
import os
import shutil
from datetime import datetime
import zipfile
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

# #region agent log
log_path = r"c:\Users\gal40\OneDrive - Technion\Legacy_LOGICOM\new_logicom\LOGICOM\.cursor\debug.log"
def debug_log(location, message, data=None, hypothesis_id=None):
    try:
        entry = {"sessionId": "debug-session", "runId": "run1", "location": location, "message": message, "timestamp": int(time.time() * 1000)}
        if hypothesis_id: entry["hypothesisId"] = hypothesis_id
        if data: entry["data"] = data
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except: pass
# #endregion

# Global flag for graceful shutdown
interrupt_requested = False

def signal_handler(signum, frame):
    """Handle interrupt signal (Ctrl+C) gracefully"""
    global interrupt_requested
    # #region agent log
    debug_log("multiple_runs.py:29", "Signal handler called", {"signum": signum, "interrupt_requested_before": interrupt_requested}, "A")
    # #endregion
    if not interrupt_requested:
        print("\n⚠️  Interrupt received! Will stop after current debate finishes...")
        print("   Press Ctrl+C again to force quit immediately.")
        interrupt_requested = True
        # #region agent log
        debug_log("multiple_runs.py:35", "Interrupt flag set to True", {}, "A")
        # #endregion
    else:
        print("\n🛑 Force quit requested!")
        # #region agent log
        debug_log("multiple_runs.py:38", "Force quit - exiting", {}, "A")
        # #endregion
        sys.exit(1)


def main():
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser(description="Run multiple LOGICOM debates with different configurations")
    
    parser.add_argument("--helper_types", nargs="+", 
                       help="List of helper types to run (default: all available)")
    parser.add_argument("--claim_indexes", nargs="+", type=str,
                       help="List of claim indexes to run. Can be individual numbers (e.g., 0 1 2) or ranges (e.g., 0-199). Default: all claims")
    parser.add_argument("--settings_path", default="./config/settings.yaml",
                       help="Path to settings YAML file")
    parser.add_argument("--models_path", default="./config/models.yaml", 
                       help="Path to models YAML file")
    parser.add_argument("--list_helpers", action="store_true",
                       help="List available helper types and exit")
    parser.add_argument("--max_workers", type=int, default=4,
                       help="Maximum number of parallel workers (default: 4, use 1 for sequential)")
    parser.add_argument("--run_name", required=True,
                       help="Name for this run (will be combined with date for results folder)")
    
    args = parser.parse_args()
    
    # List available helpers if requested
    if args.list_helpers:
        available_helpers = get_available_helper_types(args.settings_path)
        print("Available helper types:")
        for helper in available_helpers:
            print(f"  - {helper}")
        return
    
    # Create results directory for this run
    results_dir = create_results_directory(args.run_name)
    
    # Create unique debates folder for this run (inside results directory)
    debates_folder = os.path.join(results_dir, "debates")
    os.makedirs(debates_folder, exist_ok=True)
    print(f"Using debates directory: {debates_folder}")
    
    # Clear any existing summary file to start fresh for this batch
    excel_file = "all_debates_summary.xlsx"
    # #region agent log
    debug_log("multiple_runs.py:81", "Checking Excel file before deletion", {"excel_file": excel_file, "exists": os.path.exists(excel_file)}, "E")
    # #endregion
    if os.path.exists(excel_file):
        os.remove(excel_file)
        # #region agent log
        debug_log("multiple_runs.py:83", "Excel file deleted", {"excel_file": excel_file}, "E")
        # #endregion
        print(f"Cleared existing {excel_file} for fresh start")
    
    # Get helper types to run
    if args.helper_types:
        helper_types = args.helper_types
    else:
        helper_types = get_available_helper_types(args.settings_path)
        print(f"No helper types specified, using all available: {helper_types}")
    
    if not helper_types:
        print("No helper types found or specified!")
        return
    
    # Validate helper types exist
    available_helpers = get_available_helper_types(args.settings_path)
    invalid_helpers = [h for h in helper_types if h not in available_helpers]
    if invalid_helpers:
        print(f"Invalid helper types: {invalid_helpers}")
        print(f"Available helpers: {available_helpers}")
        return
    
    runs = []
    total_runs = 0
    
    # Parse claim indexes (support both individual numbers and ranges like "0-199")
    parsed_claim_indexes = []
    if args.claim_indexes:
        for item in args.claim_indexes:
            if '-' in item and item.count('-') == 1:
                # Range format: "0-199"
                try:
                    start, end = map(int, item.split('-'))
                    parsed_claim_indexes.extend(range(start, end + 1))
                except ValueError:
                    print(f"⚠️  Invalid range format: {item}. Expected format: start-end (e.g., 0-199)")
                    return
            else:
                # Single number
                try:
                    parsed_claim_indexes.append(int(item))
                except ValueError:
                    print(f"⚠️  Invalid claim index: {item}. Must be a number or range (e.g., 0-199)")
                    return
        
        # Remove duplicates and sort
        parsed_claim_indexes = sorted(set(parsed_claim_indexes))
        # #region agent log
        debug_log("multiple_runs.py:137", "Parsed claim indexes", {"input": args.claim_indexes, "parsed_count": len(parsed_claim_indexes), "first": parsed_claim_indexes[0] if parsed_claim_indexes else None, "last": parsed_claim_indexes[-1] if parsed_claim_indexes else None}, "F")
        # #endregion
    
    # Always run sequentially by claim index when claim_indexes are specified
    if parsed_claim_indexes:
        # Run each claim with each helper type
        for claim_index in parsed_claim_indexes:
            for helper_type in helper_types:
                runs.append((helper_type, claim_index))
                total_runs += 1
    else:
        # Run all claims for each helper type
        for helper_type in helper_types:
            runs.append((helper_type, None))  # None means all claims
            total_runs += 1
    
    # #region agent log
    debug_log("multiple_runs.py:165", "Runs created", {"total_runs": total_runs, "num_helper_types": len(helper_types), "claim_indexes_specified": parsed_claim_indexes is not None and len(parsed_claim_indexes) > 0, "num_claim_indexes": len(parsed_claim_indexes) if parsed_claim_indexes else 0, "max_workers": args.max_workers, "effective_parallelism": min(total_runs, args.max_workers)}, "F")
    # #endregion
    
    print(f"\nPlanning to run {total_runs} debate configurations:")
    for i, (helper_type, claim_index) in enumerate(runs, 1):
        claim_desc = f"claim {claim_index}" if claim_index is not None else "all claims"
        print(f"  {i}. {helper_type} - {claim_desc}")
    
    # #region agent log
    debug_log("multiple_runs.py:154", "Parallelism analysis", {"total_runs": total_runs, "max_workers": args.max_workers, "will_run_simultaneously": min(total_runs, args.max_workers), "limited_by_runs": total_runs < args.max_workers}, "F")
    # #endregion
    
    print(f"\nStarting runs...")
    
    # Execute runs
    successful_runs = 0
    failed_runs = 0
    start_time = time.time()
    
    # Always use parallel execution (use --max_workers 1 for sequential)
    effective_workers = min(total_runs, args.max_workers)
    print(f"Running with {args.max_workers} worker{'s' if args.max_workers > 1 else ''} (will run {effective_workers} simultaneously)")
    if total_runs < args.max_workers:
        print(f"⚠️  Note: Only {total_runs} run{'s' if total_runs == 1 else ''} created, so parallelism is limited to {total_runs} worker{'s' if total_runs == 1 else ''}")
    
    # #region agent log
    debug_log("multiple_runs.py:170", "Creating ProcessPoolExecutor", {"max_workers": args.max_workers, "total_runs": total_runs, "effective_workers": effective_workers}, "F")
    # #endregion
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all jobs
        future_to_run = {}
        for i, (helper_type, claim_index) in enumerate(runs, 1):
            # #region agent log
            debug_log("multiple_runs.py:140", "Submitting job", {"job_num": i, "helper_type": helper_type, "claim_index": claim_index}, "A")
            # #endregion
            future = executor.submit(
                run_single_debate,
                helper_type,
                claim_index,
                args.settings_path,
                args.models_path,
                debates_folder
            )
            future_to_run[future] = (i, helper_type, claim_index)
        
        # Process results as they complete
        for future in as_completed(future_to_run):
            i, helper_type, claim_index = future_to_run[future]
            
            # #region agent log
            debug_log("multiple_runs.py:154", "Checking interrupt flag", {"interrupt_requested": interrupt_requested}, "A")
            # #endregion
            if interrupt_requested:
                print(f"\n🛑 Interrupt requested, cancelling remaining jobs...")
                # #region agent log
                debug_log("multiple_runs.py:156", "Shutting down executor", {"wait": False, "cancel_futures": True}, "A")
                # #endregion
                executor.shutdown(wait=False, cancel_futures=True)
                break
            
            try:
                # #region agent log
                debug_log("multiple_runs.py:160", "Getting future result", {"helper_type": helper_type, "claim_index": claim_index}, "B")
                # #endregion
                success = future.result() # Get the return value from run_single_debate()
                # #region agent log
                debug_log("multiple_runs.py:160", "Future result received", {"success": success, "helper_type": helper_type, "claim_index": claim_index}, "B")
                # #endregion
                completed = successful_runs + failed_runs + 1
                progress_pct = (completed / total_runs) * 100
                
                if success:
                    successful_runs += 1
                    status_icon = "✓"
                    status_text = "Completed"
                else:
                    failed_runs += 1
                    status_icon = "✗"
                    status_text = "Failed"
                
                claim_desc = f" claim {claim_index}" if claim_index is not None else " all claims"
                print(f"{status_icon} [{completed}/{total_runs}] ({progress_pct:.1f}%) {status_text}: {helper_type}{claim_desc}")
                
            except Exception as e:
                # #region agent log
                debug_log("multiple_runs.py:176", "Exception in future.result()", {"error": str(e), "helper_type": helper_type, "claim_index": claim_index}, "B")
                # #endregion
                failed_runs += 1
                completed = successful_runs + failed_runs
                progress_pct = (completed / total_runs) * 100
                claim_desc = f" claim {claim_index}" if claim_index is not None else " all claims"
                print(f"✗ [{completed}/{total_runs}] ({progress_pct:.1f}%) Exception: {helper_type}{claim_desc}: {e}")
    
    # Summary
    elapsed_time = time.time() - start_time
    completed_runs = successful_runs + failed_runs
    
    print(f"\n" + "="*50)
    print(f"SUMMARY")
    print(f"="*50)
    print(f"Planned runs: {total_runs}")
    print(f"Completed runs: {completed_runs}")
    if interrupt_requested:
        print(f"⚠️  Run interrupted by user ({total_runs - completed_runs} runs skipped)")
    print(f"Successful: {successful_runs}")
    print(f"Failed: {failed_runs}")
    print(f"Time elapsed: {elapsed_time:.1f} seconds")
    
    time.sleep(3) # Make sure all processes are finished
    
    # Move Excel results to results directory
    excel_file = "all_debates_summary.xlsx"
    lock_file = "all_debates_summary.xlsx.lock"
    # #region agent log
    debug_log("multiple_runs.py:201", "Checking Excel file before move", {"excel_file": excel_file, "exists": os.path.exists(excel_file), "lock_file": lock_file, "lock_exists": os.path.exists(lock_file)}, "D")
    # #endregion
    if os.path.exists(excel_file):
        dest_excel = os.path.join(results_dir, "all_debates_summary.xlsx")
        try:
            shutil.move(excel_file, dest_excel)
            # #region agent log
            debug_log("multiple_runs.py:205", "Excel file moved successfully", {"dest": dest_excel}, "D")
            # #endregion
            print(f"✓ Moved Excel results to: {dest_excel}")
        except Exception as e:
            # #region agent log
            debug_log("multiple_runs.py:208", "Failed to move Excel file", {"error": str(e)}, "D")
            # #endregion
            print(f"✗ Failed to move Excel file: {e}")
    else:
        # #region agent log
        debug_log("multiple_runs.py:210", "Excel file not found", {"excel_file": excel_file}, "D")
        # #endregion
        print(f"✗ Excel file not found: {excel_file}")
    
    # Clean up lock file if it exists
    if os.path.exists(lock_file):
        try:
            os.remove(lock_file)
            # #region agent log
            debug_log("multiple_runs.py:217", "Lock file cleaned up", {"lock_file": lock_file}, "D")
            # #endregion
        except Exception as e:
            # #region agent log
            debug_log("multiple_runs.py:220", "Failed to remove lock file", {"error": str(e)}, "D")
            # #endregion
            print(f"⚠️  Warning: Could not remove lock file {lock_file}: {e}")
    
    # Save copy of settings.yaml for reproducibility
    if os.path.exists(args.settings_path):
        dest_settings = os.path.join(results_dir, "settings.yaml")
        try:
            shutil.copy2(args.settings_path, dest_settings)
            print(f"✓ Saved settings copy to: {dest_settings}")
        except Exception as e:
            print(f"✗ Failed to copy settings file: {e}")
    else:
        print(f"✗ Settings file not found: {args.settings_path}")
    # Create ZIP archive of prompts folder (for reproducibility)
    prompts_folder = "prompts"
    if os.path.exists(prompts_folder):
        prompts_zip = os.path.join(results_dir, "prompts.zip")
        try:
            with zipfile.ZipFile(prompts_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(prompts_folder):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # Store with relative path (e.g., prompts/debater/file.txt)
                        arcname = os.path.relpath(file_path, '.')
                        zipf.write(file_path, arcname)
            print(f"✓ Created prompts archive: {prompts_zip}")
        except Exception as e:
            print(f"✗ Failed to create prompts archive: {e}")
    else:
        print(f"✗ Prompts folder not found: {prompts_folder}")
    
    # Debates are already saved directly in results_dir/debates/ - no need to move!
    print(f"✓ Debates saved in: {debates_folder}")
    
    print(f"\n📁 Results saved in: {results_dir}")
    
    # Run analysis script automatically
    excel_file_path = os.path.join(results_dir, "all_debates_summary.xlsx")
    if os.path.exists(excel_file_path):
        print(f"\n{'='*50}")
        print(f"Running analysis script...")
        print(f"{'='*50}")
        try:
            analysis_output_dir = os.path.join(results_dir, "analysis")
            analysis_script = os.path.join("utils", "analyze_debate_results.py")
            cmd = [sys.executable, analysis_script, excel_file_path, analysis_output_dir]
            result = subprocess.run(cmd, capture_output=False, text=True, cwd=Path.cwd())
            if result.returncode == 0:
                print(f"✓ Analysis completed successfully!")
                print(f"  Analysis results saved to: {analysis_output_dir}")
            else:
                print(f"✗ Analysis script failed with return code: {result.returncode}")
        except Exception as e:
            print(f"✗ Failed to run analysis script: {e}")
    else:
        print(f"⚠ Excel file not found at {excel_file_path}, skipping analysis")
    
    if failed_runs > 0:
        sys.exit(1)
    else:
        print("All runs completed successfully!")

def load_config(settings_path: str = "./config/settings.yaml") -> dict:
    """Load settings to get available helper types"""
    try:
        with open(settings_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}

def get_available_helper_types(settings_path: str = "./config/settings.yaml") -> List[str]:
    """Get list of available helper types from settings"""
    config = load_config(settings_path)
    agent_configs = config.get('agent_configurations', {})
    return list(agent_configs.keys())

def create_results_directory(run_name: str) -> str:
    """
    Create a timestamped results directory.
    
    Args:
        run_name: Custom name for this run
        
    Returns:
        str: Path to the created results directory
    """
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directory name
    dir_name = f"{timestamp}_{run_name}"
    
    # Create results base directory if it doesn't exist
    results_base = "results"
    if not os.path.exists(results_base):
        os.makedirs(results_base)
        print(f"Created results directory: {results_base}")
    
    # Create the specific run directory
    results_dir = os.path.join(results_base, dir_name)
    os.makedirs(results_dir, exist_ok=True)
    print(f"Created results directory: {results_dir}")
    
    return results_dir

def run_single_debate(helper_type: str, claim_index: Optional[int] = None, 
                     settings_path: str = "./config/settings.yaml",
                     models_path: str = "./config/models.yaml",
                     debates_dir: str = "debates") -> bool:
    """
    Run a single debate by calling main.py
    
    Args:
        helper_type: Type of helper configuration to use
        claim_index: Specific claim to run (None for all claims)
        settings_path: Path to settings.yaml
        models_path: Path to models.yaml
        debates_dir: Directory where debate logs should be saved
    
    Returns True if successful, False if failed
    """
    # #region agent log
    debug_log("multiple_runs.py:315", "run_single_debate entry", {"helper_type": helper_type, "claim_index": claim_index, "debates_dir": debates_dir}, "C")
    # #endregion
    cmd = [sys.executable, "main.py", "--helper_type", helper_type]
    
    if claim_index is not None:
        cmd.extend(["--claim_index", str(claim_index)])
    
    if settings_path != "./config/settings.yaml":
        cmd.extend(["--settings_path", settings_path])
        
    if models_path != "./config/models.yaml":
        cmd.extend(["--models_path", models_path])
    
    # Always pass the debates directory
    cmd.extend(["--debates_dir", debates_dir])
    
    # #region agent log
    debug_log("multiple_runs.py:345", "About to run subprocess", {"cmd": " ".join(cmd)}, "C")
    # #endregion
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
        
        # #region agent log
        debug_log("multiple_runs.py:348", "Subprocess completed", {"returncode": result.returncode, "stdout_length": len(result.stdout), "stderr_length": len(result.stderr)}, "C")
        # #endregion
        
        if result.returncode == 0:
            # #region agent log
            debug_log("multiple_runs.py:350", "Subprocess success", {"helper_type": helper_type, "claim_index": claim_index}, "C")
            # #endregion
            print(f"✓ Success: {helper_type}" + (f" claim {claim_index}" if claim_index is not None else " all claims"))
            return True
        else:
            # #region agent log
            debug_log("multiple_runs.py:354", "Subprocess failed", {"returncode": result.returncode, "stderr_preview": result.stderr[:200] if result.stderr else None, "stdout_preview": result.stdout[:200] if result.stdout else None}, "B")
            # #endregion
            print(f"✗ Failed: {helper_type}" + (f" claim {claim_index}" if claim_index is not None else " all claims"))
            print(f"Error: {result.stderr}")
            if result.stdout:
                print(f"Stdout: {result.stdout[:500]}")  # Print first 500 chars of stdout for debugging
            return False
            
    except Exception as e:
        # #region agent log
        debug_log("multiple_runs.py:358", "Exception in run_single_debate", {"error": str(e), "helper_type": helper_type, "claim_index": claim_index}, "B")
        # #endregion
        print(f"✗ Exception running {helper_type}" + (f" claim {claim_index}" if claim_index is not None else " all claims") + f": {e}")
        return False
if __name__ == "__main__":
    main() 