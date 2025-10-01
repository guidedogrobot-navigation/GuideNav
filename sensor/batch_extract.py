
## python3 batch_processor.py /path/to/your/bags your_extractor_script.py
# python3 batch_processor.py /media/2t/ijrr/bags_tactile_day1 your_script.py --continue-on-error


#!/usr/bin/env python3
"""
Batch processor for ROS2 bag data extraction.
Processes all subdirectories containing metadata.yaml and .db3 files.
"""

import os
import sys
import subprocess
import glob
import argparse
from pathlib import Path
import time
import logging

def setup_logging(log_file=None):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)
    
    return logging.getLogger(__name__)

def find_ros2_bag_directories(root_dir):
    """
    Find all subdirectories that contain both metadata.yaml and at least one .db3 file
    
    Args:
        root_dir (str): Root directory to search
        
    Returns:
        list: List of valid bag directories
    """
    bag_dirs = []
    
    for root, dirs, files in os.walk(root_dir):
        has_metadata = 'metadata.yaml' in files
        has_db3 = any(f.endswith('.db3') for f in files)
        
        if has_metadata and has_db3:
            bag_dirs.append(root)
            # Don't search subdirectories of valid bag directories
            dirs.clear()
    
    return sorted(bag_dirs)

def get_bag_duration(bag_dir):
    """
    Estimate bag duration by running ros2 bag info
    
    Args:
        bag_dir (str): Path to the bag directory
        
    Returns:
        float: Estimated duration in seconds, or None if unable to determine
    """
    try:
        result = subprocess.run([
            'ros2', 'bag', 'info', bag_dir
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            # Parse duration from output (format: "Duration: Xs")  
            for line in result.stdout.split('\n'):
                if 'Duration:' in line:
                    # Extract number before 's'
                    duration_str = line.split('Duration:')[1].strip().rstrip('s')
                    try:
                        return float(duration_str)
                    except ValueError:
                        pass
    except Exception:
        pass
    
    return None

def run_ros2_play_and_extract(bag_dir, extractor_script, logger, dry_run=False, timeout_multiplier=3.0):
    """
    Run ros2 bag play and the extraction script for a single bag directory
    
    Args:
        bag_dir (str): Path to the bag directory
        extractor_script (str): Path to the extraction script
        logger: Logger instance
        dry_run (bool): If True, only print what would be done
        timeout_multiplier (float): Multiply bag duration by this for timeout
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Processing bag directory: {bag_dir}")
    
    if dry_run:
        logger.info(f"[DRY RUN] Would process: {bag_dir}")
        return True
    
    # Estimate bag duration for smart timeout
    bag_duration = get_bag_duration(bag_dir)
    if bag_duration:
        timeout = max(300, bag_duration * timeout_multiplier)  # Min 5 minutes
        logger.info(f"Bag duration: {bag_duration:.1f}s, using timeout: {timeout:.1f}s")
    else:
        timeout = 1800  # Default 30 minutes for unknown duration
        logger.info(f"Could not determine bag duration, using default timeout: {timeout}s")
    
    try:
        # Start the extraction node in the background
        logger.info("Starting extraction node...")
        extractor_process = subprocess.Popen([
            'python3', extractor_script
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Give the node time to initialize
        time.sleep(3)
        
        # Start bag playback
        logger.info(f"Playing bag from: {bag_dir}")
        start_time = time.time()
        
        play_process = subprocess.run([
            'ros2', 'bag', 'play', bag_dir, '--clock'
        ], capture_output=True, text=True, timeout=timeout)
        
        elapsed = time.time() - start_time
        logger.info(f"Bag playback completed in {elapsed:.1f}s")
        
        if play_process.returncode != 0:
            logger.error(f"ros2 bag play failed (exit code {play_process.returncode})")
            if play_process.stderr:
                logger.error(f"Error output: {play_process.stderr.strip()}")
            extractor_process.terminate()
            return False
        
        # Give extraction node time to finish processing
        logger.info("Waiting for extraction node to finish...")
        time.sleep(5)
        
        # Terminate the extraction node gracefully
        extractor_process.terminate()
        try:
            stdout, stderr = extractor_process.communicate(timeout=10)
            if stdout:
                logger.info(f"Extractor output: {stdout.strip()}")
            if stderr:
                logger.warning(f"Extractor errors: {stderr.strip()}")
        except subprocess.TimeoutExpired:
            logger.warning("Extraction node didn't terminate gracefully, killing it")
            extractor_process.kill()
            
        logger.info(f"Successfully processed: {bag_dir}")
        return True
        
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout ({timeout}s) while processing: {bag_dir}")
        if 'extractor_process' in locals():
            extractor_process.kill()
        return False
    except Exception as e:
        logger.error(f"Error processing {bag_dir}: {e}")
        if 'extractor_process' in locals():
            extractor_process.kill()
        return False

def main():
    parser = argparse.ArgumentParser(description='Batch process ROS2 bags for data extraction')
    parser.add_argument('input_dir', help='Root directory containing ROS2 bag subdirectories')
    parser.add_argument('extractor_script', help='Path to the ROS2 data extraction script')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be processed without actually doing it')
    parser.add_argument('--log-file', help='Log file path (optional)')
    parser.add_argument('--continue-on-error', action='store_true',
                       help='Continue processing other bags if one fails')
    parser.add_argument('--timeout-multiplier', type=float, default=3.0,
                       help='Multiply bag duration by this factor for timeout (default: 3.0)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        sys.exit(1)
    
    if not os.path.isfile(args.extractor_script):
        print(f"Error: Extractor script '{args.extractor_script}' does not exist")
        sys.exit(1)
    
    # Setup logging
    logger = setup_logging(args.log_file)
    
    logger.info("=== ROS2 Bag Batch Processor Started ===")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Extractor script: {args.extractor_script}")
    logger.info(f"Dry run: {args.dry_run}")
    
    # Find all valid bag directories
    logger.info("Searching for ROS2 bag directories...")
    bag_directories = find_ros2_bag_directories(args.input_dir)
    
    if not bag_directories:
        logger.warning("No valid ROS2 bag directories found!")
        logger.info("Looking for directories containing both 'metadata.yaml' and '*.db3' files")
        sys.exit(0)
    
    logger.info(f"Found {len(bag_directories)} bag directories to process:")
    for i, bag_dir in enumerate(bag_directories, 1):
        logger.info(f"  {i}. {bag_dir}")
    
    if args.dry_run:
        logger.info("Dry run completed. Use --dry-run=false to actually process the bags.")
        sys.exit(0)
    
    # Process each bag directory
    successful = 0
    failed = 0
    
    for i, bag_dir in enumerate(bag_directories, 1):
        logger.info(f"\n--- Processing {i}/{len(bag_directories)} ---")
        
        success = run_ros2_play_and_extract(bag_dir, args.extractor_script, logger, 
                                           args.dry_run, args.timeout_multiplier)
        
        if success:
            successful += 1
        else:
            failed += 1
            if not args.continue_on_error:
                logger.error("Stopping due to error. Use --continue-on-error to process remaining bags.")
                break
    
    # Final summary
    logger.info("\n=== Processing Summary ===")
    logger.info(f"Total directories found: {len(bag_directories)}")
    logger.info(f"Successfully processed: {successful}")
    logger.info(f"Failed: {failed}")
    
    if failed > 0:
        logger.warning("Some bags failed to process. Check the logs above for details.")
        sys.exit(1)
    else:
        logger.info("All bags processed successfully!")

if __name__ == '__main__':
    main()