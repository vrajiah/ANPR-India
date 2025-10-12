#!/usr/bin/env python3
"""
ANPR System Command Line Interface
Enhanced CLI with better UX and configuration options
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import pandas as pd
from tqdm import tqdm

from .core import ANPRSystem


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="🚗 ANPR System - Automatic Number Plate Recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  anpr --input video.mp4 --output result.mp4 --csv results.csv

  # High accuracy processing
  anpr --input video.mp4 --output result.mp4 --frame-skip 1 --min-conf 0.3

  # Fast processing
  anpr --input video.mp4 --output result.mp4 --frame-skip 10 --min-conf 0.7

  # Process with custom model
  anpr --weight custom_model.pt --input video.mp4 --output result.mp4

  # Web interface
  anpr-web
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input video file path'
    )
    
    parser.add_argument(
        '--output', '-o', 
        required=True,
        help='Output video file path'
    )
    
    parser.add_argument(
        '--csv', '-c',
        required=True,
        help='Output CSV file path for detected plates'
    )
    
    # Optional arguments
    parser.add_argument(
        '--weight', '-w',
        default='runs/train/exp/weights/best.pt',
        help='Path to YOLOv5 model weights (default: runs/train/exp/weights/best.pt)'
    )
    
    parser.add_argument(
        '--device', '-d',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        default='auto',
        help='Device to run on (default: auto)'
    )
    
    parser.add_argument(
        '--frame-skip', '-f',
        type=int,
        default=3,
        help='Process every Nth frame (default: 3, higher = faster)'
    )
    
    parser.add_argument(
        '--min-conf', '-m',
        type=float,
        default=0.2,
        help='Minimum confidence threshold (default: 0.2)'
    )
    
    parser.add_argument(
        '--region-threshold', '-r',
        type=float,
        default=0.6,
        help='Region size threshold for OCR (default: 0.6)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress all output except errors'
    )
    
    parser.add_argument(
        '--preview',
        action='store_true',
        help='Show preview window during processing'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show processing statistics'
    )
    
    return parser


def validate_inputs(args) -> bool:
    """Validate input arguments"""
    # Check input file exists
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file '{args.input}' not found")
        return False
    
    # Check model weights exist
    if not os.path.exists(args.weight):
        print(f"❌ Error: Model weights '{args.weight}' not found")
        return False
    
    # Check output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    csv_dir = os.path.dirname(args.csv)
    if csv_dir and not os.path.exists(csv_dir):
        os.makedirs(csv_dir, exist_ok=True)
    
    return True


def print_banner():
    """Print application banner"""
    print("🚗" + "="*50 + "🚗")
    print("   ANPR System - License Plate Recognition")
    print("   Advanced Indian License Plate Detection")
    print("🚗" + "="*50 + "🚗")
    print()


def print_config(args):
    """Print configuration summary"""
    print("📋 Configuration:")
    print(f"   Input: {args.input}")
    print(f"   Output: {args.output}")
    print(f"   CSV: {args.csv}")
    print(f"   Model: {args.weight}")
    print(f"   Device: {args.device}")
    print(f"   Frame Skip: {args.frame_skip}")
    print(f"   Min Confidence: {args.min_conf}")
    print(f"   Region Threshold: {args.region_threshold}")
    print()


def process_video(args) -> bool:
    """Process video with ANPR system"""
    try:
        # Initialize ANPR system
        if not args.quiet:
            print("🔧 Initializing ANPR System...")
        
        device = None if args.device == 'auto' else args.device
        anpr = ANPRSystem(args.weight, device)
        
        # Open video
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            print(f"❌ Error: Cannot open video file '{args.input}'")
            return False
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if not args.quiet:
            print(f"📹 Video Info: {width}x{height} @ {fps}fps, {total_frames} frames")
            print(f"⏱️  Estimated time: {total_frames // (fps * args.frame_skip):.1f} seconds")
            print()
        
        # Setup output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        
        # Processing loop
        frame_count = 0
        processed_frames = 0
        start_time = time.time()
        
        if not args.quiet:
            pbar = tqdm(total=total_frames, desc="Processing", unit="frame")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames for faster processing
            if frame_count % args.frame_skip != 0:
                out.write(frame)
                if not args.quiet:
                    pbar.update(1)
                continue
            
            processed_frames += 1
            
            # Process frame
            annotated_frame, detections = anpr.process_frame(frame)
            
            # Write frame
            out.write(annotated_frame)
            
            # Show preview if requested
            if args.preview:
                cv2.imshow('ANPR Preview', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            if not args.quiet:
                pbar.update(1)
                if args.verbose and detections:
                    for det in detections:
                        # Use tqdm's write() to avoid breaking progress bar
                        pbar.write(f"   Detected: {det['plate_text']} (conf: {det['confidence']:.3f})")
        
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        if not args.quiet:
            pbar.close()
        
        # Save results to CSV
        anpr.df = anpr.df[anpr.df['conf'] > args.min_conf]
        anpr.df.to_csv(args.csv)
        
        # Print statistics
        if args.stats or not args.quiet:
            end_time = time.time()
            processing_time = end_time - start_time
            
            print("\n📊 Processing Complete!")
            print(f"   Total frames: {frame_count}")
            print(f"   Processed frames: {processed_frames}")
            print(f"   Processing time: {processing_time:.1f}s")
            print(f"   FPS: {processed_frames/processing_time:.1f}")
            print(f"   Detected plates: {len(anpr.df)}")
            print(f"   Output video: {args.output}")
            print(f"   Results CSV: {args.csv}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        return False


def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Print banner
    if not args.quiet:
        print_banner()
    
    # Validate inputs
    if not validate_inputs(args):
        sys.exit(1)
    
    # Print configuration
    if not args.quiet:
        print_config(args)
    
    # Process video
    success = process_video(args)
    
    if success:
        if not args.quiet:
            print("\n✅ Processing completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Processing failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
