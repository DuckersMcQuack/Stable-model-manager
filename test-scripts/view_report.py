#!/usr/bin/env python3
"""
Report Viewer Utility
View processing reports in a readable format
"""

import argparse
import json
import os
from datetime import datetime


def view_report(report_file: str, show_moves_only: bool = False):
    """View a processing report"""
    
    if not os.path.exists(report_file):
        print(f"Report file not found: {report_file}")
        return
    
    try:
        with open(report_file, 'r') as f:
            data = json.load(f)
        
        print("="*80)
        print("MODEL SORTER PROCESSING REPORT")
        print("="*80)
        
        # Basic info
        print(f"Timestamp: {data['timestamp']}")
        print(f"Duration: {data['duration_seconds']:.2f} seconds")
        print(f"Mode: {'DRY RUN' if data['dry_run'] else 'LIVE PROCESSING'}")
        print(f"Config: {data['config_file']}")
        
        # Stats
        print(f"\nSTATISTICS:")
        stats = data['stats']
        for key, value in stats.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        
        # Console output
        if 'console_output' in data:
            print(f"\nCONSOLE OUTPUT ({len(data['console_output'])} lines):")
            print("-" * 60)
            
            if show_moves_only:
                print("SHOWING ONLY FILE MOVEMENTS:")
                move_lines = []
                for line in data['console_output']:
                    if any(phrase in line for phrase in [
                        'DRY RUN: Would move', 'Moved:', 'Target already exists', 
                        'moving to duplicates', 'Skipping', 'Processing model:'
                    ]):
                        move_lines.append(line)
                
                if move_lines:
                    for line in move_lines:
                        print(line)
                else:
                    print("No file movements found in this report.")
            else:
                # Show full console output
                for line in data['console_output']:
                    print(line)
        
        print("="*80)
        
    except Exception as e:
        print(f"Error reading report: {e}")


def list_reports():
    """List all available reports"""
    report_dir = "processing_report"
    if not os.path.exists(report_dir):
        print("No processing reports found (processing_report directory doesn't exist).")
        return
    
    reports = [f for f in os.listdir(report_dir) if f.startswith('processing_report_') and f.endswith('.json')]
    reports.sort(reverse=True)  # Most recent first
    
    if not reports:
        print("No processing reports found.")
        return
    
    print("AVAILABLE REPORTS:")
    print("-" * 50)
    for i, report in enumerate(reports, 1):
        # Extract timestamp from filename
        try:
            timestamp_str = report.replace('processing_report_', '').replace('.json', '')
            timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
            readable_time = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            print(f"{i}. {report} ({readable_time})")
        except:
            print(f"{i}. {report}")


def main():
    parser = argparse.ArgumentParser(description='View Model Sorter processing reports')
    parser.add_argument('--list', action='store_true', help='List all available reports')
    parser.add_argument('--file', '-f', type=str, help='Report file to view (from processing_report/ directory, or full path)')
    parser.add_argument('--latest', action='store_true', help='View the latest report')
    parser.add_argument('--moves-only', action='store_true', 
                       help='Show only file movements (useful for dry runs)')
    
    args = parser.parse_args()
    
    if args.list:
        list_reports()
        return
    
    report_file = None
    
    if args.latest:
        # Find the most recent report
        report_dir = "processing_report"
        if not os.path.exists(report_dir):
            print("No reports found (processing_report directory doesn't exist).")
            return
        
        reports = [f for f in os.listdir(report_dir) if f.startswith('processing_report_') and f.endswith('.json')]
        if reports:
            report_file = os.path.join(report_dir, sorted(reports, reverse=True)[0])
        else:
            print("No reports found.")
            return
    elif args.file:
        # Check if file path already includes directory, if not prepend processing_report/
        if os.path.dirname(args.file) == "":
            report_file = os.path.join("processing_report", args.file)
        else:
            report_file = args.file
    else:
        print("Please specify --file, --latest, or --list")
        return
    
    view_report(report_file, args.moves_only)


if __name__ == "__main__":
    main()