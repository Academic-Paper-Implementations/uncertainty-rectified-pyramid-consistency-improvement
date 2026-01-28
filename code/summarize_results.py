"""
Script tổng hợp kết quả từ các lần chạy thí nghiệm
Author: KhangPX
"""
import argparse
import os
import csv
import numpy as np
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, required=True,
                    help='Path to input CSV file with all runs')
parser.add_argument('--output_csv', type=str, default='',
                    help='Path to save summary CSV')

def summarize_results(input_csv, output_csv=None):
    """Đọc CSV và tính trung bình, std của các metrics"""
    
    if not os.path.exists(input_csv):
        print(f"Error: File not found: {input_csv}")
        return None
    
    # Read all rows
    with open(input_csv, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if len(rows) == 0:
        print("Error: No data in CSV file")
        return None
    
    # Metrics to summarize
    metric_keys = [
        'RV_dice', 'RV_hd95', 'RV_asd',
        'Myo_dice', 'Myo_hd95', 'Myo_asd',
        'LV_dice', 'LV_hd95', 'LV_asd',
        'mean_dice', 'mean_hd95', 'mean_asd'
    ]
    
    # Collect values
    metrics_data = {key: [] for key in metric_keys}
    for row in rows:
        for key in metric_keys:
            if key in row:
                metrics_data[key].append(float(row[key]))
    
    # Calculate statistics
    summary = {
        'experiment': rows[0].get('exp', 'unknown'),
        'model': rows[0].get('model', 'unknown'),
        'num_runs': len(rows),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    for key in metric_keys:
        values = np.array(metrics_data[key])
        summary[f'{key}_mean'] = np.mean(values)
        summary[f'{key}_std'] = np.std(values)
        summary[f'{key}_min'] = np.min(values)
        summary[f'{key}_max'] = np.max(values)
    
    # Print summary
    print("\n" + "="*80)
    print(f"EXPERIMENT SUMMARY: {summary['experiment']}")
    print(f"Model: {summary['model']} | Number of runs: {summary['num_runs']}")
    print("="*80)
    
    print("\n{:<10} {:>12} {:>12} {:>12} {:>12}".format(
        "Metric", "Mean", "Std", "Min", "Max"))
    print("-"*60)
    
    for key in ['RV_dice', 'Myo_dice', 'LV_dice', 'mean_dice']:
        print("{:<10} {:>12.4f} {:>12.4f} {:>12.4f} {:>12.4f}".format(
            key, summary[f'{key}_mean'], summary[f'{key}_std'],
            summary[f'{key}_min'], summary[f'{key}_max']))
    
    print("-"*60)
    
    for key in ['RV_hd95', 'Myo_hd95', 'LV_hd95', 'mean_hd95']:
        print("{:<10} {:>12.2f} {:>12.2f} {:>12.2f} {:>12.2f}".format(
            key, summary[f'{key}_mean'], summary[f'{key}_std'],
            summary[f'{key}_min'], summary[f'{key}_max']))
    
    print("-"*60)
    
    for key in ['RV_asd', 'Myo_asd', 'LV_asd', 'mean_asd']:
        print("{:<10} {:>12.2f} {:>12.2f} {:>12.2f} {:>12.2f}".format(
            key, summary[f'{key}_mean'], summary[f'{key}_std'],
            summary[f'{key}_min'], summary[f'{key}_max']))
    
    print("="*80)
    
    # Final summary
    print("\n📊 FINAL RESULTS (Mean ± Std):")
    print(f"   Dice Score: {summary['mean_dice_mean']:.4f} ± {summary['mean_dice_std']:.4f}")
    print(f"   HD95:       {summary['mean_hd95_mean']:.2f} ± {summary['mean_hd95_std']:.2f}")
    print(f"   ASD:        {summary['mean_asd_mean']:.2f} ± {summary['mean_asd_std']:.2f}")
    print("="*80 + "\n")
    
    # Save to CSV if specified
    if output_csv:
        csv_dir = os.path.dirname(output_csv)
        if csv_dir and not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
        
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary.keys())
            writer.writeheader()
            writer.writerow(summary)
        print(f"Summary saved to: {output_csv}")
    
    return summary


if __name__ == '__main__':
    args = parser.parse_args()
    summarize_results(args.input_csv, args.output_csv)
