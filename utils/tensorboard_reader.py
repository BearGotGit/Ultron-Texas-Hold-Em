"""
TensorBoard Log Reader Utility.

Reads and displays training metrics from TensorBoard event files.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("TensorBoard not installed. Run: pip install tensorboard")
    sys.exit(1)


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    steps: List[int]
    values: List[float]
    tag: str
    
    @property
    def min(self) -> float:
        return min(self.values) if self.values else 0.0
    
    @property
    def max(self) -> float:
        return max(self.values) if self.values else 0.0
    
    @property
    def mean(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0.0
    
    @property
    def last(self) -> float:
        return self.values[-1] if self.values else 0.0
    
    @property
    def first(self) -> float:
        return self.values[0] if self.values else 0.0


def load_tensorboard_logs(log_dir: str) -> Dict[str, TrainingMetrics]:
    """
    Load all scalar metrics from a TensorBoard log directory.
    
    Args:
        log_dir: Path to the TensorBoard log directory (e.g., runs/ppo_123456/)
        
    Returns:
        Dictionary mapping metric names to TrainingMetrics objects
    """
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    metrics = {}
    
    # Get all scalar tags
    scalar_tags = event_acc.Tags().get('scalars', [])
    
    for tag in scalar_tags:
        events = event_acc.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        metrics[tag] = TrainingMetrics(steps=steps, values=values, tag=tag)
    
    return metrics


def find_latest_run(runs_dir: str = "runs") -> Optional[str]:
    """Find the most recent TensorBoard run directory."""
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        return None
    
    run_dirs = [d for d in runs_path.iterdir() if d.is_dir()]
    if not run_dirs:
        return None
    
    # Sort by modification time (most recent first)
    run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(run_dirs[0])


def list_all_runs(runs_dir: str = "runs") -> List[str]:
    """List all TensorBoard run directories."""
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        return []
    
    run_dirs = [d for d in runs_path.iterdir() if d.is_dir()]
    run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return [str(d) for d in run_dirs]


def print_training_summary(log_dir: str):
    """
    Print a comprehensive summary of training metrics.
    
    Args:
        log_dir: Path to the TensorBoard log directory
    """
    print(f"\n{'='*70}")
    print(f"TRAINING SUMMARY: {log_dir}")
    print(f"{'='*70}\n")
    
    metrics = load_tensorboard_logs(log_dir)
    
    if not metrics:
        print("No metrics found in log directory.")
        return
    
    # Group metrics by category
    categories = {}
    for tag, metric in metrics.items():
        parts = tag.split('/')
        category = parts[0] if len(parts) > 1 else "other"
        if category not in categories:
            categories[category] = {}
        categories[category][tag] = metric
    
    # Print summary stats
    for category, cat_metrics in sorted(categories.items()):
        print(f"\n📊 {category.upper()}")
        print("-" * 50)
        
        for tag, metric in sorted(cat_metrics.items()):
            name = tag.split('/')[-1]
            print(f"  {name}:")
            print(f"    Steps recorded: {len(metric.steps)}")
            if metric.steps:
                print(f"    Step range: {metric.steps[0]:,} → {metric.steps[-1]:,}")
            print(f"    First: {metric.first:.4f}")
            print(f"    Last:  {metric.last:.4f}")
            print(f"    Min:   {metric.min:.4f}")
            print(f"    Max:   {metric.max:.4f}")
            print(f"    Mean:  {metric.mean:.4f}")
            
            # Show improvement
            if len(metric.values) > 1:
                change = metric.last - metric.first
                pct = (change / abs(metric.first) * 100) if metric.first != 0 else 0
                direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                print(f"    Change: {change:+.4f} ({pct:+.1f}%) {direction}")
            print()
    
    # Overall training stats
    if 'charts/fps' in metrics:
        fps = metrics['charts/fps']
        print(f"\n⚡ PERFORMANCE")
        print("-" * 50)
        print(f"  Average FPS: {fps.mean:.1f}")
        print(f"  Max FPS: {fps.max:.1f}")
    
    # Total iterations
    any_metric = next(iter(metrics.values()))
    if any_metric.steps:
        print(f"\n📈 TRAINING PROGRESS")
        print("-" * 50)
        print(f"  Total steps: {any_metric.steps[-1]:,}")
        print(f"  Data points logged: {len(any_metric.steps)}")


def plot_metrics_ascii(metrics: Dict[str, TrainingMetrics], metric_name: str, width: int = 60, height: int = 15):
    """
    Create a simple ASCII plot of a metric.
    
    Args:
        metrics: Dictionary of metrics from load_tensorboard_logs
        metric_name: Name of metric to plot
        width: Width of the plot in characters
        height: Height of the plot in characters
    """
    if metric_name not in metrics:
        print(f"Metric '{metric_name}' not found.")
        return
    
    metric = metrics[metric_name]
    values = metric.values
    
    if not values:
        print("No data to plot.")
        return
    
    # Normalize values to fit height
    min_val = min(values)
    max_val = max(values)
    val_range = max_val - min_val if max_val != min_val else 1
    
    # Sample values to fit width
    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = values
        sampled.extend([values[-1]] * (width - len(values)))
    
    # Create plot
    print(f"\n{metric_name}")
    print(f"Max: {max_val:.4f}")
    
    for row in range(height):
        threshold = max_val - (row / (height - 1)) * val_range
        line = ""
        for val in sampled:
            if val >= threshold:
                line += "█"
            else:
                line += " "
        
        if row == 0:
            print(f"│{line}│")
        elif row == height - 1:
            print(f"│{line}│")
        else:
            print(f"│{line}│")
    
    print(f"Min: {min_val:.4f}")
    print(f"Steps: 0 → {metric.steps[-1]:,}")


def export_to_csv(metrics: Dict[str, TrainingMetrics], output_file: str):
    """Export metrics to CSV file."""
    import csv
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = ['step'] + list(metrics.keys())
        writer.writerow(header)
        
        # Get all unique steps
        all_steps = set()
        for metric in metrics.values():
            all_steps.update(metric.steps)
        all_steps = sorted(all_steps)
        
        # Create step-to-value mapping for each metric
        step_values = {}
        for tag, metric in metrics.items():
            step_values[tag] = dict(zip(metric.steps, metric.values))
        
        # Write rows
        for step in all_steps:
            row = [step]
            for tag in metrics.keys():
                row.append(step_values[tag].get(step, ''))
            writer.writerow(row)
    
    print(f"Exported to {output_file}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Read TensorBoard logs")
    parser.add_argument("--log-dir", type=str, default=None, help="Path to log directory")
    parser.add_argument("--list-runs", action="store_true", help="List all available runs")
    parser.add_argument("--latest", action="store_true", help="Use the most recent run")
    parser.add_argument("--plot", type=str, default=None, help="Metric to plot (ASCII)")
    parser.add_argument("--export-csv", type=str, default=None, help="Export to CSV file")
    parser.add_argument("--runs-dir", type=str, default="runs", help="Base directory for runs")
    args = parser.parse_args()
    
    if args.list_runs:
        runs = list_all_runs(args.runs_dir)
        print(f"\nAvailable runs in '{args.runs_dir}':")
        for i, run in enumerate(runs, 1):
            print(f"  {i}. {run}")
        return
    
    # Determine log directory
    log_dir = args.log_dir
    if log_dir is None or args.latest:
        log_dir = find_latest_run(args.runs_dir)
        if log_dir is None:
            print(f"No runs found in '{args.runs_dir}'")
            return
    
    # Load and display metrics
    metrics = load_tensorboard_logs(log_dir)
    print_training_summary(log_dir)
    
    # Optional ASCII plot
    if args.plot:
        plot_metrics_ascii(metrics, args.plot)
    
    # Optional CSV export
    if args.export_csv:
        export_to_csv(metrics, args.export_csv)


if __name__ == "__main__":
    main()
