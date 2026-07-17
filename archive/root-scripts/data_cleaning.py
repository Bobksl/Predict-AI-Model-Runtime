"""
Data Cleaning Module for Predict AI Model Runtime Competition

This module provides comprehensive data cleaning utilities including:
- NaN/Inf value handling
- Outlier detection and treatment
- Feature normalization
- Runtime normalizer validation
- Schema consistency checks
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import Enum
import json

import numpy as np
import pandas as pd


class CleanStrategy(Enum):
    """Strategy for handling data quality issues."""
    REMOVE = "remove"
    REPLACE = "replace"
    CLIP = "clip"
    INTERPOLATE = "interpolate"
    MARK = "mark"


@dataclass
class CleaningConfig:
    """Configuration for data cleaning operations."""
    nan_strategy: CleanStrategy = CleanStrategy.REPLACE
    nan_fill_value: float = 0.0
    inf_strategy: CleanStrategy = CleanStrategy.REPLACE
    inf_fill_value: float = 0.0
    zero_normalizer_fill_value: float = 1.0
    outlier_std_threshold: float = 5.0
    clip_runtime_ratio: bool = True
    runtime_ratio_percentiles: Tuple[float, float] = (0.1, 99.9)


@dataclass
class CleaningReport:
    """Report of cleaning operations performed."""
    total_files: int = 0
    files_modified: int = 0
    nan_count: int = 0
    inf_count: int = 0
    zero_normalizer_count: int = 0
    outliers_clipped: int = 0
    schema_inconsistencies: List[Dict] = field(default_factory=list)
    file_reports: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "total_files": self.total_files,
            "files_modified": self.files_modified,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "zero_normalizer_count": self.zero_normalizer_count,
            "outliers_clipped": self.outliers_clipped,
            "schema_inconsistencies": self.schema_inconsistencies,
        }
    
    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def load_npz_safe(path: str) -> Tuple[Optional[dict], Optional[str]]:
    """Safely load an NPZ file with error handling."""
    try:
        data = np.load(path, allow_pickle=True)
        return data, None
    except Exception as e:
        return None, str(e)


def handle_nan_in_array(arr: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """Handle NaN values by replacing with fill_value."""
    return np.where(np.isnan(arr), fill_value, arr)


def handle_inf_in_array(arr: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """Handle Inf values by replacing with fill_value."""
    return np.where(np.isinf(arr), fill_value, arr)


def clip_outliers(arr: np.ndarray, 
                 std_threshold: float = 5.0,
                 percentiles: Tuple[float, float] = (0.1, 99.9)) -> Tuple[np.ndarray, int]:
    """Clip outliers in an array."""
    mean = np.mean(arr)
    std = np.std(arr)
    lower_bound = mean - std_threshold * std
    upper_bound = mean + std_threshold * std
    p_lower = np.percentile(arr, percentiles[0])
    p_upper = np.percentile(arr, percentiles[1])
    lower = max(lower_bound, p_lower)
    upper = min(upper_bound, p_upper)
    clipped = np.clip(arr, lower, upper)
    n_clipped = np.sum((arr < lower) | (arr > upper))
    return clipped, int(n_clipped)


def compute_safe_runtime_ratio(runtime: np.ndarray,
                               normalizer: np.ndarray,
                               fill_value: float = 1.0) -> Tuple[np.ndarray, int]:
    """Compute runtime / normalizer ratio safely handling zero/negative normalizers."""
    invalid_mask = normalizer <= 0
    n_invalid = np.sum(invalid_mask)
    safe_normalizer = np.where(invalid_mask, fill_value, normalizer)
    ratio = runtime / safe_normalizer
    return ratio, int(n_invalid)


class DataCleaner:
    """Main class for cleaning NPZ data files."""
    
    def __init__(self, config: Optional[CleaningConfig] = None):
        self.config = config or CleaningConfig()
        self.report = CleaningReport()
        self.schema_counts: Dict[str, int] = {}
        self.schema_examples: Dict[str, str] = {}
    
    def clean_file(self, path: str, output_dir: Optional[str] = None) -> bool:
        """Clean a single NPZ file and save the cleaned version."""
        self.report.total_files += 1
        data, error = load_npz_safe(path)
        
        if error is not None:
            print(f"Error loading {path}: {error}")
            return False
        
        file_report = {"path": path, "cleaned": False, "issues": []}
        
        try:
            modified = False
            
            # Check schema
            keys_str = ",".join(sorted(data.keys()))
            self.schema_counts[keys_str] = self.schema_counts.get(keys_str, 0) + 1
            if keys_str not in self.schema_examples:
                self.schema_examples[keys_str] = path
            
            # Prepare cleaned data
            cleaned_data = {}
            
            for key in data.keys():
                arr = data[key].copy()
                
                # Handle NaN/Inf in float arrays
                if arr.dtype in [np.float32, np.float64]:
                    if np.isnan(arr).any():
                        self.report.nan_count += int(np.sum(np.isnan(arr)))
                        file_report["issues"].append(f"NaN in {key}: {np.sum(np.isnan(arr))}")
                        arr = handle_nan_in_array(arr, self.config.nan_fill_value)
                        modified = True
                    
                    if np.isinf(arr).any():
                        self.report.inf_count += int(np.sum(np.isinf(arr)))
                        file_report["issues"].append(f"Inf in {key}: {np.sum(np.isinf(arr))}")
                        arr = handle_inf_in_array(arr, self.config.inf_fill_value)
                        modified = True
                
                # Handle zero/negative normalizers
                if key == "config_runtime_normalizers":
                    invalid_mask = arr <= 0
                    if invalid_mask.any():
                        self.report.zero_normalizer_count += int(np.sum(invalid_mask))
                        arr = np.where(invalid_mask, self.config.zero_normalizer_fill_value, arr)
                        modified = True
                
                # Handle config runtime outliers
                if key == "config_runtime" and self.config.clip_runtime_ratio:
                    clipped, n_clipped = clip_outliers(
                        arr, self.config.outlier_std_threshold, self.config.runtime_ratio_percentiles
                    )
                    if n_clipped > 0:
                        self.report.outliers_clipped += n_clipped
                        arr = clipped
                        file_report["issues"].append(f"Clipped {n_clipped} outliers")
                        modified = True
                
                cleaned_data[key] = arr
            
            if modified:
                self.report.files_modified += 1
                file_report["cleaned"] = True
                
                # Save cleaned version
                if output_dir:
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    output_path = os.path.join(output_dir, os.path.basename(path))
                else:
                    output_path = path.replace(".npz", "_cleaned.npz")
                
                np.savez_compressed(output_path, **cleaned_data)
            
            self.report.file_reports.append(file_report)
            return modified
            
        finally:
            data.close()
    
    def clean_directory(self, data_root: str, output_dir: Optional[str] = None) -> CleaningReport:
        """Clean all NPZ files in a directory."""
        from data_exploration import scan_directory
        
        files = scan_directory(data_root)
        print(f"Found {len(files)} files to process")
        
        for i, path in enumerate(files):
            if i % 100 == 0:
                print(f"Processing file {i+1}/{len(files)}...")
            self.clean_file(path, output_dir)
        
        if len(self.schema_counts) > 1:
            print(f"\nWARNING: Found {len(self.schema_counts)} different schemas!")
            for schema, count in sorted(self.schema_counts.items(), key=lambda x: -x[1]):
                example = self.schema_examples[schema]
                self.report.schema_inconsistencies.append({
                    "schema": schema, "count": count, "example_file": example,
                })
        
        return self.report


def quick_clean_file(input_path: str, output_path: Optional[str] = None) -> bool:
    """Quick function to clean a single file."""
    data, error = load_npz_safe(input_path)
    if error:
        print(f"Error: {error}")
        return False
    
    modified = False
    cleaned_data = {}
    
    for key in data.keys():
        arr = data[key].copy()
        if arr.dtype in [np.float32, np.float64]:
            if np.isnan(arr).any():
                arr = handle_nan_in_array(arr)
                modified = True
            if np.isinf(arr).any():
                arr = handle_inf_in_array(arr)
                modified = True
        cleaned_data[key] = arr
    
    data.close()
    
    if modified:
        output = output_path or input_path
        np.savez_compressed(output, **cleaned_data)
        print(f"Cleaned file saved to {output}")
    
    return modified


def verify_cleaned_file(path: str) -> Tuple[bool, List[str]]:
    """Verify that a cleaned file has no data quality issues."""
    data, error = load_npz_safe(path)
    if error:
        return False, [f"Failed to load: {error}"]
    
    issues = []
    for key in data.keys():
        arr = data[key]
        if np.isnan(arr).any():
            issues.append(f"NaN in {key}")
        if np.isinf(arr).any():
            issues.append(f"Inf in {key}")
    
    data.close()
    return len(issues) == 0, issues


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean TPUGraphs NPZ files")
    parser.add_argument("--input", type=str, required=True, help="Input file or directory")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--report", type=str, default="cleaning_report.json", help="Output path for report")
    
    args = parser.parse_args()
    
    config = CleaningConfig()
    cleaner = DataCleaner(config)
    
    if os.path.isdir(args.input):
        report = cleaner.clean_directory(args.input, args.output)
        report.save(args.report)
        print(f"\nCleaning complete! Total: {report.total_files}, Modified: {report.files_modified}")
        print(f"NaN handled: {report.nan_count}, Inf handled: {report.inf_count}")
        print(f"Report saved to {args.report}")
    else:
        modified = quick_clean_file(args.input, args.output)
        print("File cleaned successfully!" if modified else "No cleaning needed.")