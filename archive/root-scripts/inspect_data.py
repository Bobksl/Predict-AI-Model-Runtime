#!/usr/bin/env python
"""Script to inspect the actual NPZ data structure."""
import numpy as np
import os

def inspect_file(path):
    """Inspect a single NPZ file."""
    print(f"\n{'='*60}")
    print(f"File: {path}")
    print('='*60)
    
    d = np.load(path)
    print(f"Keys: {list(d.keys())}")
    print()
    
    for key in d.keys():
        arr = d[key]
        print(f"{key}:")
        print(f"  Shape: {arr.shape}")
        print(f"  Dtype: {arr.dtype}")
        
        # Show sample values
        if arr.dtype in [np.float32, np.float64]:
            flat = arr.flatten()
            valid = flat[np.isfinite(flat)]
            if len(valid) > 0:
                print(f"  Min: {valid.min():.4f}, Max: {valid.max():.4f}, Mean: {valid.mean():.4f}")
                print(f"  Has NaN: {np.isnan(arr).any()}, Has Inf: {np.isinf(arr).any()}")
        elif arr.dtype in [np.int32, np.int64]:
            print(f"  Min: {arr.min()}, Max: {arr.max()}, Unique values: {len(np.unique(arr))}")
        elif arr.dtype == object:
            print(f"  Object array (might be strings)")
        
        print()
    
    d.close()

def main():
    base = "data/tpugraphs/npz"
    
    # Check tile files
    tile_train = os.path.join(base, "tile/xla/train")
    if os.path.exists(tile_train):
        files = sorted(os.listdir(tile_tile_train := tile_train))[:2]
        for f in files:
            inspect_file(os.path.join(tile_train, f))
    
    # Check layout files
    layout_train = os.path.join(base, "layout/xla/random/train")
    if os.path.exists(layout_train):
        files = sorted(os.listdir(layout_train))[:2]
        for f in files:
            inspect_file(os.path.join(layout_train, f))
    
    # Check layout/nlp
    layout_nlp = os.path.join(base, "layout/nlp/random/train")
    if os.path.exists(layout_nlp):
        files = sorted(os.listdir(layout_nlp))[:2]
        for f in files:
            inspect_file(os.path.join(layout_nlp, f))

if __name__ == "__main__":
    main()
