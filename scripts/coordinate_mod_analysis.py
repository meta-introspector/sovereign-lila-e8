#!/usr/bin/env python3
"""Sum mod values as coordinates and re-mod by 16, 10, 8."""

import json
import hashlib
from pathlib import Path
from collections import defaultdict

PRIMES = [71, 59, 47, 43, 41, 37, 31, 29, 23, 19, 17, 13, 11, 7, 5, 3, 2]
REMODS = [16, 10, 8]

def hash_file(path):
    try:
        with open(path, 'rb') as f:
            return hashlib.blake2b(f.read(), digest_size=16).digest()
    except:
        return None

def analyze_coordinate_mods(file_paths):
    print(f"🔍 Analyzing {len(file_paths)} files...")
    
    # Calculate hashes and mod coordinates
    file_coords = {}
    for i, fpath in enumerate(file_paths):
        if i % 10000 == 0:
            print(f"   Processing: {i}/{len(file_paths)}")
        h = hash_file(fpath)
        if h:
            hash_int = int.from_bytes(h, 'big')
            # Calculate all prime mods
            mods = [hash_int % p for p in PRIMES]
            # Sum them as coordinate
            coord_sum = sum(mods)
            file_coords[fpath] = (coord_sum, mods)
    
    print(f"✅ Processed {len(file_coords)} files")
    
    # Re-mod by 16, 10, 8
    results = {}
    for remod in REMODS:
        buckets = defaultdict(list)
        for fpath, (coord_sum, mods) in file_coords.items():
            bucket = coord_sum % remod
            buckets[bucket].append(fpath)
        
        # Stats
        sizes = sorted([len(v) for v in buckets.values()], reverse=True)
        print(f"\nmod {remod}:")
        print(f"  Buckets: {len(buckets)}")
        print(f"  Largest: {sizes[0]} files")
        print(f"  Smallest: {sizes[-1]} files")
        print(f"  Distribution: {sizes[:10]}")
        
        results[remod] = {
            'num_buckets': len(buckets),
            'bucket_sizes': {k: len(v) for k, v in buckets.items()},
            'examples': {k: v[:5] for k, v in list(buckets.items())[:3]}
        }
    
    return results

if __name__ == "__main__":
    with open("spool_filelist.json") as f:
        file_paths = json.load(f)
    
    results = analyze_coordinate_mods(file_paths)
    
    with open("coordinate_mod_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to coordinate_mod_analysis.json")
