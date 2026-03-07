#!/usr/bin/env python3
"""Hash analysis of spool files using modulo primes to find duplicates."""

import json
import hashlib
from pathlib import Path
from collections import defaultdict

PRIMES = [71, 59, 47, 43, 41, 37, 31, 29, 23, 19, 17, 13, 11, 7, 5, 3, 2]

def hash_file(path):
    """Fast hash of file content."""
    try:
        with open(path, 'rb') as f:
            return hashlib.blake2b(f.read(), digest_size=16).digest()
    except:
        return None

def analyze_hashes(file_paths):
    """Calculate hash mod primes and find duplicates."""
    print(f"🔍 Analyzing {len(file_paths)} files...")
    
    # Calculate hashes
    hashes = {}
    for i, fpath in enumerate(file_paths):
        if i % 10000 == 0:
            print(f"   Hashing: {i}/{len(file_paths)}")
        h = hash_file(fpath)
        if h:
            hashes[fpath] = int.from_bytes(h, 'big')
    
    print(f"✅ Hashed {len(hashes)} files")
    
    # Analyze modulo each prime
    results = {}
    for prime in PRIMES:
        buckets = defaultdict(list)
        for fpath, h in hashes.items():
            buckets[h % prime].append(fpath)
        
        # Find collisions
        collisions = {k: v for k, v in buckets.items() if len(v) > 1}
        total_dupes = sum(len(v) - 1 for v in collisions.values())
        
        results[prime] = {
            'buckets': len(buckets),
            'collisions': len(collisions),
            'duplicate_files': total_dupes,
            'examples': {k: v[:3] for k, v in list(collisions.items())[:5]}
        }
        
        print(f"mod {prime:2d}: {len(buckets):3d} buckets, {len(collisions):3d} collisions, {total_dupes:5d} dupes")
    
    return results, hashes

if __name__ == "__main__":
    # Load filelist
    with open("spool_filelist.json") as f:
        file_paths = json.load(f)
    
    results, hashes = analyze_hashes(file_paths)
    
    # Save results
    with open("hash_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to hash_analysis.json")
