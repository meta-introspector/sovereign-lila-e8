#!/usr/bin/env python3
"""Preprocess spool dataset into shared memory for fast training."""

import json
import pickle
import mmap
from pathlib import Path
import multiprocessing as mp

def preprocess_to_shm():
    """Load dataset into shared memory."""
    print("📂 Loading filelist...")
    with open("spool_filelist.json") as f:
        file_paths = json.load(f)
    
    print(f"📖 Reading {len(file_paths)} files...")
    texts = []
    for i, fpath in enumerate(file_paths):
        if i % 10000 == 0:
            print(f"   Progress: {i}/{len(file_paths)}")
        try:
            text = Path(fpath).read_text()[:2000]
            if len(text) > 50:
                texts.append({'text': text})
        except:
            pass
    
    print(f"✅ Processed {len(texts)} documents")
    
    # Save as pickle for fast loading
    print(f"💾 Saving to spool_dataset.pkl...")
    with open("spool_dataset.pkl", "wb") as f:
        pickle.dump(texts, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"✅ Dataset saved - will load instantly on next run")
    return texts

if __name__ == "__main__":
    preprocess_to_shm()
