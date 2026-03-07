use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Serialize, Deserialize)]
struct TextDoc {
    text: String,
}

fn main() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(24)
        .build_global()
        .unwrap();

    println!("📂 Loading filelist...");
    let file_paths: Vec<String> = 
        serde_json::from_str(&fs::read_to_string("../spool_filelist.json").unwrap()).unwrap();
    
    println!("📖 Processing {} files with 24 CPUs...", file_paths.len());
    
    let counter = AtomicUsize::new(0);
    let texts: Vec<TextDoc> = file_paths
        .par_iter()
        .filter_map(|fpath| {
            let i = counter.fetch_add(1, Ordering::Relaxed);
            if i % 10000 == 0 {
                println!("   Progress: {}/{}", i, file_paths.len());
            }
            
            fs::read_to_string(fpath).ok().and_then(|text| {
                let truncated: String = text.chars().take(2000).collect();
                if truncated.len() > 50 {
                    Some(TextDoc { text: truncated })
                } else {
                    None
                }
            })
        })
        .collect();
    
    println!("✅ Processed {} documents", texts.len());
    println!("💾 Saving to ../spool_dataset.json...");
    
    let json = serde_json::to_string(&texts).unwrap();
    fs::write("../spool_dataset.json", json).unwrap();
    
    println!("✅ Dataset saved ({} documents)", texts.len());
}
