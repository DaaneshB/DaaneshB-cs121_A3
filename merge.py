from typing import List, Dict, Set, DefaultDict
from collections import defaultdict
import os
import json
from link_analyzer import LinkAnalyzer

def merge_partial_indexes(partial_files: List[str], final_output: str):
    """
    Merge partial indexes focusing on essential data only
    """
    print("\nStarting merge process...")
    
    # Initialize merged data structures
    merged_index = {}
    merged_doc_freqs = defaultdict(int)
    link_graph = defaultdict(list)
    anchor_texts = defaultdict(list)
    total_docs = 0
    
    # Process each partial file
    for partial_file in partial_files:
        print(f"\nProcessing {partial_file}...")
        
        try:
            # Process main index
            with open(partial_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or '|' not in line:
                        continue
                    
                    # Process token and its postings
                    token, postings_str = line.split('|')
                    if not token or not postings_str:
                        continue
                        
                    if token not in merged_index:
                        merged_index[token] = {}

                    # Process each posting
                    for posting in postings_str.split(','):
                        if not posting:
                            continue
                            
                        try:
                            doc_id, freq_str = posting.rsplit(':', 1)
                            freq = float(freq_str)  # Handle weighted frequencies
                            
                            # Update merged index
                            if doc_id in merged_index[token]:
                                merged_index[token][doc_id] += freq
                            else:
                                merged_index[token][doc_id] = freq
                                merged_doc_freqs[token] += 1
                                
                        except ValueError:
                            print(f"Warning: Invalid posting format in {partial_file}, line {line_num}")
                            continue

            # Process scores file
            scores_file = partial_file.replace('index', 'scores')
            if os.path.exists(scores_file):
                with open(scores_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                            
                        parts = line.split('|')
                        if len(parts) < 2:
                            continue
                            
                        record_type = parts[0]
                        
                        if record_type == 'df':
                            # Document frequency
                            token, freq = parts[1:]
                            merged_doc_freqs[token] += int(freq)
                            
                        elif record_type == 'link':
                            # Link structure
                            source, targets = parts[1:]
                            link_graph[source].extend(targets.split(','))
                            
                        elif record_type == 'anchor':
                            # Anchor texts
                            url, texts = parts[1:]
                            anchor_texts[url].extend(texts.split(','))

        except Exception as e:
            print(f"Error processing {partial_file}: {e}")
            continue

    # Compute final scores
    print("\nComputing ranking scores...")
    
    # Initialize link analyzer
    link_analyzer = LinkAnalyzer()
    link_analyzer.graph = {url: list(set(targets)) for url, targets in link_graph.items()}
    
    # Compute PageRank
    print("Computing PageRank scores...")
    pagerank_scores = link_analyzer.compute_pagerank()
    
    # Compute HITS
    print("Computing HITS scores...")
    hub_scores, auth_scores = link_analyzer.compute_hits()

    # Write final merged index
    print("\nWriting final index...")
    with open(final_output, 'w', encoding='utf-8') as f:
        for token in sorted(merged_index.keys()):
            postings = merged_index[token]
            if postings:
                postings_str = ','.join(f"{doc_id}:{freq}"
                                      for doc_id, freq in postings.items())
                f.write(f"{token}|{postings_str}\n")

    # Write final scores
    print("Writing final scores...")
    with open('final_scores.txt', 'w', encoding='utf-8') as f:
        # Write document frequencies
        for token, freq in merged_doc_freqs.items():
            f.write(f"df|{token}|{freq}\n")
        
        # Write link structure and scores
        for url, score in pagerank_scores.items():
            f.write(f"pr|{url}|{score}\n")
        for url, score in hub_scores.items():
            f.write(f"hub|{url}|{score}\n")
        for url, score in auth_scores.items():
            f.write(f"auth|{url}|{score}\n")
        
        # Write anchor texts (unique only)
        for url, texts in anchor_texts.items():
            unique_texts = list(set(texts))
            f.write(f"anchor|{url}|{','.join(unique_texts)}\n")

    print("\nMerge complete!")
    print(f"Processed {len(partial_files)} partial indexes")
    print(f"Final index contains {len(merged_index)} unique terms")
    print(f"Created {final_output} and final_scores.txt")

if __name__ == "__main__":
    # Get list of partial index files
    partial_files = [
        f for f in os.listdir('.')
        if f.startswith("partial_index_") and f.endswith(".txt")
    ]
    
    if not partial_files:
        print("No partial index files found!")
    else:
        # Sort them numerically
        partial_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
        merge_partial_indexes(partial_files, "final_index.txt")