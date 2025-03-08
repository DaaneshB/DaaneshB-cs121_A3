from typing import List, Dict
import os

def merge_partial_indexes(partial_files: List[str], final_output: str):
    """
    Merges multiple partial index files into one final index file.
    
    Args:
        partial_files: List of partial index filenames
        final_output: Name of final merged index file
    
    Format of partial files (input):
        token|doc1:freq1,doc2:freq2,...
    """
    merged_index: Dict[str, Dict[str, int]] = {}
    
    # Process each partial file
    for partial_file in partial_files:
        print(f"Merging {partial_file}...")
        with open(partial_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                # Split line into token and postings
                try:
                    token, postings_str = line.split('|')
                except ValueError:
                    continue  # Skip malformed lines
                    
                # Initialize dict for new tokens
                if token not in merged_index:
                    merged_index[token] = {}
                    
                # Process each posting
                for posting in postings_str.split(','):
                    if not posting:
                        continue
                    try:
                        doc_id, freq_str = posting.split(':')
                        freq = int(freq_str)
                        # Add frequency to existing value if doc exists
                        merged_index[token][doc_id] = merged_index[token].get(doc_id, 0) + freq
                    except (ValueError, TypeError):
                        continue  # Skip malformed postings
    
    # Write merged index sorted by token
    print(f"Writing merged index to {final_output}...")
    with open(final_output, 'w', encoding='utf-8') as outf:
        for token in sorted(merged_index.keys()):
            post_dict = merged_index[token]
            postings_str = ','.join(f"{doc_id}:{freq}" 
                                  for doc_id, freq in post_dict.items())
            outf.write(f"{token}|{postings_str}\n")
    
    print(f"Final merged index saved to {final_output}")