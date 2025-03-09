from typing import List, Dict
import os

def merge_partial_indexes(partial_files: List[str], final_output: str):
    """
    Merges multiple partial index files into one final index file.
    Handles encoded URLs in doc_ids.
    """
    merged_index: Dict[str, Dict[str, int]] = {}
    processed_tokens = 0
    processed_postings = 0

    for partial_file in partial_files:
        print(f"Processing {partial_file}...")
        try:
            with open(partial_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        # Split into token and postings
                        if '|' not in line:
                            print(f"Warning: No delimiter '|' in line {line_num}")
                            continue
                            
                        token, postings_str = line.split('|')
                        if not token or not postings_str:
                            continue

                        # Initialize token in merged index
                        if token not in merged_index:
                            merged_index[token] = {}

                        # Process each posting
                        for posting in postings_str.split(','):
                            if not posting or ':' not in posting:
                                continue
                                
                            try:
                                # Split on the last occurrence of ':'
                                doc_id, freq_str = posting.rsplit(':', 1)
                                # Decode the URL
                                doc_id = doc_id.replace('%3A', ':')
                                
                                try:
                                    freq = int(freq_str)
                                    if freq > 0:
                                        merged_index[token][doc_id] = (
                                            merged_index[token].get(doc_id, 0) + freq
                                        )
                                        processed_postings += 1
                                except ValueError:
                                    print(f"Warning: Invalid frequency in posting: {posting}")
                                    
                            except ValueError:
                                print(f"Warning: Malformed posting: {posting}")

                        processed_tokens += 1

                    except Exception as e:
                        print(f"Error processing line {line_num}: {e}")
                        print(f"Line content: {line[:200]}...")

        except Exception as e:
            print(f"Error reading {partial_file}: {e}")
            continue

    # Write merged index
    print(f"\nWriting merged index...")
    try:
        with open(final_output, 'w', encoding='utf-8') as outf:
            for token in sorted(merged_index.keys()):
                postings = merged_index[token]
                if postings:
                    # Encode URLs again for consistent storage
                    postings_str = ','.join(
                        f"{doc_id.replace(':', '%3A')}:{freq}"
                        for doc_id, freq in postings.items()
                    )
                    outf.write(f"{token}|{postings_str}\n")

    except Exception as e:
        print(f"Error writing final index: {e}")

    print(f"Merge complete:")
    print(f"Processed tokens: {processed_tokens}")
    print(f"Processed postings: {processed_postings}")
    print(f"Final unique tokens: {len(merged_index)}")

def merge_ranking_scores(partial_files: List[str]) -> Dict:
    """
    Merge ranking scores from partial indexes
    """
    merged_scores = {
        'pagerank': {},
        'hub': {},
        'authority': {},
        'anchor_texts': defaultdict(list),
        'doc_frequencies': defaultdict(int),
        'doc_lengths': {},
        'total_docs': 0
    }
    
    # Process each partial index's scores
    for partial_file in partial_files:
        partial_num = partial_file.split('_')[2].split('.')[0]
        scores = load_scores(partial_num)
        
        # Merge PageRank scores (take max)
        for doc_id, score in scores.get('pagerank', {}).items():
            merged_scores['pagerank'][doc_id] = max(
                merged_scores['pagerank'].get(doc_id, 0),
                score
            )
        
        # Merge HITS scores (take max)
        for doc_id, score in scores.get('hub', {}).items():
            merged_scores['hub'][doc_id] = max(
                merged_scores['hub'].get(doc_id, 0),
                score
            )
        for doc_id, score in scores.get('authority', {}).items():
            merged_scores['authority'][doc_id] = max(
                merged_scores['authority'].get(doc_id, 0),
                score
            )
        
        # Merge anchor texts (combine lists)
        for doc_id, anchors in scores.get('anchor_texts', {}).items():
            merged_scores['anchor_texts'][doc_id].extend(anchors)
        
        # Update document frequencies and lengths
        for term, freq in scores.get('doc_frequencies', {}).items():
            merged_scores['doc_frequencies'][term] += freq
        
        merged_scores['doc_lengths'].update(scores.get('doc_lengths', {}))
        merged_scores['total_docs'] += scores.get('total_docs', 0)
    
    # Save merged scores
    with open('final_index_scores.json', 'w') as f:
        json.dump(merged_scores, f)
    
    return merged_scores