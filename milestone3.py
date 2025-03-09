import os
import time
from inverted_index import InvertedIndex
import search
from merge import merge_partial_indexes
from typing import List, Dict, Tuple, Set


good_queries = [
    "machine learning",
    "donald bren",
    "computer science",
    "artificial intelligence",
    "undergraduate degree",
    "research papers",
    "faculty directory",
    "programming languages",
    "student resources",
    "course schedule"
]

# List of bad search queries
bad_queries = [
    "database systems",
    "cristina lopes contact info",
    "ACM club meeting",
    "masters program",
    "computing labs printer access",
    "fall winter spring courses",
    "dean's office",
    "research assistant positions",
    "transfer credit requirements CS",
    "internship opportunities"
]

def search_helper(index, query: str, scores_data: Dict):
    """
    Enhanced search helper with detailed scoring output
    """
    start_time = time.time()
    matching_docs, scores = search.boolean_search(index, query, scores_data)
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    if matching_docs:
        ranked_docs = sorted(matching_docs, key=lambda doc: scores[doc], reverse=True)
        print(f"\nQuery: '{query}'")
        print(f"Found {len(ranked_docs)} matching document(s) in {elapsed_time:.3f} seconds:")
        
        # Show top 5 results with scoring breakdown
        for i, doc_id in enumerate(ranked_docs[:5], 1):
            print(f"\n{i}. Document: {doc_id}")
            print(f"   Final Score: {scores[doc_id]:.4f}")
            if isinstance(scores[doc_id], dict):  # If detailed scores available
                print("   Score Components:")
                for component, value in scores[doc_id].items():
                    print(f"      {component}: {value:.4f}")
    else:
        print(f"\nQuery: '{query}'")
        print("No matching documents found.")



def build_index():
    """
    Builds the index using partial indexing and merging.
    Returns the path to the final merged index file.
    """
    start_time = time.time()
    
    # Initialize indexer
    indexer = InvertedIndex()
    
    # Set your corpus directory path
    corpus_directory = "C:\\Users\\DanBo\\Downloads\\developer\\DEV"
    #corpus_directory = "C:\\Users\\Owner\\Downloads\\developer\\DEV"
    
    # Build partial indexes (adjust chunk_size as needed)
    print("Building partial indexes...")
    indexer.build_index_from_corpus(corpus_directory, partial_chunk_size=1000)

    # Get list of created partial index files
    partial_files = [
        f for f in os.listdir('.')
        if f.startswith("partial_index_") and f.endswith(".txt")
    ]
    
    # Merge partial indexes
    print("\nMerging partial indexes...")
    final_index_file = "final_index.txt"
    merge_partial_indexes(partial_files, final_index_file)
    
    # Optional: Remove partial index files after merging
    for partial_file in partial_files:
        os.remove(partial_file)
        print(f"Removed partial file: {partial_file}")
    
    end_time = time.time()
    print(f"\nIndex building and merging completed in {end_time - start_time:.2f} seconds")
    
    return final_index_file, indexer



def main():
    """
    Modified main function to handle ranking scores
    """
    start = time.time()
    index_file = "final_index.txt"
    
    # Load main index
    index = search.load_index(index_file)
    
    # Load ranking scores
    scores_data = search.load_scores("final")  # Load scores from final merged index
    
    end = time.time()
    print(f"Index and scores loaded in {end - start:.5f} seconds")
    
    # Process queries with enhanced scoring
    print("\nProcessing good queries...")
    for query in good_queries:
        search_helper(index, query, scores_data)
    
    print("\nProcessing bad queries...")
    for query in bad_queries:
        search_helper(index, query, scores_data)

        

if __name__ == "__main__":
    main()

    
    
    
    