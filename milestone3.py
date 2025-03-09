import os
import time
from inverted_index import InvertedIndex
import search
from merge import merge_partial_indexes

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

def search_helper(index, query: str):
    """
    Process a single query and display results
    """
    start_time = time.time()
    matching_docs, scores = search.boolean_search(index, query)
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\nQuery: '{query}'")
    if matching_docs:
        # Rank documents by score
        ranked_docs = sorted(matching_docs, key=lambda doc: scores[doc], reverse=True)
        print(f"Found {len(ranked_docs)} matching document(s) in {elapsed_time:.3f} seconds:")
        
        # Display top 5 results
        for i, doc in enumerate(ranked_docs[:5], 1):
            print(f"{i}. Document: {doc}")
            print(f"   Score: {scores[doc]:.3f}")
    else:
        print("No matching documents found.")

def build_index():
    """
    Build the index using partial indexing and enhanced features
    """
    start_time = time.time()
    
    # Initialize indexer with enhanced features
    indexer = InvertedIndex()
    
    # Set corpus directory
    corpus_directory = "C:\\Users\\Owner\\Downloads\\developer\\DEV"
    
    # Build partial indexes (with enhanced features)
    print("Building partial indexes...")
    indexer.build_index_from_corpus(corpus_directory, partial_chunk_size=100)
    
    # Get list of created partial index files
    partial_files = [
        f for f in os.listdir('.')
        if f.startswith("partial_index_") and f.endswith(".txt")
    ]
    
    # Sort them numerically
    partial_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
    
    print(f"\nFound {len(partial_files)} partial indexes to merge:")
    for f in partial_files:
        print(f"  - {f}")
    
    # Merge partial indexes
    print("\nMerging partial indexes...")
    final_index_file = "final_index.txt"
    merge_partial_indexes(partial_files, final_index_file)
    
    # Optional: Remove partial files after merging
    for partial_file in partial_files:
        os.remove(partial_file)
        print(f"Removed partial file: {partial_file}")
    
    end_time = time.time()
    print(f"\nIndex building and merging completed in {end_time - start_time:.2f} seconds")
    
    return final_index_file, indexer



def main():
    # Build the enhanced index
    print("Starting index building process...")
    final_index_file, indexer = build_index()
    
    # Load the index for searching
    start = time.time()
    index = search.load_index(final_index_file)
    end = time.time()
    print(f"Loaded index in {end - start:.5f} seconds")
    
    # Process queries
    print("\nProcessing good queries...")
    for query in good_queries:
        search_helper(index, indexer, query)
    
    print("\nProcessing bad queries...")
    for query in bad_queries:
        search_helper(index, indexer, query)
        

if __name__ == "__main__":
    main()

    
    
    
    