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



 # End timer
def search_helper(index, query):
    start_time = time.time()  # Start timer
    matching_docs, scores = search.boolean_search(index, query)
    end_time = time.time()
    elapsed_time = end_time - start_time  # Calculate elapsed time 
    if matching_docs:
        # Rank the documents by their accumulated score (higher is better)
        ranked_docs = sorted(matching_docs, key=lambda doc: scores[doc], reverse=True)
        
        print(f"\nFound {len(ranked_docs)} matching document(s) in {elapsed_time: .3f} seconds: Top results:")
        for doc in ranked_docs[:5]:  # show top 5 results
            print(f"Document: {doc}, Score: {scores[doc]}")
    else:
        print(f"\nQuery: '{query}'")
        print("No documents found for the given query.")

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
    
    # Build partial indexes (adjust chunk_size as needed)
    print("Building partial indexes...")
    indexer.build_index_from_corpus(corpus_directory, partial_chunk_size=100)
    
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
    
    return final_index_file

def main():
    start = time.time()
    index_file = build_index()
    index = search.load_index(index_file)
    print("Inverted index loaded successfully.")
    end = time.time()
    print(f"{end - start: 5f} seconds to load index")
    for query in good_queries:
        search_helper(index,query)
    for query in bad_queries:
        search_helper(index,query)
        

if __name__ == "__main__":
    main()

    
    
    
    