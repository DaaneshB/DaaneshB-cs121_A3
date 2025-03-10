import os
import time
from inverted_index import InvertedIndex
import search
from merge import merge_partial_indexes
from nltk.stem import PorterStemmer
from typing import List, Dict, Tuple, Set
from search_structure import SearchStructure



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


def search_helper(search_struct: SearchStructure, query: str):
    """Enhanced search helper using all ranking features"""
    start_time = time.time()
    
    # Process query
    stemmer = PorterStemmer()
    query_terms = [stemmer.stem(term.lower()) for term in query.split()]
    
    # Get initial matches using auxiliary dictionary
    matching_docs = None
    for term in query_terms:
        postings = search_struct.get_postings(term)
        docs = set(doc_id for doc_id, _ in postings)
        if matching_docs is None:
            matching_docs = docs
        else:
            matching_docs &= docs
    
    if matching_docs:
        # Score documents using all features
        scored_docs = []
        for doc_id in matching_docs:
            final_score, components = search_struct.compute_final_score(
                doc_id, query, query_terms
            )
            scored_docs.append((doc_id, final_score, components))
        
        # Sort by final score
        ranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"\nQuery: '{query}'")
        print(f"Found {len(ranked_docs)} matching document(s) in {elapsed_time:.3f} seconds:")
        
        # Show top 5 results with detailed scoring
        for i, (doc_id, score, components) in enumerate(ranked_docs[:5], 1):
            print(f"\n{i}. Document: {doc_id}")
            print(f"   Final Score: {score:.4f}")
            print("   Score Components:")
            for component, component_score in components.items():
                print(f"      {component}: {component_score:.4f}")
            
            # Show matching anchor texts if any
            anchors = search_struct.anchor_texts.get(doc_id, [])
            if anchors:
                print("   Anchor texts:")
                for anchor in anchors[:2]:  # Show first 2 anchor texts
                    print(f"      - {anchor}")
    else:
        end_time = time.time()
        print(f"\nQuery: '{query}'")
        print(f"No matching documents found ({end_time - start_time:.3f} seconds)")

def main():
    """
    Main function with optional index building
    """
    # Configuration
    BUILD_INDEX = True  # Set to True to rebuild index
    corpus_directory = "C:\\Users\\DanBo\\Downloads\\developer\\DEV"
    
    if BUILD_INDEX:
        print("\nBuilding index...")
        # Clean up any existing files
        for f in os.listdir('.'):
            if f.startswith(('partial_index_', 'partial_scores_')) or f in ['final_index.txt', 'final_scores.txt']:
                os.remove(f)
                print(f"Removed: {f}")
        
        # Build new index
        indexer = InvertedIndex()
        indexer.build_index_from_corpus(corpus_directory, partial_chunk_size=10)
        
        # Get list of partial index files
        partial_files = sorted(
            [f for f in os.listdir('.') if f.startswith('partial_index_') and f.endswith('.txt')],
            key=lambda x: int(x.split('_')[2].split('.')[0])
        )
        
        # Merge partial indexes
        print("\nMerging partial indexes...")
        merge_partial_indexes(partial_files, "final_index.txt")
        
        print("\nIndex building and merging complete!")
    
    # Initialize search structure
    print("\nInitializing search structure...")
    search_struct = SearchStructure()
    
    # Load structures
    print("Loading index and scores...")
    start_time = time.time()
    search_struct.build_auxiliary_structures(
        index_file="final_index.txt",
        scores_file="final_scores.txt"
    )
    end_time = time.time()
    print(f"Structures loaded in {end_time - start_time:.2f} seconds")
    
    # Process test queries
    print("\nProcessing good queries...")
    for query in good_queries:
        search_helper(search_struct, query)
    
    print("\nProcessing bad queries...")
    for query in bad_queries:
        search_helper(search_struct, query)
    
    # Interactive mode
    print("\nEntering interactive mode...")
    print("Type 'quit' to exit")
    while True:
        query = input("\nEnter query: ").strip()
        if query.lower() == 'quit':
            break
        if query:
            search_helper(search_struct, query)

if __name__ == "__main__":
    main()