import time
import os
from typing import List, Dict, Tuple
from collections import defaultdict
from nltk.stem import PorterStemmer
import nltk
from bs4 import BeautifulSoup
import math

# Download required NLTK data
nltk.download('punkt', quiet=True)

def load_index(file_path: str) -> Dict:
    """
    Load index from text file format: token|doc1:freq1,doc2:freq2,...
    """
    index = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Skip empty lines
                if not line.strip():
                    continue
                
                try:
                    # Split line into token and postings
                    parts = line.strip().split('|')
                    if len(parts) != 2:
                        continue
                    
                    token, postings_str = parts
                    
                    # Process postings
                    postings = []
                    for posting in postings_str.split(','):
                        if not posting:
                            continue
                        try:
                            doc_id, freq = posting.rsplit(':', 1)  # Use rsplit to handle URLs with colons
                            postings.append((doc_id, int(freq)))
                        except (ValueError, IndexError):
                            continue
                    
                    if postings:
                        index[token] = postings
                        
                except Exception as e:
                    print(f"Error processing line: {line[:100]}...")
                    continue
                    
        print(f"Loaded {len(index)} tokens from index")
        return index
    
    except Exception as e:
        print(f"Error loading index: {e}")
        return {}

def compute_final_score(doc_id: str, basic_score: float, query_terms: List[str], 
                       pagerank_scores: Dict, hub_scores: Dict, 
                       auth_scores: Dict, anchor_texts: Dict) -> float:
    """
    Combine all ranking signals into a final score
    """
    # Get link-based scores (default to 0 if not found)
    pagerank = pagerank_scores.get(doc_id, 0)
    hub = hub_scores.get(doc_id, 0)
    auth = auth_scores.get(doc_id, 0)
    
    # Calculate anchor text score
    anchor_score = 0
    doc_anchors = anchor_texts.get(doc_id, [])
    for anchor in doc_anchors:
        matching_terms = sum(1 for term in query_terms if term in anchor.lower())
        anchor_score += matching_terms
    
    # Combine scores (weights can be adjusted)
    final_score = (
        0.4 * basic_score +     # Basic tf-idf
        0.3 * pagerank +        # PageRank
        0.1 * hub +            # HITS hub score
        0.1 * auth +           # HITS authority score
        0.1 * anchor_score     # Anchor text relevance
    )
    
    return final_score

def boolean_search(index: Dict, query: str, 
                  pagerank_scores: Dict={}, hub_scores: Dict={}, 
                  auth_scores: Dict={}, anchor_texts: Dict={}) -> Tuple[set, Dict]:
    """
    Perform boolean AND search with enhanced scoring
    """
    stemmer = PorterStemmer()
    
    # Tokenize and stem query terms
    query_terms = [stemmer.stem(term.lower()) for term in query.split()]
    
    result_docs = None
    basic_scores = {}
    final_scores = {}
    
    # Process each query term
    for term in query_terms:
        postings = index.get(term, [])
        docs_with_term = {doc_id: freq for doc_id, freq in postings}
        
        if result_docs is None:
            result_docs = set(docs_with_term.keys())
            basic_scores = docs_with_term.copy()
        else:
            result_docs &= set(docs_with_term.keys())
            # Update scores for remaining docs
            new_scores = {}
            for doc_id in result_docs:
                if doc_id in docs_with_term:
                    new_scores[doc_id] = basic_scores[doc_id] + docs_with_term[doc_id]
            basic_scores = new_scores
            
        if not result_docs:
            return set(), {}
    
    # Compute final scores using all ranking signals
    for doc_id in result_docs:
        final_scores[doc_id] = compute_final_score(
            doc_id,
            basic_scores[doc_id],
            query_terms,
            pagerank_scores,
            hub_scores,
            auth_scores,
            anchor_texts
        )
            
    return result_docs, final_scores

def search_helper(index, query: str, pagerank_scores: Dict={}, 
                 hub_scores: Dict={}, auth_scores: Dict={}, 
                 anchor_texts: Dict={}):
    """
    Process a single query and display results with enhanced scoring
    """
    start_time = time.time()
    matching_docs, scores = boolean_search(
        index, query, 
        pagerank_scores, hub_scores, 
        auth_scores, anchor_texts
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\nQuery: '{query}'")
    if matching_docs:
        ranked_docs = sorted(matching_docs, key=lambda doc: scores[doc], reverse=True)
        print(f"Found {len(ranked_docs)} matching document(s) in {elapsed_time:.3f} seconds:")
        
        # Display top 5 results with detailed scoring
        for i, doc in enumerate(ranked_docs[:5], 1):
            print(f"{i}. Document: {doc}")
            print(f"   Final Score: {scores[doc]:.3f}")
            if pagerank_scores:
                print(f"   PageRank: {pagerank_scores.get(doc, 0):.3f}")
            if hub_scores:
                print(f"   Hub Score: {hub_scores.get(doc, 0):.3f}")
            if auth_scores:
                print(f"   Authority Score: {auth_scores.get(doc, 0):.3f}")
            if doc in anchor_texts:
                print(f"   Anchor texts: {', '.join(anchor_texts[doc][:3])}...")
    else:
        print("No matching documents found.")

def run_queries(interactive: bool = False):
    """
    Load index and run queries with enhanced scoring
    """
    try:
        # Load the existing index
        print("Loading index...")
        start_time = time.time()
        index = load_index("final_index.txt")
        end_time = time.time()
        
        if not index:
            print("Error: Failed to load index or index is empty!")
            return
            
        print(f"Index loaded in {end_time - start_time:.2f} seconds")
        print(f"Index contains {len(index)} unique terms")

        # Initialize empty scoring dictionaries
        # In practice, these would be loaded from files
        pagerank_scores = {}
        hub_scores = {}
        auth_scores = {}
        anchor_texts = {}

        if interactive:
            print("\nEnter queries (type 'quit' to exit):")
            while True:
                query = input("\nQuery: ").strip()
                if query.lower() == 'quit':
                    break
                if query:
                    search_helper(index, query, pagerank_scores, 
                                hub_scores, auth_scores, anchor_texts)
        else:
            # Test queries
            queries = [
                "machine learning",
                "donald bren",
                "computer science",
                "artificial intelligence",
                "database systems",
                "cristina lopes contact info",
                "ACM club meeting",
                "masters program",
                "computing labs printer access",
                "fall winter spring courses"
            ]
            print("\nProcessing test queries...")
            for query in queries:
                search_helper(index, query, pagerank_scores, 
                            hub_scores, auth_scores, anchor_texts)

    except FileNotFoundError:
        print("Error: final_index.txt not found!")
        print("Make sure the index file exists in the current directory.")
    except Exception as e:
        print(f"Error during search: {e}")
        raise  # For debugging

if __name__ == "__main__":
    import sys
    # Run in interactive mode if --interactive flag is provided
    interactive_mode = "--interactive" in sys.argv
    run_queries(interactive_mode)