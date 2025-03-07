import time
import search

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
        print("No documents found for the given query.")

def main():
    start = time.time()
    index_file = "inverted_index_nltk.json"
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

    
    
    
    