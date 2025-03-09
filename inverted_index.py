# inverted_index.py - Section 1
import os
import json
from typing import List, Dict, Tuple, Set
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
from collections import defaultdict
import math
from numpy import dot
from numpy.linalg import norm
import numpy as np
from link_analyzer import LinkAnalyzer
from sim_hasher import EnhancedSimHash

# Download required NLTK data
nltk.download('punkt', quiet=True)

def nltk_tokenize(text: str) -> List[str]:
    """
    Tokenizes text into lowercase alphanumeric tokens using NLTK.
    """
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalnum()]
    return tokens

class InvertedIndex:
    def __init__(self):
        # Main index: token -> list of (doc_id, frequency, positions, weight)
        self.index: Dict[str, List[Tuple[str, int, List[int], float]]] = {}
        
        # Document statistics
        self.num_documents = 0
        self.total_docs = 0
        self.doc_lengths = {}  # For document length normalization
        self.doc_frequencies = {}  # Document frequency of each term
        
        # Text processing
        self.stemmer = PorterStemmer()
        
        # Enhanced features
        self.simhasher = EnhancedSimHash(
            hash_bits=64,
            ngram_range=(1, 3),
            threshold=0.8
        )
        self.link_analyzer = LinkAnalyzer()
        
        # Document hashes for duplicate detection
        self.document_hashes = {}
        
        # Scoring components
        self.pagerank_scores = {}
        self.hub_scores = {}
        self.auth_scores = {}
        self.idf = {}  # Inverse document frequencies

    def compute_idf(self):
        """
        Compute inverse document frequency for all terms
        """
        self.idf = {}
        for term, postings in self.index.items():
            # Number of documents containing this term
            doc_freq = len(postings)
            # IDF = log(N/df) where N is total docs and df is doc frequency
            self.idf[term] = math.log10(self.total_docs / (doc_freq + 1))

    def compute_tf_idf_vector(self, text: str, is_query: bool = False) -> Dict[str, float]:
        """
        Compute tf-idf vector for a text string or document
        
        Args:
            text: The text to vectorize
            is_query: Whether this is a query (affects tf computation)
            
        Returns:
            Dictionary mapping terms to their tf-idf weights
        """
        # Get term frequencies
        term_freq = defaultdict(float)
        tokens = nltk_tokenize(text)
        stemmed_tokens = [self.stemmer.stem(token) for token in tokens]
        
        for token in stemmed_tokens:
            term_freq[token] += 1
            
        # Compute tf-idf scores
        tf_idf = {}
        for term, freq in term_freq.items():
            if term in self.idf:
                # For queries, often use binary or raw tf
                if is_query:
                    tf = 1.0 if freq > 0 else 0.0  # Binary tf
                else:
                    tf = 1 + math.log10(freq)  # Log normalization
                
                tf_idf[term] = tf * self.idf[term]
                
        return tf_idf

    def cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """
        Compute cosine similarity between two sparse vectors
        """
        # Get common terms
        common_terms = set(vec1.keys()) & set(vec2.keys())
        
        if not common_terms:
            return 0.0
            
        # Compute dot product for common terms
        dot_product = sum(vec1[term] * vec2[term] for term in common_terms)
        
        # Compute magnitudes
        norm1 = math.sqrt(sum(v*v for v in vec1.values()))
        norm2 = math.sqrt(sum(v*v for v in vec2.values()))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def index_document(self, doc_id: str, html_content: str):
        """
        Process a single document: extract text, tokenize, stem, update index.
        Includes position tracking, HTML structure awareness, and tf-idf computation.
        """
        # Check for duplicates
        doc_hash = self.simhasher.compute_document_hash(html_content)
        
        for existing_id, existing_hash in self.document_hashes.items():
            similarity = self.simhasher.compute_similarity(doc_hash, existing_hash)
            if similarity['is_duplicate']:
                print(f"Duplicate detected: {doc_id} matches {existing_id}")
                return
                
        self.document_hashes[doc_id] = doc_hash
        
        # Process links for link analysis
        self.link_analyzer.process_page(doc_id, html_content)
        
        # Parse HTML and extract features
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Track term frequencies for this document
        doc_term_freq = defaultdict(float)
        
        # Extract features with weights based on HTML structure
        features = {
            'title': self._extract_text(soup.title, weight=4.0) if soup.title else [],
            'headers': self._extract_headers(soup, weight=3.0),
            'meta': self._extract_meta(soup, weight=2.0),
            'content': self._extract_content(soup, weight=1.0)
        }
        
        # Process each feature type
        for feature_type, feature_data in features.items():
            for position, (token, weight) in enumerate(feature_data):
                stemmed_token = self.stemmer.stem(token)
                
                # Update term frequency with weight
                doc_term_freq[stemmed_token] += weight
                
                if stemmed_token not in self.index:
                    self.index[stemmed_token] = []
                    self.doc_frequencies[stemmed_token] = 0
                
                # Update posting
                posting_updated = False
                for i, posting in enumerate(self.index[stemmed_token]):
                    if posting[0] == doc_id:
                        freq, positions, curr_weight = posting[1:]
                        positions.append(position)
                        new_weight = max(curr_weight, weight)
                        self.index[stemmed_token][i] = (doc_id, freq + 1, positions, new_weight)
                        posting_updated = True
                        break
                
                if not posting_updated:
                    self.index[stemmed_token].append((doc_id, 1, [position], weight))
                    self.doc_frequencies[stemmed_token] += 1
        
        # Store document length (magnitude of tf vector)
        self.doc_lengths[doc_id] = math.sqrt(sum(freq * freq for freq in doc_term_freq.values()))
        
        self.num_documents += 1
        self.total_docs = self.num_documents



    def _extract_text(self, element, weight: float) -> List[Tuple[str, float]]:
        """
        Extract text with weight from HTML element or string
        """
        if not element:
            return []
        
        # If element is already a string, use it directly
        if isinstance(element, str):
            text = element
        else:
            # If it's a BeautifulSoup element, get its text
            text = element.get_text()
        
        tokens = nltk_tokenize(text)
        return [(token, weight) for token in tokens]

    def _extract_meta(self, soup: BeautifulSoup, weight: float) -> List[Tuple[str, float]]:
        """Extract text from meta tags"""
        meta = []
        for tag in soup.find_all('meta', {'name': ['description', 'keywords']}):
            if 'content' in tag.attrs:
                # Pass the content string directly
                meta.extend(self._extract_text(tag['content'], weight))
        return meta

    def _extract_headers(self, soup: BeautifulSoup, weight: float) -> List[Tuple[str, float]]:
        """Extract text from header tags"""
        headers = []
        for tag in soup.find_all(['h1', 'h2', 'h3']):
            headers.extend(self._extract_text(tag, weight))
        return headers

    def _extract_content(self, soup: BeautifulSoup, weight: float) -> List[Tuple[str, float]]:
        """Extract main content text"""
        # Remove script and style elements
        for script in soup(['script', 'style']):
            script.decompose()
        return self._extract_text(soup, weight)



    def _write_partial_index(self, filename: str):
        """
        Write current in-memory index to a text file.
        Format: token|doc_id:freq:positions:weight,doc_id:freq:positions:weight,...
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                for token, postings in self.index.items():
                    if not postings:
                        continue
                    
                    # Convert postings to string format
                    postings_str = ','.join(
                        f"{doc_id}:{freq}:{','.join(map(str,positions))}:{weight}"
                        for doc_id, freq, positions, weight in postings
                    )
                    f.write(f"{token}|{postings_str}\n")
                    
        except Exception as e:
            print(f"Error writing partial index {filename}: {e}")



    def _save_scores(self):
        """Save link analysis scores and other ranking data"""
        scores = {
            'pagerank': self.pagerank_scores,
            'hub': self.hub_scores,
            'authority': self.auth_scores,
            'anchor_texts': dict(self.link_analyzer.anchor_texts),
            'doc_lengths': self.doc_lengths,
            'doc_frequencies': self.doc_frequencies,
            'idf': self.idf
        }
        
        with open('ranking_scores.json', 'w') as f:
            json.dump(scores, f)

    def _load_scores(self):
        """Load previously computed ranking scores"""
        try:
            with open('ranking_scores.json', 'r') as f:
                scores = json.load(f)
                self.pagerank_scores = scores['pagerank']
                self.hub_scores = scores['hub']
                self.auth_scores = scores['authority']
                self.doc_lengths = scores['doc_lengths']
                self.doc_frequencies = scores['doc_frequencies']
                self.idf = scores['idf']
        except FileNotFoundError:
            print("No existing ranking scores found.")
        except Exception as e:
            print(f"Error loading ranking scores: {e}")

    def build_index_from_corpus(self, top_level_directory: str, partial_chunk_size: int = 100):
        """
        Build index by processing documents in chunks and writing partial indexes.
        Also computes and saves all ranking signals.
        """
        print(f"Building index from corpus in directory: {top_level_directory}")
        current_chunk_count = 0
        partial_counter = 0

        for domain_folder in os.listdir(top_level_directory):
            domain_path = os.path.join(top_level_directory, domain_folder)
            if not os.path.isdir(domain_path):
                continue
                
            print(f"Processing domain folder: {domain_folder}")
            
            for filename in os.listdir(domain_path):
                if filename.endswith('.json'):
                    file_path = os.path.join(domain_path, filename)
                    print(f"Processing file: {file_path}")
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                    except UnicodeDecodeError:
                        with open(file_path, 'r', encoding='latin-1') as f:
                            data = json.load(f)
                            
                    doc_id = data.get('url', file_path)
                    html_content = data.get('content', '')
                    
                    if html_content.strip():
                        self.index_document(doc_id, html_content)
                        current_chunk_count += 1
                    
                    # Check if chunk size reached
                    if current_chunk_count >= partial_chunk_size:
                        partial_counter += 1
                        partial_filename = f"partial_index_{partial_counter}.txt"
                        print(f"Writing partial index {partial_counter}")
                        self._write_partial_index(partial_filename)
                        self.index = {}
                        current_chunk_count = 0

        # Handle remaining documents
        if current_chunk_count > 0:
            partial_counter += 1
            partial_filename = f"partial_index_{partial_counter}.txt"
            print(f"Writing final partial index")
            self._write_partial_index(partial_filename)
            self.index = {}

        # Compute all ranking signals
        print("Computing IDF scores...")
        self.compute_idf()
        
        print("Computing PageRank scores...")
        self.pagerank_scores = self.link_analyzer.compute_pagerank()
        
        print("Computing HITS scores...")
        self.hub_scores, self.auth_scores = self.link_analyzer.compute_hits()
        
        # Save all scoring data
        self._save_scores()
        
        print(f"Indexing complete. Created {partial_counter} partial indexes.")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float, Dict[str, float]]]:
        """
        Search using combined ranking signals: tf-idf, PageRank, HITS, and anchor text.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of tuples (doc_id, final_score, component_scores)
        """
        # Compute query tf-idf vector
        query_vector = self.compute_tf_idf_vector(query, is_query=True)
        
        # Track document scores and components
        doc_scores = defaultdict(float)
        score_components = defaultdict(lambda: {'tf_idf': 0.0, 'pagerank': 0.0, 
                                              'hub': 0.0, 'auth': 0.0, 
                                              'anchor': 0.0})
        
        # First pass: compute tf-idf scores
        for term, query_weight in query_vector.items():
            if term in self.index:
                for doc_id, freq, positions, weight in self.index[term]:
                    # Compute document's tf-idf weight for this term
                    tf = 1 + math.log10(freq)
                    tf_idf = tf * self.idf[term]
                    
                    # Update cosine similarity
                    doc_scores[doc_id] += query_weight * tf_idf
                    score_components[doc_id]['tf_idf'] = doc_scores[doc_id]
        
        # Normalize by document lengths
        for doc_id in doc_scores:
            if doc_id in self.doc_lengths and self.doc_lengths[doc_id] > 0:
                doc_scores[doc_id] /= self.doc_lengths[doc_id]
                score_components[doc_id]['tf_idf'] = doc_scores[doc_id]
        
        # Get query terms for anchor text matching
        query_terms = set(self.stemmer.stem(term) for term in nltk_tokenize(query))
        
        # Combine with other ranking signals
        final_scores = {}
        for doc_id, cosine_score in doc_scores.items():
            # Get other ranking signals
            pagerank = self.pagerank_scores.get(doc_id, 0)
            hub = self.hub_scores.get(doc_id, 0)
            auth = self.auth_scores.get(doc_id, 0)
            
            # Store component scores
            score_components[doc_id]['pagerank'] = pagerank
            score_components[doc_id]['hub'] = hub
            score_components[doc_id]['auth'] = auth
            
            # Compute anchor text score
            anchor_score = 0
            for anchor in self.link_analyzer.get_anchor_texts(doc_id):
                anchor_terms = set(self.stemmer.stem(term) 
                                 for term in nltk_tokenize(anchor.lower()))
                anchor_score += len(query_terms & anchor_terms)
            score_components[doc_id]['anchor'] = anchor_score
            
            # Combine scores (adjust weights as needed)
            final_scores[doc_id] = (
                0.4 * cosine_score +    # tf-idf cosine similarity
                0.3 * pagerank +        # PageRank
                0.1 * hub +             # HITS hub score
                0.1 * auth +            # HITS authority score
                0.1 * anchor_score      # Anchor text relevance
            )
        
        # Sort by final score and get top-k
        ranked_results = sorted(
            [(doc_id, score, score_components[doc_id]) 
             for doc_id, score in final_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return ranked_results

if __name__ == '__main__':
    corpus_directory = "C:\\Users\\DanBo\\Downloads\\developer\\DEV"
    
    print("Starting index build...")
    indexer = InvertedIndex()
    
    # Build the index
    indexer.build_index_from_corpus(corpus_directory)
    
    # Example search
    query = "machine learning"
    results = indexer.search(query)
    
    print(f"\nTop results for query: '{query}'")
    for doc_id, score, components in results:
        print(f"\nDocument: {doc_id}")
        print(f"Final Score: {score:.4f}")
        print("Score Components:")
        print(f"  tf-idf: {components['tf_idf']:.4f}")
        print(f"  PageRank: {components['pagerank']:.4f}")
        print(f"  Hub Score: {components['hub']:.4f}")
        print(f"  Authority Score: {components['auth']:.4f}")
        print(f"  Anchor Score: {components['anchor']:.4f}")