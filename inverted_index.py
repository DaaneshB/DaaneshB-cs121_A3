# inverted_index.py - Section 1
import hashlib
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
        '''
        self.position_index = defaultdict(lambda: defaultdict(list))  # token -> {doc_id -> [positions]}
        self.ngram_index = {
            2: defaultdict(set),  # bigrams -> [doc_ids]
            3: defaultdict(set)   # trigrams -> [doc_ids]
        }
        '''
        self.html_weights = {
            'title': 4.0,
            'header': 3.0,
            'bold': 2.0,
            'normal': 1.0
        }

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
    
    def index_document(self, doc_id: str, html_content: str, weighted_features: Dict = None):
        """
        Index a document with essential features only
        """
        print(f"Indexing document: {doc_id}")
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Get features if not provided
        if weighted_features is None:
            features = self.simhasher._extract_features(soup)
            weighted_features = self.simhasher._compute_feature_weights(features)
        
        try:
            # Process each token with its weight
            for token, weight in weighted_features.items():
                stemmed_token = self.stemmer.stem(token.lower())
                
                # Update inverted index with weight
                if stemmed_token not in self.index:
                    self.index[stemmed_token] = []
                self.index[stemmed_token].append((doc_id, weight))
                
                # Update document frequency
                if stemmed_token not in self.doc_frequencies:
                    self.doc_frequencies[stemmed_token] = 0
                self.doc_frequencies[stemmed_token] += 1
            
            # Process links
            self.link_analyzer.process_page(doc_id, html_content)
            
            # Update document count
            self.num_documents += 1
            
        except Exception as e:
            print(f"Error processing document {doc_id}: {str(e)}")




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
        Format: token|doc1:freq1,doc2:freq2,...
        """
        print(f"\nWriting partial index to {filename}")
        print(f"Current index size: {len(self.index)} tokens")
        
        try:
            entries_written = 0
            with open(filename, 'w', encoding='utf-8') as f:
                # Sort tokens for consistent output
                for token in sorted(self.index.keys()):
                    postings = self.index[token]
                    if not postings:
                        continue
                    
                    # Convert postings to string format
                    postings_str = ','.join(
                        f"{doc_id}:{freq}"
                        for doc_id, freq in postings
                        if doc_id and freq
                    )
                    
                    if postings_str:
                        line = f"{token}|{postings_str}\n"
                        f.write(line)
                        entries_written += 1
                        
                        # Debug output for first few entries
                        if entries_written <= 5:
                            print(f"Sample entry {entries_written}: {line.strip()}")

            print(f"Finished writing {entries_written} entries to {filename}")
            
            # Verify file was created and has content
            if os.path.exists(filename):
                file_size = os.path.getsize(filename)
                if file_size > 0:
                    print(f"Successfully created {filename} ({file_size/1024:.2f} KB)")
                else:
                    print(f"Warning: {filename} was created but is empty")
            else:
                print(f"Error: Failed to create {filename}")

        except Exception as e:
            print(f"Error writing partial index {filename}: {e}")
            print("Current index state:")
            print(f"Number of tokens: {len(self.index)}")
            # Print a sample token and its postings
            if self.index:
                sample_token = next(iter(self.index))
                print(f"Sample token '{sample_token}' postings: {self.index[sample_token]}")



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

    def _save_partial_scores(self, partial_counter: int):
        """
        Save essential scoring data only (document frequencies, links, anchors)
        """
        scores_filename = f"partial_scores_{partial_counter}.txt"
        print(f"Writing scores to {scores_filename}")
        
        try:
            with open(scores_filename, 'w', encoding='utf-8') as f:
                # Document frequencies (for tf-idf)
                for token, freq in self.doc_frequencies.items():
                    f.write(f"df|{token}|{freq}\n")
                
                # Link structure (for PageRank/HITS)
                for source, targets in self.link_analyzer.graph.items():
                    if targets:
                        unique_targets = list(set(targets))
                        f.write(f"link|{source}|{','.join(unique_targets)}\n")
                
                # Anchor texts
                for url, anchors in self.link_analyzer.anchor_texts.items():
                    if anchors:
                        unique_anchors = list(set(anchors))
                        f.write(f"anchor|{url}|{','.join(unique_anchors)}\n")

        except Exception as e:
            print(f"Error writing scores file {scores_filename}: {e}")

    def build_index_from_corpus(self, top_level_directory: str, partial_chunk_size: int = 1000):
        """
        Build index with essential features only (no positions or n-grams)
        """
        print(f"\nBuilding index from corpus in directory: {top_level_directory}")
        
        # Initialize counters
        current_chunk_count = 0
        partial_counter = 0
        stats = {
            'total_docs_seen': 0,
            'docs_indexed': 0,
            'duplicates_excluded': 0,
            'exact_duplicates': 0,
            'near_duplicates': 0,
            'unique_terms': set(),
            'links_processed': 0,
            'anchor_texts_found': 0
        }

        # Track duplicates
        self.document_hashes = {}  # doc_id -> SimHash

        for domain_folder in os.listdir(top_level_directory):
            domain_path = os.path.join(top_level_directory, domain_folder)
            if not os.path.isdir(domain_path):
                continue
                
            print(f"\nProcessing domain folder: {domain_folder}")
            
            for filename in os.listdir(domain_path):
                if not filename.endswith('.json'):
                    continue
                    
                file_path = os.path.join(domain_path, filename)
                stats['total_docs_seen'] += 1
                
                try:
                    # Load document
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    doc_id = data.get('url', file_path)
                    html_content = data.get('content', '')
                    
                    if not html_content.strip():
                        print(f"Warning: Empty content in {file_path}")
                        continue
                    
                    # Check for duplicates
                    doc_hash = self.simhasher.compute_document_hash(html_content)
                    is_duplicate = False
                    
                    for existing_id, existing_hash in self.document_hashes.items():
                        similarity = self.simhasher.compute_similarity(doc_hash, existing_hash)
                        if similarity['is_duplicate']:
                            stats['duplicates_excluded'] += 1
                            if similarity['similarity_score'] == 1.0:
                                stats['exact_duplicates'] += 1
                                print(f"Exact duplicate found: {doc_id} matches {existing_id}")
                            else:
                                stats['near_duplicates'] += 1
                                print(f"Near duplicate found: {doc_id} matches {existing_id}")
                                print(f"Similarity score: {similarity['similarity_score']:.3f}")
                            is_duplicate = True
                            break
                    
                    if is_duplicate:
                        continue
                    
                    # Process non-duplicate document
                    self.document_hashes[doc_id] = doc_hash
                    
                    # Index the document
                    initial_terms = len(self.index)
                    initial_links = len(self.link_analyzer.graph)
                    initial_anchors = sum(len(anchors) for anchors in self.link_analyzer.anchor_texts.values())
                    
                    self.index_document(doc_id, html_content)
                    
                    # Update statistics
                    stats['docs_indexed'] += 1
                    stats['unique_terms'].update(self.index.keys())
                    stats['links_processed'] += len(self.link_analyzer.graph) - initial_links
                    stats['anchor_texts_found'] += (
                        sum(len(anchors) for anchors in self.link_analyzer.anchor_texts.values()) - 
                        initial_anchors
                    )
                    
                    current_chunk_count += 1
                    
                    # Check if chunk size reached
                    if current_chunk_count >= partial_chunk_size:
                        partial_counter += 1
                        
                        # Write partial index
                        partial_filename = f"partial_index_{partial_counter}.txt"
                        print(f"\nWriting partial index {partial_counter}")
                        print(f"Documents in chunk: {current_chunk_count}")
                        print(f"Terms in index: {len(self.index)}")
                        
                        self._write_partial_index(partial_filename)
                        self._save_partial_scores(partial_counter)
                        
                        # Clear memory
                        self.index = {}
                        current_chunk_count = 0
                        
                        print(f"\nChunk {partial_counter} Statistics:")
                        print(f"Documents processed: {stats['docs_indexed']}")
                        print(f"Duplicates found: {stats['duplicates_excluded']}")
                        print(f"Links processed: {stats['links_processed']}")
                        print(f"Anchor texts found: {stats['anchor_texts_found']}")

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue

        # Handle remaining documents
        if current_chunk_count > 0:
            partial_counter += 1
            partial_filename = f"partial_index_{partial_counter}.txt"
            print(f"\nWriting final partial index")
            self._write_partial_index(partial_filename)
            self._save_partial_scores(partial_counter)
            self.index = {}

        print(f"\nFinal Indexing Statistics:")
        print(f"Total documents seen: {stats['total_docs_seen']}")
        print(f"Documents indexed: {stats['docs_indexed']}")
        print(f"Duplicates excluded: {stats['duplicates_excluded']}")
        print(f"  - Exact duplicates: {stats['exact_duplicates']}")
        print(f"  - Near duplicates: {stats['near_duplicates']}")
        print(f"Unique terms: {len(stats['unique_terms'])}")
        print(f"Links processed: {stats['links_processed']}")
        print(f"Anchor texts found: {stats['anchor_texts_found']}")
        print(f"Partial indexes created: {partial_counter}")

        return partial_counter

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

def main():
    # Set paths
    corpus_directory = "C:\\Users\\DanBo\\Downloads\\developer\\DEV"
    
    # Initialize indexer
    print("Starting test index build...")
    indexer = InvertedIndex()
    
    # Process first 300 documents (3 partial indexes of 100 each)
    doc_count = 0
    max_docs = 300  # Total documents to process
    
    for domain_folder in os.listdir(corpus_directory):
        if doc_count >= max_docs:
            break
            
        domain_path = os.path.join(corpus_directory, domain_folder)
        if not os.path.isdir(domain_path):
            continue
            
        print(f"\nProcessing domain folder: {domain_folder}")
        
        for filename in os.listdir(domain_path):
            if doc_count >= max_docs:
                break
                
            if filename.endswith('.json'):
                file_path = os.path.join(domain_path, filename)
                print(f"Processing file {doc_count + 1}/300: {file_path}")
                
                indexer.build_index_from_corpus(corpus_directory, partial_chunk_size=100)
                doc_count += 1
    
    print("\nIndex building complete!")
    print(f"Processed {doc_count} documents")
    print("Created files:")
    
    # Show created files
    partial_indexes = [f for f in os.listdir('.') if f.startswith('partial_index_')]
    partial_scores = [f for f in os.listdir('.') if f.startswith('partial_scores_')]
    
    print("\nPartial index files:")
    for f in partial_indexes:
        print(f"  - {f}")
    
    print("\nPartial score files:")
    for f in partial_scores:
        print(f"  - {f}")
    
    # Merge the files
    print("\nMerging partial indexes...")
    from merge import merge_partial_indexes
    merge_partial_indexes(partial_indexes, "final_index.txt")
    
    print("\nFinal files:")
    print("  - final_index.txt")
    print("  - ranking_scores.json")

if __name__ == "__main__":
    main()