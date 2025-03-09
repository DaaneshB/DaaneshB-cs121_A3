import os
import json
import sys
from typing import List, Dict, Tuple
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup

# Download NLTK tokenizers if not already present
nltk.download('punkt')
nltk.download('punkt_tab')

def nltk_tokenize(text: str) -> List[str]:
    """
    Tokenizes the input text into lowercase alphanumeric tokens using NLTK.
    This function splits text on punctuation, spaces, etc. and then filters out 
    any tokens that are not alphanumeric (so punctuation is removed).
    """
    # Tokenize: splits text into words and punctuation
    tokens = word_tokenize(text.lower())
    # Keep only alphanumeric tokens (removes punctuation)
    tokens = [token for token in tokens if token.isalnum()]
    return tokens

class InvertedIndex:
    def __init__(self):
        self.index: Dict[str, List[Tuple[str, int]]] = {}
        self.num_documents = 0
        self.stemmer = PorterStemmer()
        
        # Initialize enhanced features
        self.simhasher = EnhancedSimHash(
            hash_bits=64,
            ngram_range=(1, 3),
            threshold=0.8
        )
        self.link_analyzer = LinkAnalyzer()
        self.document_hashes = {}
        
        # Ranking scores
        self.pagerank_scores = {}
        self.hub_scores = {}
        self.auth_scores = {}

    def index_document(self, doc_id: str, html_content: str):
        """
        Process a single document: extract text, tokenize, stem, update index.
        Now includes position tracking and HTML structure awareness.
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract features using SimHash's feature extraction
        features = self.simhasher._extract_features(soup)
        
        # Process each feature type with position information
        for feature_type, tokens in features.items():
            weight_multiplier = {
                'title': 4.0,
                'headers': 3.0,
                'meta': 2.0,
                'content': 1.0
            }[feature_type]
            
            stemmed_tokens = [self.stemmer.stem(token) for token in tokens]
            
            # Update index with position and weight information
            for position, token in enumerate(stemmed_tokens):
                if token not in self.index:
                    self.index[token] = []
                
                # Store doc_id, frequency, position, and weight
                position_weight = weight_multiplier * (1.0 / (1 + 0.1 * position))
                self.index[token].append((doc_id, 1, position, position_weight))

        self.num_documents += 1

    def _process_json_file(self, file_path: str):
        """Process JSON file with duplicate detection and link analysis"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return

        doc_id = data.get('url', file_path)
        html_content = data.get('content', '')

        if not html_content.strip():
            print(f"Warning: No content in file {file_path}")
            return

        # Compute document hash and check for duplicates
        doc_hash = self.simhasher.compute_document_hash(html_content)
        
        is_duplicate = False
        for existing_id, existing_hash in self.document_hashes.items():
            similarity = self.simhasher.compute_similarity(doc_hash, existing_hash)
            if similarity['is_duplicate']:
                print(f"Duplicate detected: {doc_id} matches {existing_id}")
                print(f"Similarity score: {similarity['similarity_score']:.3f}")
                is_duplicate = True
                break

        if not is_duplicate:
            # Store hash and process document
            self.document_hashes[doc_id] = doc_hash
            print(f"Indexing document: {doc_id}")
            
            # Process links and anchor text
            self.link_analyzer.process_page(doc_id, html_content)
            
            # Index the document
            self.index_document(doc_id, html_content)

    def _write_partial_index(self, filename: str):
        """
        Write current index to file with enhanced information.
        Format: token|doc_id:freq:pos:weight,doc_id:freq:pos:weight,...
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                for token, postings in self.index.items():
                    if not postings:
                        continue
                    
                    # Create postings string with position and weight info
                    postings_str = ','.join(
                        f"{doc_id}:{freq}:{pos}:{weight}"
                        for doc_id, freq, pos, weight in postings
                    )
                    f.write(f"{token}|{postings_str}\n")
                    
        except Exception as e:
            print(f"Error writing partial index {filename}: {e}")

    def build_index_from_corpus(self, top_level_directory: str, partial_chunk_size: int = 100):
        """Build index with partial writing and link analysis"""
        print(f"Building index from corpus in directory: {top_level_directory}")
        current_chunk_count = 0
        partial_counter = 0
        current_chunk_documents = []

        for domain_folder in os.listdir(top_level_directory):
            domain_path = os.path.join(top_level_directory, domain_folder)
            if not os.path.isdir(domain_path):
                continue
                
            print(f"Processing domain folder: {domain_folder}")
            
            for filename in os.listdir(domain_path):
                if filename.endswith('.json'):
                    file_path = os.path.join(domain_path, filename)
                    print(f"Processing file: {file_path}")
                    
                    # Process the document
                    self._process_json_file(file_path)
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

        # Compute link-based scores
        print("Computing PageRank scores...")
        self.pagerank_scores = self.link_analyzer.compute_pagerank()
        
        print("Computing HITS scores...")
        self.hub_scores, self.auth_scores = self.link_analyzer.compute_hits()
        
        print(f"Indexing complete. Created {partial_counter} partial indexes.")