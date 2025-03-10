from collections import defaultdict
from typing import List, Tuple, Dict, Set
import math
from nltk.stem import PorterStemmer

class SearchStructure:
    def __init__(self):
        # Main auxiliary dictionary
        self.term_offsets = {}  # token -> byte offset in final_index.txt
        
        # Document statistics
        self.doc_frequencies = {}  # token -> number of docs containing token
        self.total_docs = 0
        
        # Ranking components
        self.pagerank_scores = {}  # url -> pagerank score
        self.hub_scores = {}      # url -> hub score
        self.auth_scores = {}     # url -> authority score
        
        # Enhanced features
        self.position_index = defaultdict(lambda: defaultdict(list))  # token -> {doc_id -> [positions]}
        self.ngram_index = {
            2: defaultdict(set),  # bigrams -> set of doc_ids
            3: defaultdict(set)   # trigrams -> set of doc_ids
        }
        self.anchor_texts = defaultdict(list)  # url -> list of anchor texts
        
        self.stemmer = PorterStemmer()
        
    def build_auxiliary_structures(self, index_file: str, scores_file: str):
        """Build all auxiliary structures from final files"""
        print("Building auxiliary dictionary...")
        # Build main auxiliary dictionary
        with open(index_file, 'r', encoding='utf-8') as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                token = line.split('|', 1)[0]
                self.term_offsets[token] = offset
        
        print("Loading scores and auxiliary data...")
        # Load scores and other data
        with open(scores_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) < 2:
                    continue
                    
                record_type = parts[0]
                
                if record_type == 'df':
                    token, freq = parts[1:]
                    self.doc_frequencies[token] = int(freq)
                    
                elif record_type == 'pos':
                    token, doc_id, positions = parts[1:]
                    self.position_index[token][doc_id].extend(map(int, positions.split(',')))
                    
                elif record_type == 'ngram':
                    n, ngram, docs = parts[1:]
                    self.ngram_index[int(n)][ngram].update(docs.split(','))
                    
                elif record_type == 'link':
                    url, score = parts[1:]
                    self.pagerank_scores[url] = float(score)
                    
                elif record_type == 'hits':
                    url, hub, auth = parts[1:]
                    self.hub_scores[url] = float(hub)
                    self.auth_scores[url] = float(auth)
                    
                elif record_type == 'anchor':
                    url, texts = parts[1:]
                    self.anchor_texts[url].extend(texts.split(','))
        
        self.total_docs = len(set().union(*[set(docs) for docs in self.doc_frequencies.values()]))
        print(f"Loaded {len(self.term_offsets)} terms and {self.total_docs} documents")

    def get_postings(self, term: str) -> List[Tuple[str, float]]:
        """Get postings for a term using auxiliary dictionary"""
        if term not in self.term_offsets:
            return []
            
        with open('final_index.txt', 'r', encoding='utf-8') as f:
            f.seek(self.term_offsets[term])
            line = f.readline()
            _, postings_str = line.split('|', 1)
            return [
                (doc_id, float(freq))
                for doc_id, freq in [
                    posting.split(':')
                    for posting in postings_str.strip().split(',')
                ]
            ]

    def compute_tf_idf_score(self, doc_id: str, query_terms: List[str]) -> float:
        """Compute tf-idf score for document"""
        score = 0
        for term in query_terms:
            postings = self.get_postings(term)
            for posting_doc_id, freq in postings:
                if posting_doc_id == doc_id:
                    tf = 1 + math.log10(freq)
                    idf = math.log10(self.total_docs / (1 + self.doc_frequencies.get(term, 0)))
                    score += tf * idf
        return score

    def compute_position_score(self, doc_id: str, query_terms: List[str]) -> float:
        """Compute score based on proximity of query terms"""
        positions = defaultdict(list)
        for term in query_terms:
            if doc_id in self.position_index[term]:
                positions[term].extend(self.position_index[term][doc_id])
                
        if not positions:
            return 0
            
        # Find minimum distance between query terms
        min_distance = float('inf')
        for i, term1 in enumerate(query_terms[:-1]):
            for term2 in query_terms[i+1:]:
                if term1 in positions and term2 in positions:
                    for pos1 in positions[term1]:
                        for pos2 in positions[term2]:
                            distance = abs(pos1 - pos2)
                            min_distance = min(min_distance, distance)
                            
        if min_distance == float('inf'):
            return 0
            
        return 1.0 / (1.0 + min_distance)

    def compute_ngram_score(self, doc_id: str, query: str) -> float:
        """Compute score based on n-gram matches"""
        query_terms = query.lower().split()
        score = 0
        
        # Check bigrams
        if len(query_terms) >= 2:
            for i in range(len(query_terms)-1):
                bigram = f"{query_terms[i]} {query_terms[i+1]}"
                if doc_id in self.ngram_index[2].get(bigram, set()):
                    score += 0.5
                    
        # Check trigrams
        if len(query_terms) >= 3:
            for i in range(len(query_terms)-2):
                trigram = f"{query_terms[i]} {query_terms[i+1]} {query_terms[i+2]}"
                if doc_id in self.ngram_index[3].get(trigram, set()):
                    score += 1.0
                    
        return score

    def compute_anchor_score(self, doc_id: str, query_terms: List[str]) -> float:
        """Compute score based on query terms in anchor texts"""
        score = 0
        for anchor in self.anchor_texts.get(doc_id, []):
            anchor_terms = set(self.stemmer.stem(term.lower()) for term in anchor.split())
            matching_terms = len(set(query_terms) & anchor_terms)
            score += matching_terms
        return score

    def compute_final_score(self, doc_id: str, query: str, query_terms: List[str]) -> Tuple[float, Dict[str, float]]:
        """
        Compute final score combining all ranking signals
        Returns (final_score, component_scores)
        """
        # Compute component scores
        components = {
            'tf_idf': self.compute_tf_idf_score(doc_id, query_terms),
            'position': self.compute_position_score(doc_id, query_terms),
            'ngram': self.compute_ngram_score(doc_id, query),
            'pagerank': self.pagerank_scores.get(doc_id, 0),
            'hits_hub': self.hub_scores.get(doc_id, 0),
            'hits_auth': self.auth_scores.get(doc_id, 0),
            'anchor': self.compute_anchor_score(doc_id, query_terms)
        }
        
        # Combine scores with weights
        weights = {
            'tf_idf': 0.3,
            'position': 0.15,
            'ngram': 0.15,
            'pagerank': 0.2,
            'hits_hub': 0.05,
            'hits_auth': 0.05,
            'anchor': 0.1
        }
        
        final_score = sum(score * weights[component] 
                         for component, score in components.items())
                         
        return final_score, components