from collections import defaultdict
import math
from bs4 import BeautifulSoup

class LinkAnalyzer:
    def __init__(self):
        self.graph = defaultdict(list)
        self.reverse_graph = defaultdict(list)
        self.anchor_texts = defaultdict(list)
        
    def process_page(self, url: str, html_content: str):
        """Extract links and anchor text"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        for a in soup.find_all('a', href=True):
            target_url = a['href']
            anchor_text = a.get_text(strip=True)
            
            self.graph[url].append(target_url)
            self.reverse_graph[target_url].append(url)
            
            if anchor_text:
                self.anchor_texts[target_url].append(anchor_text)
    
    def compute_pagerank(self, damping=0.85, iterations=50):
        """Compute PageRank scores"""
        N = len(self.graph)
        if N == 0:
            return {}
            
        scores = {url: 1/N for url in self.graph}
        
        for _ in range(iterations):
            new_scores = {}
            for url in self.graph:
                score = (1-damping)/N
                for inlink in self.reverse_graph[url]:
                    if len(self.graph[inlink]) > 0:
                        score += damping * scores[inlink] / len(self.graph[inlink])
                new_scores[url] = score
            scores = new_scores
            
        return scores
    
    def compute_hits(self, iterations=50):
        """Compute HITS hub and authority scores"""
        hub_scores = {url: 1.0 for url in self.graph}
        auth_scores = {url: 1.0 for url in self.graph}
        
        for _ in range(iterations):
            new_auth = {url: sum(hub_scores[inlink] 
                               for inlink in self.reverse_graph[url])
                       for url in self.graph}
            
            new_hub = {url: sum(auth_scores[outlink] 
                              for outlink in self.graph[url])
                      for url in self.graph}
            
            auth_norm = math.sqrt(sum(score**2 for score in new_auth.values()))
            hub_norm = math.sqrt(sum(score**2 for score in new_hub.values()))
            
            if auth_norm > 0:
                auth_scores = {url: score/auth_norm 
                             for url, score in new_auth.items()}
            if hub_norm > 0:
                hub_scores = {url: score/hub_norm 
                            for url, score in new_hub.items()}
            
        return hub_scores, auth_scores

class InvertedIndex:
    def __init__(self):
        # Existing initializations...
        self.link_analyzer = LinkAnalyzer()
        self.pagerank_scores = {}
        self.hub_scores = {}
        self.auth_scores = {}
        
    def index_document(self, doc_id: str, html_content: str):
        # Existing indexing code...
        
        # Process links and anchor text
        self.link_analyzer.process_page(doc_id, html_content)
        
    def compute_link_scores(self):
        """Compute all link-based scores"""
        print("Computing PageRank scores...")
        self.pagerank_scores = self.link_analyzer.compute_pagerank()
        
        print("Computing HITS scores...")
        self.hub_scores, self.auth_scores = self.link_analyzer.compute_hits()