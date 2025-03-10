from collections import defaultdict
import math
from bs4 import BeautifulSoup

# link_analyzer.py
class LinkAnalyzer:
    def __init__(self, domain="ics.uci.edu"):
        self.graph = defaultdict(list)
        self.reverse_graph = defaultdict(list)
        self.anchor_texts = defaultdict(list)
        self.domain = domain  # Base domain to filter URLs
        self.all_urls = set()  # Track all valid URLs

    def is_valid_url(self, url: str) -> bool:
        """Check if URL belongs to our domain"""
        return self.domain in url.lower()

    def process_page(self, url: str, html_content: str):
        """Extract links and anchor text"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Add source URL to tracked URLs if valid
        if self.is_valid_url(url):
            self.all_urls.add(url)
        
        for a in soup.find_all('a', href=True):
            target_url = a['href']
            anchor_text = a.get_text(strip=True)
            
            # Only process internal links
            if not self.is_valid_url(target_url):
                continue
                
            self.all_urls.add(target_url)
            self.graph[url].append(target_url)
            self.reverse_graph[target_url].append(url)
            
            if anchor_text:
                self.anchor_texts[target_url].append(anchor_text)

    def compute_hits(self, iterations=50):
        """Compute HITS hub and authority scores"""
        # Initialize scores for valid URLs only
        hub_scores = {url: 1.0 for url in self.all_urls}
        auth_scores = {url: 1.0 for url in self.all_urls}
        
        for _ in range(iterations):
            # Update authority scores
            new_auth = {url: sum(hub_scores[inlink] 
                               for inlink in self.reverse_graph[url]
                               if inlink in hub_scores)  # Check if inlink is valid
                       for url in self.all_urls}
            
            # Update hub scores
            new_hub = {url: sum(auth_scores[outlink] 
                              for outlink in self.graph[url]
                              if outlink in auth_scores)  # Check if outlink is valid
                      for url in self.all_urls}
            
            # Normalize scores
            auth_norm = math.sqrt(sum(score * score for score in new_auth.values()))
            hub_norm = math.sqrt(sum(score * score for score in new_hub.values()))
            
            if auth_norm > 0:
                auth_scores = {url: score/auth_norm 
                             for url, score in new_auth.items()}
            if hub_norm > 0:
                hub_scores = {url: score/hub_norm 
                            for url, score in new_hub.items()}
            
        return hub_scores, auth_scores

    def compute_pagerank(self, damping=0.85, iterations=50):
        """Compute PageRank scores"""
        if not self.all_urls:
            return {}
            
        # Initialize scores for valid URLs only
        scores = {url: 1/len(self.all_urls) for url in self.all_urls}
        
        for _ in range(iterations):
            new_scores = {}
            for url in self.all_urls:
                score = (1-damping)/len(self.all_urls)
                
                # Only consider valid inlinks
                valid_inlinks = [inlink for inlink in self.reverse_graph[url] 
                               if inlink in self.all_urls]
                
                for inlink in valid_inlinks:
                    valid_outlinks = [outlink for outlink in self.graph[inlink] 
                                    if outlink in self.all_urls]
                    if valid_outlinks:  # Avoid division by zero
                        score += damping * scores[inlink] / len(valid_outlinks)
                        
                new_scores[url] = score
            scores = new_scores
            
        return scores