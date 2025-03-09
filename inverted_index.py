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
        # The inverted index maps each stemmed token to a list of postings.
        # Each posting is a tuple: (document_id, term_frequency).
        self.index: Dict[str, List[Tuple[str, int]]] = {}
        self.num_documents = 0
        self.stemmer = PorterStemmer()

    def _write_partial_index(self, filename: str):
        """
        Writes current in-memory index to a text file.
        Format: token|doc_id1:freq1,doc_id2:freq2,...
        Ensures proper URL encoding
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                for token, postings in self.index.items():
                    if not postings:
                        continue
                    
                    # Create postings string with proper URL handling
                    valid_postings = []
                    for doc_id, freq in postings:
                        if doc_id and freq > 0:
                            # Encode the URL if it contains special characters
                            safe_doc_id = doc_id.replace(':', '%3A')
                            valid_postings.append(f"{safe_doc_id}:{freq}")
                    
                    if valid_postings:
                        postings_str = ','.join(valid_postings)
                        f.write(f"{token}|{postings_str}\n")
                        
        except Exception as e:
            print(f"Error writing partial index {filename}: {e}")
                        
        except Exception as e:
            print(f"Error writing partial index {filename}: {e}")
    
    def build_index_from_corpus(self, top_level_directory: str, partial_chunk_size: int = 1000):
        """
        Builds index by processing JSON files and writing partial indexes as txt files.
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
                if filename.endswith('.json'):  # Keep looking for JSON files
                    file_path = os.path.join(domain_path, filename)
                    print(f"Indexing file: {file_path}")
                    self._process_json_file(file_path)
                    current_chunk_count += 1

                    if current_chunk_count >= partial_chunk_size:
                        partial_counter += 1
                        partial_filename = f"partial_index_{partial_counter}.txt"
                        print(f"Creating partial index {partial_counter} with {current_chunk_count} documents")
                        self._write_partial_index(partial_filename)
                        self.index = {}
                        current_chunk_count = 0

        # Handle remaining documents
        if current_chunk_count > 0:
            partial_counter += 1
            partial_filename = f"partial_index_{partial_counter}.txt"
            print(f"Creating final partial index with {current_chunk_count} documents")
            self._write_partial_index(partial_filename)
            self.index = {}

        print(f"Indexing complete. Created {partial_counter} partial indexes.")

    def index_document(self, doc_id: str, html_content: str):
        """
        Extracts visible text from the HTML content, tokenizes, stems, and updates the inverted index.
        """
        # Parse HTML to handle broken or malformed tags gracefully
        soup = BeautifulSoup(html_content, 'html.parser')
        text = soup.get_text(separator=' ')
        
        # Log a snippet of the extracted text
        print(f"Extracted text from document {doc_id}: {text[:60]}...")

        # Tokenize using NLTK
        tokens = nltk_tokenize(text)
        print(f"Tokenized {len(tokens)} tokens from document {doc_id}.")

        # Apply Porter Stemmer to each token
        stemmed_tokens = [self.stemmer.stem(token) for token in tokens]

        # Calculate term frequency for each stemmed token
        term_freq: Dict[str, int] = {}
        for token in stemmed_tokens:
            term_freq[token] = term_freq.get(token, 0) + 1

        # Update the inverted index
        for token, freq in term_freq.items():
            if token not in self.index:
                self.index[token] = []
            self.index[token].append((doc_id, freq))

        self.num_documents += 1
        print(f"Indexed document {doc_id} with {len(term_freq)} unique tokens.")


    def _process_json_file(self, file_path: str):
        """
        Loads the JSON, extracts 'url' and 'content', and indexes the HTML content.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except UnicodeDecodeError:
            # If UTF-8 fails, try a different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return

        # data should have fields: "url", "content", "encoding"
        url = data.get('url', file_path)
        html_content = data.get('content', '')

        # Use the URL or filename as the doc_id
        doc_id = url

        if not html_content.strip():
            print(f"Warning: No content in file {file_path}")
        else:
            print(f"Indexing document with URL: {doc_id}")

        # Index the document
        self.index_document(doc_id, html_content)

    def save_index(self, output_file: str):
        """
        Saves the inverted index as a JSON file. Posting tuples are converted to lists
        for JSON serialization.
        """
        print(f"Saving inverted index to {output_file}...")
        serializable_index = {
            token: [[doc_id, freq] for (doc_id, freq) in postings]
            for token, postings in self.index.items()
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_index, f, indent=4)
        print("Inverted index saved.")

if __name__ == '__main__':
    # Replace with the directory that contains your domain folders
    corpus_directory = "C:\\Users\\Owner\\Downloads\\developer\\DEV"

    #Run for Daanesh
    corpus_directory = "C:\\Users\\DanBo\\Downloads\\developer\\DEV"

    
    # Name of the output JSON file
    output_index_file = "inverted_index_nltk.json"

    print("Starting index build...")
    # Create the indexer and build the index
    indexer = InvertedIndex()
    indexer.build_index_from_corpus(corpus_directory)

    print(f"Number of documents indexed: {indexer.num_documents}")
    print(f"Number of unique tokens: {len(indexer.index)}")

    # Save the index to a JSON file
    indexer.save_index(output_index_file)
    print("Index build complete.")
