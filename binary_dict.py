def build_auxiliary_dictionary(index_file_path: str) -> dict:
    """
    Builds an auxiliary term dictionary from a final inverted index file.
    The dictionary maps each term to the byte offset at which its posting list starts.
    
    The final index file is assumed to be line-based, with each line in the format:
        term|[posting list]
    For example:
        apple|[("doc1",5),("doc2",2)]
    
    Parameters:
        index_file_path (str): Path to the final inverted index file.
    
    Returns:
        dict: A mapping from term (str) to byte offset (int).
    """
    auxiliary_dict = {}
    print(f"DEBUG: Opening index file: {index_file_path}")
    
    # Open the file in read mode; this creates a file handle without loading the full file.
    with open(index_file_path, 'r', encoding='utf-8') as f:
        while True:
            # Record the current byte offset using f.tell()
            offset = f.tell()
            # Read the next line from the file
            line = f.readline()
            # Debug: Print the current offset and the raw line read
            print(f"DEBUG: Offset: {offset}, Raw Line: '{line.rstrip()}'")
            
            # If no line is returned, we have reached the end of the file
            if not line:
                print("DEBUG: End of file reached")
                break
            
            # Remove any trailing whitespace (including newline characters)
            line = line.strip()
            if not line:
                print("DEBUG: Skipping empty line")
                continue
            
            # Split the line by the delimiter (here we assume the delimiter is '|')
            parts = line.split('|', 1)
            if len(parts) < 2:
                print(f"DEBUG: Malformed line at offset {offset}: '{line}'")
                continue  # Skip malformed lines
            
            # Extract the term (token) from the first part
            term = parts[0].strip()
            # Debug: Print extracted term and its offset
            print(f"DEBUG: Extracted term: '{term}' at offset: {offset}")
            # Store the term and its corresponding byte offset in the auxiliary dictionary
            auxiliary_dict[term] = offset
            
    print("DEBUG: Finished building auxiliary dictionary")
    print(f"DEBUG: Final dictionary: {auxiliary_dict}")
    return auxiliary_dict

# Example usage:
aux_dict = build_auxiliary_dictionary("final_index.txt")
