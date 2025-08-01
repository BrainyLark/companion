import re
import weaviate

def clean_and_chunk_text(text, filename: str, chunk_size=200, overlap=50):
    """
    Clean text and split it into chunks of specified word count
    
    Args:
        text (str): Raw text from PDF extraction
        chunk_size (int): Target number of words per chunk (default: 200)
        overlap (int): Number of words to overlap between chunks (default: 50)
    
    Returns:
        list: List of text chunks ready for vector database
    """
    
    # Clean the text - remove multiple newlines and extra whitespace
    cleaned_text = re.sub(r'\n+', ' ', text)  # Replace newlines with spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Replace multiple spaces with single space
    cleaned_text = cleaned_text.strip()  # Remove leading/trailing whitespace

    # Split into words
    words = cleaned_text.split()
    
    if len(words) == 0:
        return []
    
    chunks = []
    start_idx = 0
    
    while start_idx < len(words):
        # Get chunk_size words starting from start_idx
        end_idx = min(start_idx + chunk_size, len(words))
        chunk_words = words[start_idx:end_idx]
        chunk_text = ' '.join(chunk_words)
        
        chunks.append({
            "content": chunk_text,
            "app_id": "egune-test",
            "document_path": filename
        })
        
        # If this is the last chunk, break
        if end_idx >= len(words):
            break
            
        # Move start_idx forward, accounting for overlap
        start_idx = end_idx - overlap
        
        # Ensure we don't go backwards
        if start_idx <= 0:
            start_idx = end_idx

    return chunks