import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from app import load_documents
from langchain_core.documents import Document

def test_loading():
    # Create a dummy txt file
    data_dir = os.path.join(os.path.dirname(__file__), '../data')
    os.makedirs(data_dir, exist_ok=True)
    
    with open(os.path.join(data_dir, 'test.txt'), 'w', encoding='utf-8') as f:
        f.write("This is a test document.")
        
    documents = load_documents()
    print(f"Loaded {len(documents)} documents.")
    
    found = False
    for doc in documents:
        if "This is a test document." in doc.page_content:
            found = True
            print("Successfully loaded test.txt")
            break
            
    if not found:
        print("Failed to find test.txt content.")

if __name__ == "__main__":
    test_loading()
