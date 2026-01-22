import os
import sys
from langchain_community.document_loaders import PyPDFLoader

def check_pdf():
    data_dir = os.path.join(os.path.dirname(__file__), '../data')
    pdf_name = "Introduction to Agents (1).pdf"
    pdf_path = os.path.join(data_dir, pdf_name)
    
    print(f"Checking {pdf_path}...")
    
    if not os.path.exists(pdf_path):
        print("File not found!")
        return

    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        print(f"Loaded {len(docs)} pages.")
        
        if docs:
            print("Content sample from page 1:")
            print(docs[0].page_content[:500])
            
            total_len = sum(len(d.page_content) for d in docs)
            print(f"Total characters extracted: {total_len}")
        else:
            print("No pages loaded.")
            
    except Exception as e:
        print(f"Error loading PDF: {e}")

if __name__ == "__main__":
    check_pdf()
