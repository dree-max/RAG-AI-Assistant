import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from app import RAGAssistant, load_documents
from utils import setup_logger

logger = setup_logger()

def test_e2e():
    print("Initializing Assistant...")
    assistant = RAGAssistant()
    
    print("Loading documents...")
    docs = load_documents()
    print(f"Loaded {len(docs)} docs.")
    
    # Filter for the specific agent doc to be sure
    agent_docs = [d for d in docs if "Introduction to Agents" in d.page_content]
    if agent_docs:
        print(f"Found 'Introduction to Agents' content ({len(agent_docs)} parts).")
    else:
        print("WARNING: Did NOT find 'Introduction to Agents' content in loaded docs.")
        
    print("Ingesting documents (this may take time)...")
    assistant.add_documents(docs)
    
    print("Querying...")
    response = assistant.invoke("What is an agent?")
    print("\nResponse:")
    print(response)

if __name__ == "__main__":
    test_e2e()
