import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from vectordb import VectorDB

def test_chunking():
    # Mock init to avoid chroma/model load if we just want to test chunking?
    # Actually, we can just instantiate it. It might fail if no model, but let's see.
    # We'll skip model loading if possible or just let it run if environment is okay.
    # The environment might not have internet or keys, but SentenceTransformer downloads models.
    # We'll try to instantiate.
    
    try:
        vdb = VectorDB()
        text = "This is a sentence. " * 100
        chunks = vdb.chunk_text(text)
        print(f"Text length: {len(text)}")
        print(f"Number of chunks: {len(chunks)}")
        if len(chunks) > 0:
            print("Chunking successful.")
        else:
            print("Chunking returned empty list.")
    except Exception as e:
        print(f"Error during chunking test: {e}")

if __name__ == "__main__":
    test_chunking()
