import os
import chromadb
from typing import List, Dict, Any
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer


class VectorDB:
    """
    A simple vector database wrapper using ChromaDB with HuggingFace embeddings.
    """

    def __init__(self, collection_name: str = None, embedding_model: str = None):
        """
        Initialize the vector database.

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: HuggingFace model name for embeddings
        """
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "rag_documents"
        )
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path="./chroma_db")

        # Load embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document collection"},
        )

        print(f"Vector database initialized with collection: {self.collection_name}")

    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """
        Simple text chunking by splitting on spaces and grouping into chunks.

        Args:
            text: Input text to chunk
            chunk_size: Approximate number of characters per chunk

        Returns:
            List of text chunks
        """
        # TODO: Implement text chunking logic
        # You have several options for chunking text - choose one or experiment with multiple:
        #
        # OPTION 1: Simple word-based splitting
        #   - Split text by spaces and group words into chunks of ~chunk_size characters
        #   - Keep track of current chunk length and start new chunks when needed
        #
        # OPTION 2: Use LangChain's RecursiveCharacterTextSplitter
        #   - from langchain_text_splitters import RecursiveCharacterTextSplitter
        #   - Automatically handles sentence boundaries and preserves context better
        #
        # OPTION 3: Semantic splitting (advanced)
        #   - Split by sentences using nltk or spacy
        #   - Group semantically related sentences together
        #   - Consider paragraph boundaries and document structure
        #
        # Feel free to try different approaches and see what works best!

        words = text.split()
        chunks = []
        current_chunk = []
        current_chunk_len = 0

        for word in words:
            if current_chunk_len + len(word) + 1 > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_chunk_len = len(word)
            else:
                current_chunk.append(word)
                current_chunk_len += len(word) + 1  # +1 for space

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector database.

        Args:
            documents: List of documents
        """
        # TODO: Implement document ingestion logic
        # HINT: Loop through each document in the documents list
        # HINT: Extract 'content' and 'metadata' from each document dict
        # HINT: Use self.chunk_text() to split each document into chunks
        # HINT: Create unique IDs for each chunk (e.g., "doc_0_chunk_0")
        # HINT: Use self.embedding_model.encode() to create embeddings for all chunks
        # HINT: Store the embeddings, documents, metadata, and IDs in your vector database
        # HINT: Print progress messages to inform the user

        print(f"Processing {len(documents)} documents...")

        for doc_idx, doc in enumerate(documents):
            content = doc.page_content
            metadata = doc.metadata

            if not content:
                print(f"Skipping document {doc_idx} due to empty content.")
                continue

            chunks = self.chunk_text(content)
            
            chunk_ids = []
            chunk_metadatas = []
            chunk_contents = []

            for chunk_idx, chunk in enumerate(chunks):
                unique_id = f"doc_{doc_idx}_chunk_{chunk_idx}"
                chunk_ids.append(unique_id)
                chunk_metadatas.append({**metadata, "chunk_id": unique_id, "document_id": doc_idx})
                chunk_contents.append(chunk)

            if chunk_contents:
                embeddings = self.embedding_model.encode(chunk_contents).tolist()
                self.collection.add(
                    embeddings=embeddings,
                    documents=chunk_contents,
                    metadatas=chunk_metadatas,
                    ids=chunk_ids
                )
                print(f"Added {len(chunk_contents)} chunks for document {doc_idx}.")

        print("Documents added to vector database")

    def search(self, query: str, n_results: int = 5) -> List[Document]:
        """
        Search for similar documents in the vector database.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            Dictionary containing search results with keys: 'documents', 'metadatas', 'distances', 'ids'
        """
        # TODO: Implement similarity search logic
        # HINT: Use self.embedding_model.encode([query]) to create query embedding
        # HINT: Convert the embedding to appropriate format for your vector database
        # HINT: Use your vector database's search/query method with the query embedding and n_results
        # HINT: Return a dictionary with keys: 'documents', 'metadatas', 'distances', 'ids'
        # HINT: Handle the case where results might be empty

        query_embedding = self.embedding_model.encode([query]).tolist()

        try:
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
        except Exception as e:
            print(f"Error during ChromaDB query: {e}")
            return {
                "documents": [],
                "metadatas": [],
                "distances": [],
                "ids": [],
            }

        # ChromaDB returns a list of lists for each key, so we flatten them
        return [Document(page_content=doc, metadata=meta) for doc, meta in zip(results.get('documents', [[]])[0], results.get('metadatas', [[]])[0])]
