"""
Module for processing the Bhagavad Gita PDF and saving chunks to local storage.
"""
import os
import json
import hashlib
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LocalChunkStorage:
    """
    Class to handle storage and retrieval of text chunks and their embeddings
    in local file system instead of using a vector database like Pinecone.
    """
    def __init__(self, storage_dir: str, embedding_model: Any):
        """
        Initialize with storage directory and embedding model.
        
        Args:
            storage_dir: Directory to store chunks and embeddings
            embedding_model: Model to generate embeddings
        """
        self.storage_dir = storage_dir
        self.embedding_model = embedding_model
        self.index_file = os.path.join(storage_dir, "index.json")
        self.chunks_dir = os.path.join(storage_dir, "chunks")
        
        # Create directories if they don't exist
        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(self.chunks_dir, exist_ok=True)
        
        # Load or create index
        self.index = self._load_index()

    def _load_index(self) -> Dict[str, Any]:
        """Load the index file or create a new one if it doesn't exist."""
        if os.path.exists(self.index_file):
            with open(self.index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"chunks": [], "metadata": {"chunk_count": 0, "source": "The Bhagavad Gita"}}

    def _save_index(self) -> None:
        """Save the index file."""
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self.index, f, indent=2)

    def _generate_chunk_id(self, text: str) -> str:
        """Generate a unique ID for a chunk based on its content."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def add_chunks(self, chunks: List[str], metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Add chunks to storage with their embeddings.
        
        Args:
            chunks: List of text chunks
            metadata: Optional list of metadata dicts for each chunk
        """
        if metadata is None:
            metadata = [{"source": "The Bhagavad Gita"} for _ in chunks]
        
        for i, chunk in enumerate(chunks):
            # Generate chunk ID and embeddings
            chunk_id = self._generate_chunk_id(chunk)
            embedding = self.embedding_model.embed_query(chunk)
            
            # Save chunk data
            chunk_file = os.path.join(self.chunks_dir, f"{chunk_id}.json")
            chunk_data = {
                "id": chunk_id,
                "text": chunk,
                "embedding": embedding,
                "metadata": metadata[i] if i < len(metadata) else {"source": "The Bhagavad Gita"}
            }
            
            with open(chunk_file, 'w', encoding='utf-8') as f:
                json.dump(chunk_data, f)
            
            # Update index
            self.index["chunks"].append({
                "id": chunk_id,
                "metadata": metadata[i] if i < len(metadata) else {"source": "The Bhagavad Gita"}
            })
        
        # Update metadata
        self.index["metadata"]["chunk_count"] = len(self.index["chunks"])
        self._save_index()
        logger.info(f"Added {len(chunks)} chunks to local storage")

    def similarity_search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for chunks similar to the query.
        
        Args:
            query: The search query
            k: Number of results to return
            
        Returns:
            List of dictionaries containing chunk data and similarity score
        """
        if not self.index["chunks"]:
            logger.warning("No chunks in storage to search")
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.embed_query(query)
        
        # Calculate similarity for all chunks
        results = []
        for chunk_info in self.index["chunks"]:
            chunk_id = chunk_info["id"]
            chunk_file = os.path.join(self.chunks_dir, f"{chunk_id}.json")
            
            if os.path.exists(chunk_file):
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    chunk_data = json.load(f)
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, chunk_data["embedding"])
                results.append({
                    "id": chunk_id,
                    "text": chunk_data["text"],
                    "metadata": chunk_data["metadata"],
                    "similarity": similarity
                })
        
        # Sort by similarity and take top k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:k]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as string
    """
    logger.info(f"Extracting text from {pdf_path}")
    reader = PdfReader(pdf_path)
    text = ""
    
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    logger.info(f"Extracted {len(text)} characters from PDF")
    return text


def process_pdf_to_chunks(pdf_path: str, storage_dir: str) -> None:
    """
    Process PDF file, split into chunks, and store locally.
    
    Args:
        pdf_path: Path to the PDF file
        storage_dir: Directory to store chunks
    """
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    
    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks = splitter.split_text(text)
    logger.info(f"Split PDF into {len(chunks)} chunks")
    
    # Initialize embedding model
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"  # This is a small but effective model
        )
        logger.info("Using sentence-transformers/all-MiniLM-L6-v2 embedding model")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        raise
    
    # Create storage and save chunks
    storage = LocalChunkStorage(storage_dir, embeddings)
    
    # Generate basic metadata (adding page numbers would be ideal but requires more work)
    metadata = [{"source": "The Bhagavad Gita", "chunk_index": i} for i in range(len(chunks))]
    
    # Store chunks with embeddings
    storage.add_chunks(chunks, metadata)
    logger.info("Successfully processed and stored PDF chunks")


if __name__ == "__main__":
    # This enables running the module directly to process the PDF
    pdf_path = "Data/The_Bhagavad_Gita.pdf"
    storage_dir = "data_blocks"
    process_pdf_to_chunks(pdf_path, storage_dir)