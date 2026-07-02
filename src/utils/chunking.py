# src/utils/chunking.py
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.models.schemas import Document
from src.utils.logging import get_logger

logger = get_logger("alqac25")


class DocumentChunker:
    """Chunk documents using RecursiveCharacterTextSplitter optimized for Vietnamese legal texts."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 100):
        """
        Initialize the document chunker.
        
        Args:
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks for context continuity
        """
        # Separators optimized for Vietnamese legal documents
        separators = [
            "\n\n",                    # Paragraph breaks
            r"\n\d+\.",                # Numbered items (1., 2., etc.)
            "\\. ",                    # Sentence breaks
            "\n",                      # Line breaks
            " ",                       # Word breaks
            ""                         # Character level
        ]
        
        self.splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=True
        )
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk a list of documents.
        
        Args:
            documents: List of original documents
        
        Returns:
            List of chunked documents with modified IDs
        """
        chunked_docs = []
        
        for doc in documents:
            text = doc.text
            # Split text into chunks
            chunks = self.splitter.split_text(text)
            
            # Create chunked document objects
            for chunk_idx, chunk_text in enumerate(chunks):
                if len(chunk_text.strip()) > 0:  # Skip empty chunks
                    chunked_doc = Document(
                        law_id=doc.law_id,
                        article_id=f"{doc.article_id}_chunk{chunk_idx}",
                        text=chunk_text
                    )
                    chunked_docs.append(chunked_doc)
        
        logger.info(f"Chunked {len(documents)} documents into {len(chunked_docs)} chunks")
        return chunked_docs
    
    def chunk_document_dict(self, document_store: Dict[str, List[Document]]) -> Dict[str, List[Document]]:
        """
        Chunk documents organized in a dictionary by category.
        
        Args:
            document_store: Dictionary of documents keyed by category
        
        Returns:
            Dictionary with chunked documents
        """
        chunked_store = {}
        
        for category, docs in document_store.items():
            chunked_store[category] = self.chunk_documents(docs)
        
        return chunked_store