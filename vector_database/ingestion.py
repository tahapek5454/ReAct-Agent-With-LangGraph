from typing import List
from dotenv import load_dotenv, get_key
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_openai import AzureOpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_core.documents import Document
from langchain.vectorstores.base import VectorStoreRetriever
from langchain_community.document_loaders import PyPDFLoader

class QdrantVectorDatabaseService:
    """Implementation of Vector Database Service using Qdrant""" 
    def __init__(self):
        load_dotenv()
        self.qdrant_host = get_key(".env", "QDRANT_HOST")
        self.qdrant_port = int(get_key(".env", "QDRANT_HOST_PORT") or "6333")
        self.embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-large")
           
    def create_collection(self, collection_name: str, vector_size: int = 3072) -> None:
        """Create a new collection in Qdrant"""
        client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port, https=False)
        
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            print(f"Collection '{collection_name}' created successfully.")
        except Exception as e:
            print(f"Error creating collection: {e}")
   
    def get_collections_names(self) -> List[str]:
        """List all collections in Qdrant"""
        try:
            client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port, https=False)
            collectionsResponse = client.get_collections()
            return [col.name for col in collectionsResponse.collections]
        except Exception as e:
            print(f"Error retrieving collections: {e}")
            return [] 
                
    def get_retriever(self, collection_name: str, https: bool = False) -> VectorStoreRetriever | None:
        """Get retriever for the specified collection"""
        try:
            qdrant = QdrantVectorStore.from_existing_collection(
                collection_name,
                self.embeddings,
                url=f"http{'s' if https else ''}://{self.qdrant_host}:{self.qdrant_port}",
                prefer_grpc=False,
            )
            return qdrant.as_retriever()
        except Exception as e:
            print(f"Error getting retriever: {e}")
            return None
    
    def process_pdf_and_add_to_collection(self, pdf_path: str, collection_name: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> QdrantVectorStore | None:
        """Load PDF, split into chunks, and add to collection"""
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=chunk_size,     
                chunk_overlap=chunk_overlap,   
                separators=["\n\n", "\n", " ", ""] 
            )
            
            chunks = text_splitter.split_documents(documents)
            
            print(f"PDF {len(documents)} sayfa içeriyor ve {len(chunks)} chunk'a bölündü.")
            
            return self.add_documents_to_collection(chunks, collection_name)
        
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return None
        
    def add_documents_to_collection(self, documents: List[Document], collection_name: str) -> QdrantVectorStore | None:
        """Add documents to the specified collection"""
        try:
            qdrant = QdrantVectorStore.from_documents(
                documents,
                self.embeddings,
                url=f"http://{self.qdrant_host}:{self.qdrant_port}",
                collection_name=collection_name,
                prefer_grpc=False,
            )
            print(f"Documents added to collection '{collection_name}' successfully.")
            return qdrant
        except Exception as e:
            print(f"Error adding documents to collection: {e}")
            return None

    def format_docs(self, docs):
        """Format documents into a single string"""
        return "\n\n".join(doc.page_content for doc in docs)







