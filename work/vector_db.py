import os 
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb 
from chromadb.config import Settings
import uuid
import numpy as np
from typing import List, Dict, Tuple, Any
from dotenv import load_dotenv


class VectorDB:
    def __init__(self, collection_name:str = "pdf_info",
                 persist_directory:str = "vector_store"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self._initialize_store()
    def _initialize_store(self):
        try:
            # Create file
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            self.collection = self.client.get_or_create_collection(
                name = self.collection_name,
                metadata={"info":"PDF Embeddings"}
            )
            
            print(f"Vector_store initialized {self.collection_name}")
        except Exception as e:
            print(f"Store Not loaded {e}")
            self.client = None
            raise
    def add_docs(self, documents:List[Any], embeddings:np.ndarray):
        if len(documents) != len(embeddings):
            raise ValueError("Length of Documents and embeddings should be same")
        ids = []
        document_text = []
        metadatas = []
        embedding_list = []
        
        for i, (document, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"Doc_{uuid.uuid4().hex[:10]}_{i}"
            ids.append(doc_id)
            metadata = dict(document.metadata)
            metadata["id"] = i
            metadata["content"] = len(document.page_content)
            metadatas.append(metadata)
            
            document_text.append(document.page_content)
            embedding_list.append(embedding.tolist())
        
        try:
            self.collection.add(
                ids=ids,
                embeddings=embedding_list,
                metadatas=metadatas,
                documents=document_text
            )
            print(f"successfully added {len(documents)} documents to vector store")
            print(f"Total documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error adding docs to vector store: {e}")
            raise
