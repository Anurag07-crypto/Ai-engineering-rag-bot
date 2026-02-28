import os 
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb 
from chromadb.config import Settings
from embedding_manager import Embedding_manager
from vector_db import VectorDB
import numpy as np
from typing import List, Dict, Tuple, Any


# Retrivel Pipeline

class RAG:
    def __init__(self, vector_store:VectorDB, embedding_manager:Embedding_manager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
    
    def retrieve(self, query:str, top_k:int = 5, threshold:float = 0.0)-> List[Dict[str, Any]]:
        print(f"Top_k = {top_k}, threshold = {threshold}")
        try:
            query_embeddings = self.embedding_manager.generate([query])[0]
            
            results = self.vector_store.collection.query(
                query_embeddings = [query_embeddings.tolist()],
                n_results = top_k,
                include=["metadatas", "documents", "distances"]
            )
            
            retrieved_docs = []
            
            if results["documents"] and results["documents"][0]:
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]
                ids = results["ids"][0]
                
                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    similarity = 1 - distance
                    if similarity < threshold:
                        continue
                    retrieved_docs.append({
                        "ids":doc_id,
                        "content":document,
                        "metadata":metadata,
                        "distance":distance,
                        "rank":i+1
                    })
            if not retrieved_docs:
                print("Document not found")
                
            return retrieved_docs
        
        except Exception as e:
            print(f"error during retrivel |-> {e}")
            return []
        
