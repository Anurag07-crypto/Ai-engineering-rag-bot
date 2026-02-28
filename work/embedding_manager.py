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


class Embedding_manager:
    def __init__(self, model_name:str="BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self.model = None
        self._load_model()
    def _load_model(self):
        try:
            self.model = SentenceTransformer(model_name_or_path=self.model_name)
            print(f"{self.model_name} loaded from Hugging_face")
        except Exception as e:
            raise f"Model {self.model_name} not Loaded: error - {e}"
    def generate(self, text:List[str])->np.ndarray:
        if not self.model:
            raise "Model Not loaded"
        embeddings = self.model.encode(text)
        print(f"Embeddings {len(embeddings)} generated")
        return embeddings