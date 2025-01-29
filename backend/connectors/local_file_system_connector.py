import os
import requests
from langchain.schema import Document
from bs4 import BeautifulSoup
from typing import List, Optional
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
import sys
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from openai import OpenAI
from langchain.prompts import PromptTemplate
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from util.envutils import EnvUtils
from connectors.data_connector_base import DataSourceConnector
class LocalFileSystemConnector(DataSourceConnector):
    """
    Connector for monitoring local file system folders.
    """
    def __init__(self):
        # Load environment variables
        self.env_utils = EnvUtils()
        # Initialize Pinecone
        self.pinecone_api_key = self.env_utils.get_required_env("PINECONE_API_KEY")
        self.openai_api_key = self.env_utils.get_required_env("OPENAI_API_KEY")
        self.index_name = self.env_utils.get_required_env("PINECONE_INDEX")
        if not all([self.pinecone_api_key, self.index_name]):
            raise ValueError("Missing required environment variables")
        # Initialize embedding model
        self.embedding_model = SentenceTransformer("intfloat/multilingual-e5-large")
        self.langchain_embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        
        # Get or create index
        self.index = self.pc.Index('ragindex')
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        self.vectorstore = PineconeVectorStore(
            index=self.index,
            embedding=self.langchain_embeddings,
            text_key="text"
        )
        
    def load_data(self, json_data: dict):
                data_id = json_data['data_id']
                self.load_from_local(data_id)
                #DocumentProcessor().process_document(message)
               # self.producer.send(self.kafka_topic, json.dumps(message).encode("utf-8"))
                

    def load_from_local(self, file_path: str) -> None:
        """
        Load and chunk a document from local storage, then store in Pinecone.
        
        Args:
            file_path: Path to the local document
        """
        # Determine file type and load accordingly
        print(f"Loading document from {file_path}")
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path)
        else:
            raise ValueError("Unsupported file type. Only PDF and TXT files are supported.")
        
        # Load and chunk the document
        document = loader.load()
        chunks = self.text_splitter.split_documents(document)
        
        # Prepare vectors for Pinecone
        vectors = []
        for i, chunk in enumerate(chunks):
            # Generate embedding
            embedding = self.embedding_model.encode(chunk.page_content)
            
            # Create metadata
            metadata = {
                'text': chunk.page_content,
                'source': file_path,
                'chunk_id': i
            }
            
            # Add any additional metadata from the document
            if hasattr(chunk, 'metadata'):
                metadata.update(chunk.metadata)
            
            # Prepare vector for upsert
            vectors.append((
                f"{os.path.basename(file_path)}_{i}",  # ID
                embedding.tolist(),  # Vector
                metadata  # Metadata
            ))
        
        # Upsert to Pinecone in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
        print(f"Document {file_path} loaded and stored in Pinecone.")