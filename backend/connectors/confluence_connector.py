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
class ConfluenceConnector(DataSourceConnector):
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
                space = json_data['metadata']['space']
                self.load_from_confluence(space,data_id)
                #DocumentProcessor().process_document(message)
               # self.producer.send(self.kafka_topic, json.dumps(message).encode("utf-8"))
                

    def load_from_confluence(self, space_key: str, page_id: str) -> None:
        """
        Load and chunk content from a Confluence page, then store in Pinecone.
        
        Args:
            space_key: Confluence space key
            page_id: Confluence page ID
        """
        try:
            print(f"Loading content from Confluence page {page_id} in space {space_key}")
            
            # Get Confluence credentials using EnvUtils
            env_utils = EnvUtils()
            confluence_url = env_utils.get_required_env("CONFLUENCE_URL")
            username = env_utils.get_required_env("CONFLUENCE_USERNAME")
            api_token = env_utils.get_required_env("CONFLUENCE_API_TOKEN")
            
            # Construct API endpoint
            api_endpoint = f"{confluence_url}/rest/api/content/{page_id}?expand=body.storage,version"
            
            # Make API request
            response = requests.get(api_endpoint, auth=(username, api_token))
            response.raise_for_status()
            
            page_data = response.json()
            html_content = page_data['body']['storage']['value']
            page_title = page_data['title']
            
            # Convert HTML to text
            soup = BeautifulSoup(html_content, 'html.parser')
            text_content = soup.get_text(separator='\n', strip=True)
            
            # Create document
            document = Document(
                page_content=text_content,
                metadata={
                    'source': f"confluence/{space_key}/{page_title}"
                }
            )
            
            # Split into chunks
            chunks = self.text_splitter.split_documents([document])
            
            # Prepare vectors for Pinecone
            vectors = []
            for i, chunk in enumerate(chunks):
                # Generate embedding
                embedding = self.embedding_model.encode(chunk.page_content)
                
                # Create metadata matching exact file format
                metadata = {
                    'text': chunk.page_content,
                    'source': chunk.metadata['source'],
                    'chunk_id': i
                }
                
                # Prepare vector for upsert
                vectors.append((
                    f"confluence_{space_key}_{page_id}_{i}",  # ID
                    embedding.tolist(),  # Vector
                    metadata  # Metadata
                ))
            
            # Upsert to Pinecone in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
                
            print(f"Successfully loaded Confluence page {page_id}")
            print(f"Created {len(vectors)} vectors from page content")
            
        except requests.exceptions.RequestException as e:
            print(f"Error accessing Confluence API: {str(e)}")
            raise
        except Exception as e:
            print(f"Error processing Confluence content: {str(e)}")
            raise