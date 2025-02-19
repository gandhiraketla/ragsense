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
class AgenticRAGManager:
    def __init__(self):
        # Load environment variables
        self.env_utils = EnvUtils()
        # Initialize Pinecone
        self.pinecone_api_key = self.env_utils.get_required_env("PINECONE_API_KEY")
        self.openai_api_key = self.env_utils.get_required_env("OPENAI_API_KEY")
        self.perplexity_api_key = self.env_utils.get_required_env("PERPLEXITY_API_KEY")
        self.index_name = self.env_utils.get_required_env("PINECONE_INDEX")
        print(self.pinecone_api_key)
        print(self.index_name)
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
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4", api_key=self.openai_api_key)
        self.perplexity_client = OpenAI(
            api_key=self.perplexity_api_key,
            base_url="https://api.perplexity.ai"
        )
        #os.environ["PERPLEXITY_API_KEY"] = self.perplexity_api_key
        #os.environ["PERPLEXITY_MODEL"] = self.env_utils.get_required_env("PERPLEXITY_MODEL_NAME")
    
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
    def extract_citations(self, web_response) -> tuple[str, list[str]]:
        """
        Extract content and citations from Perplexity API response.
        
        Args:
            web_response: Raw response from Perplexity API
            
        Returns:
            Tuple of (content, citations)
        """
        try:
            # Extract the content from the message
            print(web_response)
            content = web_response.choices[0].message.content
            
            # Extract citations from message metadata if available
            citations = web_response.citations
            print(f"Extracted {len(citations)} citations from response")
            print(f"Citations: {citations}")
            return content, citations
        except Exception as e:
            self.logger.error(f"Error extracting citations: {str(e)}")
            return web_response.choices[0].message.content, []
    def process_query(self, query: str):
        print("Processing query")
        env_utils = EnvUtils()
        retrieved_docs  = self.vectorstore.similarity_search(query, k=3)
        retrieved_content = []
        internal_citations = []
        for i, doc in enumerate(retrieved_docs, 1):
            retrieved_content.append(doc.page_content)
            internal_citations.append({
                "number": i,
                "source": doc.metadata.get('source', 'Unknown source'),
                "text": doc.page_content[:200] + "..."  # Preview of the content
            })
        
        print(f"Retrieved {len(retrieved_content)} documents")
        print(f"First document preview: {retrieved_content[0][:200]}..." if retrieved_content else "No documents retrieved")
        print("Internal citations:")
        for citation in internal_citations:
          print(f"[{citation['number']}] {citation['source']}: {citation['text']}")
        evaluation_prompt = PromptTemplate.from_template("""Review the following query and context. If the context is sufficient to provide a complete and accurate answer, respond with just 1. 
            If additional information is needed, respond with a search query that would help find the missing information.
            
            Query: {query}
            
            Context: {retrieved_docs}
            
            Response (either 1 or a search query):"""
        )
        evaluation_chain = evaluation_prompt | self.llm
        eval_input = {
            "query": query,
            "retrieved_docs": "\n".join(retrieved_content) if retrieved_content else "No context available"
        }
        evaluation_result = evaluation_chain.invoke(eval_input)
        evaluation_response = evaluation_result.content.strip().upper()
        print(f"LLM evaluation response: {evaluation_response}")
        print(type(evaluation_response))
        web_context = None
        citations = None
        if evaluation_response!="1":
            print(f"Performing web search with query: {evaluation_response}")
            try:
                    web_results = self.perplexity_client.chat.completions.create(
                        model=self.env_utils.get_required_env("PERPLEXITY_MODEL_NAME"),
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that provides information based on web searches."},
                            {"role": "user", "content": evaluation_response}
                        ],
                        temperature=0.1
                    )
                    web_context, citations = self.extract_citations(web_results)
                    print("Web search completed successfully")
                    #print(f"Web context preview: {web_context[:200]}...")
            except Exception as e:
                    print(f"Error in web search: {str(e)}")
                    web_context = None
        #print("Skipping the web search, I have all the information I need")
        answer_prompt = PromptTemplate.from_template(
            """Answer the following query based on the provided context. Ensure your answer is comprehensive and accurate.
            
            Query: {query}
            
            Retrieved Context:
            {retrieved_docs}
            
            Additional Context:
            {web_context}
            
            Answer:"""
        )
        
        answer_chain = answer_prompt | self.llm
        
        # Create the input dictionary
        answer_input = {
            "query": query,
            "retrieved_docs": "\n".join(retrieved_content) if retrieved_content else "No primary context available",
            "web_context": f"Additional Web Context:\n{web_context}" if web_context else "No additional context available"
        }
        
        # Invoke the chain with the input
        answer_result = answer_chain.invoke(answer_input)
        answer = answer_result.content
        return {
            "answer": answer,
            "citations": citations,
            "internal_citations": internal_citations
        }
def main():
    rag_manager = AgenticRAGManager()
    #rag_manager.load_from_local("C:\\datachime-demo\\AI-Retail-Suite.pdf")
    #answer=rag_manager.process_query("What are the features of AI Retail Suite")
    #print(answer)
    rag_manager.load_from_confluence("~71202039d865b29006441a9b87f545b2b19e93", "1867795")

if __name__ == "__main__":
    main()