import os
import sys
import asyncio
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.graph_transformers import LLMGraphTransformer
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from langchain_openai import ChatOpenAI
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.retrievers import VectorCypherRetriever
from langchain_neo4j import Neo4jGraph
from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
from neo4j_graphrag.indexes import create_vector_index
from langchain.chains.graph_qa.cypher import GraphCypherQAChain
from langchain.prompts import PromptTemplate
from langchain_neo4j import Neo4jVector
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from util.envutils import EnvUtils
class DatachimeGraphRAG:
    def __init__(self):
        self.env_utils = EnvUtils()
        NEO4J_URL = self.env_utils.get_env('NEO4J_URL')
        NEO4J_USERNAME =  self.env_utils.get_env('NEO4J_USERNAME')
        NEO4J_PASSWORD = self.env_utils.get_env('NEO4J_PASSWORD')
        os.environ['OPENAI_API_KEY']=self.env_utils.get_env('OPENAI_API_KEY')
        self.llm = ChatOpenAI(model_name="gpt-4o")
        self.graph = Neo4jGraph(
                url=NEO4J_URL,
                username=NEO4J_USERNAME,
                password=NEO4J_PASSWORD
        )
        self.transformer = LLMGraphTransformer(
            llm=self.llm,
            allowed_nodes=["Product", "Feature", "Customer", "Competitor", "Document", "Industry"],
            allowed_relationships=[
                    "HAS_FEATURE",
                    "INTERESTED_IN",
                    "COMPETES_WITH",
                    "DESCRIBES",
                    "OPERATES_IN",
                    "BELONGS_TO"
                ],
                node_properties=True,
                relationship_properties=True
        )
        print("Connected to Neo4j")
        
    async def create_product_graph(self,file_path):  
        print("Creating product graph...")
        uri = self.env_utils.get_env('NEO4J_URL')
        username = self.env_utils.get_env('NEO4J_USERNAME')
        password = self.env_utils.get_env('NEO4J_PASSWORD')
        neo4j_driver  = GraphDatabase.driver(uri, auth=(username, password))
        embedder = OpenAIEmbeddings()
        llm=OpenAILLM(
            model_name="gpt-4o-mini",
            model_params={
                "response_format": {"type": "json_object"}, # use json_object formatting for best results
                "temperature": 0 # turning temperature down for more deterministic results
         }
      )
        print("Connected to LLM")
        prompt_template = '''
            You are tasked with extracting structured information from business documents 
            and organizing it into a property graph for analytics and Q&A.

            Extract the entities (nodes) and relationships from the following text. Relationships are directed from the start node to the end node.

            Return the result in JSON:
            {{
            "nodes": [
                {{
                "id": "unique_id", 
                "label": "entity_type", 
                "properties": {{
                    "name": "entity_name",
                    "description": "entity_description"
                }}
                }}
            ],
            "relationships": [
                {{
                "type": "RELATIONSHIP_TYPE", 
                "start_node_id": "start_node_unique_id", 
                "end_node_id": "end_node_unique_id", 
                "properties": {{
                    "details": "relationship_description"
                }}
                }}
            ]
            }}

            - Use only the information in the text. Do not infer or add extra details.
            - Use these nodes: ["Product", "Feature", "Customer", "Competitor", "Document", "Industry"]
            - Use these relationships:
            - "HAS_FEATURE" (Product → Feature)
            - "INTERESTED_IN" (Customer → Product)
            - "COMPETES_WITH" (Competitor → Product)
            - "OPERATES_IN" (Product → Industry)
            - "BELONGS_TO" (Customer → Industry)
            - "DESCRIBES" (Document → Product)

            If the input is empty, return empty JSON. Assign unique IDs for all nodes and reuse them in relationships.

            **Input Text:**
            {text}
            '''
        allowed_nodes=["Product", "Feature", "Customer", "Competitor", "Document", "Industry"]
        allowed_relationships=[
                    "HAS_FEATURE",
                    "INTERESTED_IN",
                    "COMPETES_WITH",
                    "DESCRIBES",
                    "OPERATES_IN",
                    "BELONGS_TO"
                ]
        kg_builder_pdf = SimpleKGPipeline(
                llm=llm,
                driver=neo4j_driver,
                text_splitter=FixedSizeSplitter(chunk_size=200, chunk_overlap=40),
                embedder=embedder,
                entities=allowed_nodes,
                relations=allowed_relationships,
                prompt_template=prompt_template,
                from_pdf=True,
        )
        print(f"Processing : {file_path}")
        pdf_result = await kg_builder_pdf.run_async(file_path=file_path)
        print(f"Result: {pdf_result}")

    def process_pdf_and_create_graph(self,file_path):
        print("Processing PDF and creating graph...")
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
        docs = text_splitter.split_documents(pages)
        print("Loaded documents")
        #print(docs)
        lc_docs = []
        for doc in docs:
            lc_docs.append(Document(page_content=doc.page_content.replace("\n", ""), metadata={"source": file_path}))
        embeddings = OpenAIEmbeddings()
        llm = ChatOpenAI(model_name="gpt-4o-mini")
        #print(lc_docs)
        graph_documents = self.transformer.convert_to_graph_documents(lc_docs)
        print("Converted to graph documents")
        self.graph.add_graph_documents(graph_documents, include_source=True)
        print("Added graph documents to Neo4j")
        index = Neo4jVector.from_existing_graph(
                    embedding=embeddings,
                    url=self.env_utils.get_env('NEO4J_URL'),
                    username=self.env_utils.get_env('NEO4J_USERNAME'),
                    password=self.env_utils.get_env('NEO4J_PASSWORD'),
                    database="neo4j",
                    text_node_properties=["name", "description"], 
                    node_label="Document",
                    embedding_node_property="embedding", 
                    index_name="vector_index", 
                    keyword_index_name="entity_index", 
                    search_type="hybrid" 
                )
        print(f"Processing is Completed For: {file_path}")
    def query_graph(self, query):
        print("Querying graph...")
        uri = self.env_utils.get_env('NEO4J_URL')
        username = self.env_utils.get_env('NEO4J_USERNAME')
        password = self.env_utils.get_env('NEO4J_PASSWORD')
        embedder = OpenAIEmbeddings()
        driver = GraphDatabase.driver(uri, auth=(username, password))
        embeddings = OpenAIEmbeddings()
        retriever = VectorRetriever(
            driver,
            index_name="vector_index",
            embedder=embeddings,
        )
        rag = GraphRAG(retriever=retriever, llm=self.llm)
        response = rag.search(query_text=query, retriever_config={"top_k": 5})
# Example usage
if __name__ == "__main__":
    datachime_graph_rag = DatachimeGraphRAG()
    file_path = "C:\\datachime-demo\\AI-Retail-Suite.pdf"  # Replace with the path to your PDF file
    #datachime_graph_rag.process_pdf_and_create_graph(file_path)
    #asyncio.run(datachime_graph_rag.create_product_graph(file_path))
    datachime_graph_rag.query_graph("What is AI Retail Suite")
    # Example query
    #print("Graph schema:", graph.get_schema())
