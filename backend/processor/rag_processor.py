import json
import sys
import os
from datetime import datetime
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from util.envutils import EnvUtils
from haystack.utils import Secret
from haystack.components.converters import PyPDFToDocument
from haystack.components.converters import PPTXToDocument
from haystack.components.converters import TextFileToDocument
from haystack.components.converters.docx import DOCXToDocument, DOCXTableFormat
from haystack import Pipeline
from pathlib import Path
from haystack.components.builders import PromptBuilder
from haystack.components.writers import DocumentWriter
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.generators import OpenAIGenerator
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack.components.generators import HuggingFaceAPIGenerator
from haystack.components.generators import OpenAIGenerator
import logging
from haystack import tracing
from haystack.tracing.logging_tracer import LoggingTracer

class DocumentProcessor:
    def __init__(self):
        env_utils = EnvUtils()
        os.environ['PINECONE_API_KEY'] = env_utils.get_required_env('PINECONE_API_KEY')
        self.document_store = PineconeDocumentStore(
        index=env_utils.get_required_env('PINECONE_INDEX'),
        metric="cosine",
        dimension=1024,
        spec={"serverless": {"region": "us-east-1", "cloud": "aws"}},
       )
        self.document_embedder = SentenceTransformersDocumentEmbedder(
            model="intfloat/multilingual-e5-large"
) 
    def get_file_extension(self,file_path):
        _, extension = os.path.splitext(file_path)
        return extension[1:] if extension else "No extension found"       
        
    def process_document(self, message: dict):
        source = message["source"]
        file_path = message["data_id"]
        base_metadata = message["metadata"]
        print(f"Processing document: {file_path}")
        indexing = Pipeline()
        file_extension = self.get_file_extension(file_path)
        if file_extension == "pdf":
            indexing.add_component("converter", PyPDFToDocument())
        elif file_extension == "docx":
            indexing.add_component("converter", DocxToDocument())
        elif file_extension == "txt":
            indexing.add_component("converter", TextToDocument())
        elif file_extension == "pptx":
            indexing.add_component("converter", PptxToDocument())
        indexing.add_component("splitter", DocumentSplitter(split_by="period", split_length=2))
        indexing.add_component("embedder", self.document_embedder)
        indexing.add_component("writer", DocumentWriter(self.document_store))
        indexing.connect("converter", "splitter")
        indexing.connect("splitter", "embedder")
        indexing.connect("embedder", "writer")
        indexing.run({"converter": {"sources": [file_path]}})
        print(f"Document {file_path} processed and stored in Pinecone.")
      
class PromptProcessor:
    def __init__(self):
        env_utils = EnvUtils()
        os.environ['PINECONE_API_KEY'] = EnvUtils().get_required_env('PINECONE_API_KEY')
        self.document_store = PineconeDocumentStore(
        index=env_utils.get_required_env('PINECONE_INDEX'),
        metric="cosine",
        dimension=1024,
        spec={"serverless": {"region": "us-east-1", "cloud": "aws"}},
      )
        self.text_embedder = SentenceTransformersTextEmbedder(
            model="intfloat/multilingual-e5-large"
     )  
    def process_query(self, query):
        prompt_template = """Answer the following query based on the provided context. Don't hallucinate, you should answer ONLY from the context provided\n
                     Query: {{query}}
                     Documents:
                     {% for doc in documents %}
                        {{ doc.content }}
                     {% endfor %}
                     Answer: 
                  """
        
        query_pipeline = Pipeline()
        query_pipeline.add_component("text_embedder", self.text_embedder)
        query_pipeline.add_component("retriever", PineconeEmbeddingRetriever(document_store=self.document_store))
        query_pipeline.add_component("prompt_builder", PromptBuilder(template=prompt_template))
        query_pipeline.add_component("llm", OpenAIGenerator(api_key=Secret.from_token("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")))
        query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        query_pipeline.connect("retriever.documents", "prompt_builder.documents")
        query_pipeline.connect("prompt_builder", "llm")
        results = query_pipeline.run(
            {
                "text_embedder": {"text": query},
                "prompt_builder": {"query": query},
                "retriever": {"top_k": 1}
                
            },
            include_outputs_from={"prompt_builder"}
       )
        print(results)
    

if __name__ == "__main__":
    #assistant = PromptProcessor()
    #user_prompt = input("Enter your query: ")
    #assistant.process_query(user_prompt)
    #document_processor = DocumentProcessor()
    #document_processor.process_document("C:/datachime-data/Assignment-Physics-1.pdf")
    PromptProcessor().process_query("WHo is the author of Serverless Dataengineering paper ")
    #print(result)
