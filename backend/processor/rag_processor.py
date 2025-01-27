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
from openai import OpenAI


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
        self.env_utils = EnvUtils()
        os.environ['PINECONE_API_KEY'] = self.env_utils.get_required_env('PINECONE_API_KEY')
        self.openai_api_key = Secret.from_token(self.env_utils.get_required_env('OPENAI_API_KEY'))
        self.perplexityapi_key = Secret.from_token(self.env_utils.get_required_env('PERPLEXITY_API_KEY'))
        self.model = self.env_utils.get_required_env('PERPLEXITY_MODEL_NAME')
        self.client = OpenAI(
            api_key=self.perplexityapi_key,
            base_url="https://api.perplexity.ai"
        )
        self.document_store = PineconeDocumentStore(
        index=self.env_utils.get_required_env('PINECONE_INDEX'),
        metric="cosine",
        dimension=1024,
        spec={"serverless": {"region": "us-east-1", "cloud": "aws"}},
      )
        self.text_embedder = SentenceTransformersTextEmbedder(
            model="intfloat/multilingual-e5-large"
     )
    def _create_evaluation_and_search_prompt(self, query: str, context: list[str]) -> str:
        print(f"Creating evaluation prompt for query: {query}")
        prompt = """Review the following query and context. If the context is sufficient to provide a complete and accurate answer, respond with just "YES". 
                 If additional information is needed, respond with a search query that would help find the missing information. 
                 The search query should be concise and focused on the missing aspects.
                 
                 User Query: {query}
                 
                 Available Context:
                 {context}
                 
                 Response (either "YES" or a search query):""".format(
            query=query,
            context="\n".join(context) if context else "No context available"
        )
        print(f"Generated prompt: {prompt[:200]}...")  # Log first 200 chars of prompt
        return prompt  
    def _create_final_prompt(self, query: str, pinecone_context: list[str], web_context: str = None) -> str:
        print("Creating final answer prompt")
        base_template = """Answer the following query based on the provided context. Ensure your answer is comprehensive and accurate.
                          
                          Query: {query}
                          
                          Primary Context from Database:
                          {pinecone_context}
                          """
        
        if web_context:
            self.logger.info("Including web context in final prompt")
            base_template += """
                          Additional Context from Web Search:
                          {web_context}
                          """
            
        base_template += "\nAnswer:"
        
        prompt = base_template.format(
            query=query,
            pinecone_context="\n".join(pinecone_context) if pinecone_context else "No database context available",
            web_context=web_context
        )
        print(f"Generated final prompt: {prompt[:200]}...")
        return prompt
    def _evaluate_context_and_get_search_query(self, query: str, context: list[str]) -> tuple[bool, str]:
        """Evaluate context and get search query in one LLM call."""
        print("Evaluating context sufficiency and generating search query if needed")
        
        evaluation_pipeline = Pipeline()
        evaluation_pipeline.add_component(
            "evaluator",
            OpenAIGenerator(api_key=self.openai_api_key)
        )
        
        print("Sending evaluation request to LLM")
        results = evaluation_pipeline.run(
            {
                "evaluator": {
                    "prompt": self._create_evaluation_and_search_prompt(query, context)
                }
            }
        )
        
        response = results["evaluator"]["replies"][0].strip().upper()
        print(f"LLM evaluation response: {response}")
        
        is_sufficient = response == "YES"
        search_query = None if is_sufficient else response
        
        print(f"Context sufficient: {is_sufficient}")
        if not is_sufficient:
            print(f"Generated search query: {search_query}")
        
        return is_sufficient, search_query
    def process_query(self, query: str) -> dict[str, any]:
        print(f"Processing query: {query}")
        
        # Step 1: Initial retrieval from Pinecone
        print("Step 1: Retrieving documents from Pinecone")
        retrieval_pipeline = Pipeline()
        retrieval_pipeline.add_component("text_embedder", self.text_embedder)
        retrieval_pipeline.add_component("retriever", PineconeEmbeddingRetriever(document_store=self.document_store))
        retrieval_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        
        initial_results = retrieval_pipeline.run(
            {
                "text_embedder": {"text": query},
                "retriever": {"top_k": 3}
            }
        )
        
        retrieved_docs = [doc.content for doc in initial_results["retriever"]["documents"]]
        print(f"Retrieved {len(retrieved_docs)} documents from Pinecone")
        print(f"First document preview: {retrieved_docs[0][:200]}..." if retrieved_docs else "No documents retrieved")

        # Step 2: Evaluate context and get search query if needed
        print("Step 2: Evaluating context and generating search query if needed")
        is_sufficient, search_query = self._evaluate_context_and_get_search_query(query, retrieved_docs)
        
        # Step 3: Web search if needed
        web_context = None
        if not is_sufficient:
            print(f"Step 3: Performing web search with query: {search_query}")
            web_results = self.client.chat.completions.create(
                model=self.model,
                messages=search_query,
                temperature=0.1  # Lower temperature for more consistent JSON
            )
            #web_results = self.perplexity.search(search_query)
            web_context = "\n".join([result.text for result in web_results])
            print(f"Web search results preview: {web_context[:200]}..." if web_context else "No web results")
        else:
            print("Step 3: Web search not needed")

        # Step 4: Generate final answer
        print("Step 4: Generating final answer")
        final_pipeline = Pipeline()
        final_pipeline.add_component(
            "prompt_builder",
            PromptBuilder(template=self._create_final_prompt(query, retrieved_docs, web_context))
        )
        final_pipeline.add_component(
            "llm",
            OpenAIGenerator(api_key=self.openai_api_key)
        )
        final_pipeline.connect("prompt_builder", "llm")
        
        final_results = final_pipeline.run(
            {
                "prompt_builder": {"query": query},
            }
        )
        
        answer = final_results["llm"]["replies"][0]
        print("Final answer generated")
        print(f"Answer preview: {answer[:200]}...")

        # Prepare and return results
        results = {
            "query": query,
            "search_query_used": search_query,
            "answer": answer,
            "used_web_search": not is_sufficient,
            "pinecone_context": retrieved_docs,
            "web_context": web_context if web_context else None
        }
        print("Query processing completed")
        return results
    

if __name__ == "__main__":
    #assistant = PromptProcessor()
    #user_prompt = input("Enter your query: ")
    #assistant.process_query(user_prompt)
    #document_processor = DocumentProcessor()
    #document_processor.process_document("C:/datachime-data/Assignment-Physics-1.pdf")
    result = PromptProcessor().process_query("How does AI Retail Suite compare to Salesforce Einstein and SAP Commerce Cloud for dynamic pricing?")
    print(result)
