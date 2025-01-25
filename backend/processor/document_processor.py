import json
import sys
import os
import re
import nltk
import logging
import fitz
from datetime import datetime
from pathlib import Path
from haystack.nodes import PreProcessor
from haystack.schema import Document
from pinecone import Pinecone
from unstructured.partition.auto import partition
from unstructured.documents.elements import Title, NarrativeText, ListItem
from sentence_transformers import SentenceTransformer

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from util.envutils import EnvUtils

class DocumentProcessor:
    def __init__(
        self,
        sentences_per_chunk: int = 3,
        sentence_overlap: int = 1
    ):
        nltk.download('punkt_tab')
        nltk.download('punkt')
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        self.logger = logging.getLogger(__name__)
        self.model = SentenceTransformer('intfloat/multilingual-e5-large')
        env_utils = EnvUtils()
        pinecone_api_key = env_utils.get_required_env('PINECONE_API_KEY')
        pinecone_host = env_utils.get_required_env('PINECONE_HOST')
        
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index("ragindex")
        
        self.preprocessor = PreProcessor(
            clean_empty_lines=True,
            clean_whitespace=True,
            clean_header_footer=True,
            split_by="word",
            split_length=100,
            split_overlap=20,
            split_respect_sentence_boundary=True
        )

    def extract_section_metadata(self, content: str) -> dict:
        patterns = {
            'chapter': r'Chapter\s+(\d+|[IVX]+)[:\s]+(.*?)(?=\n|$)',
            'section': r'(?:Section|Unit)\s+(\d+(?:\.\d+)?)[:\s]+(.*?)(?=\n|$)',
            'subsection': r'(?:\d+\.)+\d+\s+(.*?)(?=\n|$)'
        }
        
        metadata = {}
        for key, pattern in patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                metadata[f'{key}_number'] = matches[0][0] if len(matches[0]) > 1 else matches[0]
                metadata[f'{key}_title'] = matches[0][1] if len(matches[0]) > 1 else ''
        
        return metadata

    def store_documents(self, documents: list) -> None:
        try:
            for doc in documents:
                # Clean metadata
                metadata = {
                    "text": doc.content,
                    "source": doc.meta.get("source", ""),
                    "filename": doc.meta.get("filename", ""),
                    "timestamp": doc.meta.get("timestamp", "")
                }
                
                # Generate embeddings
                embeddings = self.model.encode(doc.content, normalize_embeddings=True)
                
                self.index.upsert(
                    vectors=[(
                        doc.id,
                        embeddings.tolist(),
                        metadata
                    )]
                )
            
            self.logger.info(f"Successfully stored {len(documents)} chunks in Pinecone")
            
        except Exception as e:
            self.logger.error(f"Error storing documents in Pinecone: {str(e)}")
            

    def process_document(self, message: dict) -> None:
        if not all(key in message for key in ["source", "data_id", "metadata"]):
            raise ValueError("Missing required fields in message")
            
        source = message["source"]
        data_id = message["data_id"]
        base_metadata = message["metadata"]
        
        if source == "local_filesystem":
            file_path = Path(data_id)
            current_chapter = {}
            current_section = {}
            chunks = []
            
            if file_path.suffix.lower() == '.pdf':
                with fitz.open(str(file_path)) as doc:
                    text = ""
                    for page in doc:
                        text += page.get_text()
                    
                    doc = Document(
                        content=text,
                        meta={
                            **base_metadata,
                            "source": source,
                            "data_id": str(data_id),
                            "filetype": "pdf"
                        }
                    )
                    chunks.extend(self.preprocessor.process([doc]))
            else:
                elements = partition(str(data_id))
                for element in elements:
                    if isinstance(element, Title):
                        section_metadata = self.extract_section_metadata(str(element))
                        if 'chapter_number' in section_metadata:
                            current_chapter = section_metadata
                        elif 'section_number' in section_metadata:
                            current_section = section_metadata
                    
                    element_metadata = {
                        **base_metadata,
                        **current_chapter,
                        **current_section,
                        'source': source,
                        'data_id': str(data_id),
                        'element_type': element.__class__.__name__,
                    }
                    
                    if isinstance(element, (NarrativeText, ListItem)):
                        doc = Document(
                            content=str(element),
                            meta=element_metadata
                        )
                        chunks.extend(self.preprocessor.process([doc]))

            self.store_documents(chunks)
            self.logger.info(f"Processed and stored document: {data_id}")
            print(f"Processed and stored document: {data_id}")
        else:
            raise ValueError(f"Unsupported source: {source}")

def main():
    try:
        processor = DocumentProcessor()
        
        test_file = "C:/datachime-data/Assignment-Physics-1.pdf"
        
        message = {
            "source": "local_filesystem",
            "data_id": test_file,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "filename": Path(test_file).name
            }
        }
        
        processor.process_document(message)
        print(f"Successfully processed file: {test_file}")
        
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        raise

if __name__ == "__main__":
    main()