import requests
import os
import logging
from typing import Dict, List, Optional
from langchain.llms.base import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import LLMResult, Generation
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings  # Updated import
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory

logger = logging.getLogger(__name__)
try:
    logger.info(f"Loading helper.py from: {os.path.abspath(__file__)}")
except NameError:
    logger.info("Loading helper.py from an unknown location (no __file__ available)")

# GrokLLM and GrokInterface Classes
class GrokLLM(LLM):
    def __init__(self, api_key: str):
        super().__init__()
        self._api_key = api_key
        self.api_url = "https://api.x.ai/v1/chat/completions"
        logger.info(f"GrokLLM initialized with API key: {'*' * len(api_key) if api_key else 'None'}")
        logger.debug(f"Confirming _api_key attribute: {hasattr(self, '_api_key')}, value: {'*' * len(self._api_key) if self._api_key else 'None'}")

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        logger.info("Calling Grok API")
        logger.debug(f"Using _api_key: {'*' * len(self._api_key) if self._api_key else 'None'}")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "grok-3",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "temperature": 0.7
        }
        if stop:
            data["stop"] = stop
        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Error calling Grok API: {e}")
            return "Error processing query"

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> LLMResult:
        generations = []
        for prompt in prompts:
            prompt_str = str(prompt)
            text = self._call(prompt_str, stop, run_manager, **kwargs)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        return "grok"

class GrokInterface:
    def __init__(self, api_key: str):
        logger.info("Initializing GrokInterface")
        self.llm = GrokLLM(api_key=api_key)
        logger.debug(f"GrokInterface llm _api_key: {getattr(self.llm, '_api_key', 'Not set')}")
        self.retrieval_chain = None

    def setup_retrieval_chain(self, retriever):
        """Set up the retrieval-augmented generation chain."""
        try:
            prompt_template = """Use the following context to answer the question:
            {context}
            
            Question: {question}
            
            Answer:"""
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            self.retrieval_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt}
            )
            logger.info("Retrieval chain set up successfully")
        except Exception as e:
            logger.error(f"Error setting up retrieval chain: {e}")
            raise

    def query(self, question: str) -> Dict:
        """Query the system with a question and return the answer."""
        if not self.retrieval_chain:
            logger.error("Retrieval chain not initialized")
            raise ValueError("Retrieval chain not initialized")
        try:
            result = self.retrieval_chain.invoke({"query": question})
            return {
                "answer": result["result"],
                "source_documents": result["source_documents"]
            }
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {"answer": "Error processing query", "source_documents": []}

# DocumentProcessor Class
class DocumentProcessor:
    def __init__(self, pdf_directory: Optional[str] = None, pdf_paths: Optional[List[str]] = None):
        self.pdf_directory = pdf_directory
        self.pdf_paths = pdf_paths
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=20,
            length_function=len
        )

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a single PDF file."""
        try:
            pdf_reader = PdfReader(pdf_path)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""

    def process_pdfs(self) -> List[str]:
        """Process PDFs from either a directory or a list of file paths."""
        all_texts = []
        pdf_files = []

        if self.pdf_paths:
            pdf_files = self.pdf_paths
        elif self.pdf_directory:
            import glob
            pdf_files = glob.glob(os.path.join(self.pdf_directory, "*.pdf"))
            if not pdf_files:
                logger.warning(f"No PDF files found in {self.pdf_directory}")
                return []

        if not pdf_files:
            logger.warning("No PDF files provided or found.")
            return []

        for pdf_file in pdf_files:
            logger.info(f"Processing {pdf_file}")
            text = self.extract_text_from_pdf(pdf_file)
            if text:
                chunks = self.text_splitter.split_text(text)
                all_texts.extend(chunks)
        return all_texts

# VectorStore Class
class VectorStore:
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vector_store = None

    def create_vector_store(self, texts: List[str]):
        """Create a FAISS vector store from text chunks."""
        try:
            self.vector_store = FAISS.from_texts(texts, self.embeddings)
            logger.info("Vector store created successfully.")
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise

    def get_retriever(self):
        """Return the retriever for the vector store."""
        if not self.vector_store:
            logger.error("Vector store not initialized.")
            raise ValueError("Vector store not initialized.")
        return self.vector_store.as_retriever(search_kwargs={"k": 3})