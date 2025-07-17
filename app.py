import streamlit as st
from src.helper import DocumentProcessor, VectorStore, GrokInterface
from dotenv import load_dotenv
import os
import logging
import tempfile
import shutil
from langchain.memory import ConversationBufferMemory

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GROK_API_KEY = os.getenv("GROK_API_KEY")
logger.info(f"GROK_API_KEY loaded: {'True' if GROK_API_KEY else 'False'}")
logger.info(f".env file path: {os.path.abspath('.env') if os.path.exists('.env') else 'Not found'}")

def main():
    st.set_page_config(page_title="PDF Information Retrieval System", page_icon="📚")
    st.title("PDF Information Retrieval System")

    # Initialize session state
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "grok_interface" not in st.session_state:
        st.session_state.grok_interface = None
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "temp_dir" not in st.session_state:
        st.session_state.temp_dir = None
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        pdf_option = st.radio("Select PDF input method:", ("Upload PDFs", "Use PDF Directory"))
        
        if pdf_option == "Upload PDFs":
            uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
            if st.button("Process Uploaded PDFs"):
                if uploaded_files:
                    with st.spinner("Processing uploaded PDFs..."):
                        try:
                            # Create a temporary directory
                            temp_dir = tempfile.mkdtemp()
                            st.session_state.temp_dir = temp_dir
                            logger.info(f"Temporary directory created: {temp_dir}")
                            
                            # Save uploaded files to temporary directory
                            pdf_paths = []
                            for uploaded_file in uploaded_files:
                                file_path = os.path.join(temp_dir, uploaded_file.name)
                                with open(file_path, "wb") as f:
                                    f.write(uploaded_file.getbuffer())
                                pdf_paths.append(file_path)
                            logger.info(f"Uploaded PDFs: {pdf_paths}")
                            
                            # Process PDFs
                            processor = DocumentProcessor(pdf_paths=pdf_paths)
                            texts = processor.process_pdfs()
                            if not texts:
                                st.error("No texts extracted from PDFs.")
                                return

                            vector_store = VectorStore()
                            vector_store.create_vector_store(texts)
                            st.session_state.vector_store = vector_store
                            logger.info("Vector store created successfully")

                            # Initialize GrokInterface
                            logger.info(f"Initializing GrokInterface with API key: {'*' * len(GROK_API_KEY) if GROK_API_KEY else 'None'}")
                            grok_interface = GrokInterface(GROK_API_KEY)
                            logger.debug(f"GrokLLM _api_key: {getattr(grok_interface.llm, '_api_key', 'Not set')}")
                            grok_interface.setup_retrieval_chain(vector_store.get_retriever())
                            st.session_state.grok_interface = grok_interface
                            st.session_state.processed = True
                            st.success("PDFs processed successfully!")
                        except Exception as e:
                            st.error(f"Error processing PDFs: {e}")
                            logger.error(f"Error in processing: {e}", exc_info=True)
                            # Clean up temporary directory
                            if st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
                                shutil.rmtree(st.session_state.temp_dir)
                                st.session_state.temp_dir = None
                else:
                    st.warning("Please upload at least one PDF file.")

        else:
            pdf_directory = st.text_input("PDF Directory", "./pdfs")
            if st.button("Process PDFs from Directory"):
                with st.spinner("Processing PDFs..."):
                    try:
                        processor = DocumentProcessor(pdf_directory=pdf_directory)
                        texts = processor.process_pdfs()
                        if not texts:
                            st.error("No texts extracted from PDFs.")
                            return

                        vector_store = VectorStore()
                        vector_store.create_vector_store(texts)
                        st.session_state.vector_store = vector_store
                        logger.info("Vector store created successfully")

                        # Initialize GrokInterface
                        logger.info(f"Initializing GrokInterface with API key: {'*' * len(GROK_API_KEY) if GROK_API_KEY else 'None'}")
                        grok_interface = GrokInterface(GROK_API_KEY)
                        logger.debug(f"GrokLLM _api_key: {getattr(grok_interface.llm, '_api_key', 'Not set')}")
                        grok_interface.setup_retrieval_chain(vector_store.get_retriever())
                        st.session_state.grok_interface = grok_interface
                        st.session_state.processed = True
                        st.success("PDFs processed successfully!")
                    except Exception as e:
                        st.error(f"Error processing PDFs: {e}")
                        logger.error(f"Error in processing: {e}", exc_info=True)

    # Main interface
    if st.session_state.processed:
        st.header("Ask a Question")
        question = st.text_input("Enter your question:")
        if st.button("Get Answer"):
            if question:
                with st.spinner("Generating answer..."):
                    try:
                        result = st.session_state.grok_interface.query(question)
                        st.write("**Answer:**")
                        st.write(result["answer"])
                        st.write("**Source Documents:**")
                        for i, doc in enumerate(result["source_documents"], 1):
                            st.write(f"**Doc {i}:** {doc.page_content[:200]}...")
                        # Update memory
                        st.session_state.memory.save_context({"query": question}, {"answer": result["answer"]})
                    except Exception as e:
                        st.error(f"Error generating answer: {e}")
                        logger.error(f"Error in query: {e}", exc_info=True)
            else:
                st.warning("Please enter a question.")
        
        # Display chat history
        st.header("Chat History")
        for msg in st.session_state.memory.load_memory_variables({})["chat_history"]:
            st.write(f"{msg.type}: {msg.content}")

        # Clean up temporary directory
        if st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
            shutil.rmtree(st.session_state.temp_dir)
            st.session_state.temp_dir = None

    else:
        st.info("Please process PDFs using the sidebar configuration.")

if __name__ == "__main__":
    if not GROK_API_KEY:
        st.error("GROK_API_KEY not found in environment variables. Please set it in a .env file.")
        logger.error("GROK_API_KEY not found in environment variables.")
    else:
        main()