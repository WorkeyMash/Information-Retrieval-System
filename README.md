**Project Overview**

PDF Information Retrieval System
This project is a Streamlit-based application designed to process PDF documents and enable information retrieval using a retrieval-augmented generation (RAG) system powered by the xAI Grok API.


### Setup Instructions
1. **Environment Setup**:
   Clone the repository:
git clone https://github.com/your-repo/information-retrieval-system.git
cd information-retrieval-system


Create a virtual environment and activate it:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install the required dependencies:
pip install -r requirements.txt


Set up your xAI Grok API key in a .env file:
GROK_API_KEY=your_api_key_here


Run the Streamlit app:
streamlit run app.py


2. **Directory Structure**:
 The core functionality is packaged into a single src/helper.py file, which includes the following components:

Document Processor (document_processor.py): Extracts text from PDFs and splits it into manageable chunks using LangChain's RecursiveCharacterTextSplitter. This allows for efficient text processing and preparation for embedding.

Vector Store (vector_store.py): Creates embeddings using HuggingFace's distilbert-base-uncased model and stores them in a FAISS vector database. This enables fast similarity searches for retrieving relevant document chunks.

Grok Interface (grok_interface.py): Integrates with the xAI Grok API to perform retrieval-augmented generation (RAG), combining retrieved document context with the Grok model to generate accurate and context-aware responses.

3. **Run the Application**:
   - Create a `pdfs` directory and place your PDF files there.
   - Run the Streamlit app:
     ```bash
     streamlit run app.py
     ```
   - Open the provided URL (e.g., `http://localhost:8501`) in your browser.

4. **Usage**:
   
- Upload PDF files or specify a directory containing PDFs via the sidebar.
- Process the PDFs to create a vector store and initialize the Grok interface.
- Ask questions in the main interface to retrieve answers based on the processed documents.

 ### Customization
- **PDF Directory**: Modify the default directory in the Streamlit interface or `app.py`.
- **Embedding Model**: Change the `model_name` in `vector_store.py` to another HuggingFace model (e.g., `sentence-transformers/all-MiniLM-L6-v2`).
- **Grok Parameters**: Adjust `max_tokens` or `temperature` in `grok_interface.py` for different response lengths or creativity levels.
- **Chunk Size**: Modify `chunk_size` and `chunk_overlap` in `document_processor.py` for different text splitting behavior.

### Limitations and Improvements
- **Grok API**: Requires a valid API key and credits; usage is subject to rate limits (1 request/second, 60 or 1200 requests/hour depending on the model).[](https://www.merge.dev/blog/grok-api-key)
- **PDF Processing**: PyPDF2 may struggle with complex PDFs (e.g., scanned documents). Consider adding OCR support (e.g., using `pytesseract`) for scanned PDFs.
- **Vector Store**: FAISS is CPU-based; for large-scale applications, consider FAISS-GPU or cloud-based vector databases like Pinecone.
- **Scalability**: Add caching for embeddings to avoid reprocessing PDFs on each run.
- **Interface**: Enhance the Streamlit app with features like PDF upload via the UI or chat history.

This implementation provides a modular, deployable solution for PDF-based information retrieval using Grok and Streamlit, suitable for both local development and potential production deployment.

**Techstack Used:**
- Python
- LangChain
- Streamlit
- GROK
- FAISS

Contributing
Feel free to submit issues or pull requests to enhance the project!
License
MIT License (or specify your preferred license).
 - This project implements a modular PDF Information Retrieval System with the following components:
