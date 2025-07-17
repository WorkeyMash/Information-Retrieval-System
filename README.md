**Project Overview**
 - This project implements a modular PDF Information Retrieval System with the following components:

1. **Document Processor (`document_processor.py`)**: Extracts text from PDFs and splits it into chunks using LangChain's text splitter.
2. **Vector Store (`vector_store.py`)**: Creates embeddings using HuggingFace's `distilbert-base-uncased` and stores them in a FAISS vector database.
3. **Grok Interface (`grok_interface.py`)**: Integrates with the xAI Grok API for retrieval-augmented generation (RAG).
4. **Streamlit App (`app.py`)**: Provides a user-friendly interface for uploading PDFs and querying the system.
5. **Requirements (`requirements.txt`)**: Lists all necessary Python dependencies.

### Setup Instructions
1. **Environment Setup**:
   - Create a virtual environment:
     ```bash
     python -m venv venv
     source venv/bin/activate  # On Windows: venv\Scripts\activate
     ```
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```
   - Obtain a Grok API key from [xAI Console](https://console.x.ai) and create a `.env` file in the project root:
     ```plaintext
     GROK_API_KEY=your-api-key-here
     ```

2. **Directory Structure**:
   ```
   pdf_retrieval_system/
   ├── pdfs/                 # Directory for PDF files
   ├── .env                  # Environment variables (GROK_API_KEY)
   ├── requirements.txt      # Dependencies
   ├── document_processor.py # PDF processing module
   ├── vector_store.py       # Vector store module
   ├── grok_interface.py     # Grok API integration
   └── app.py                # Streamlit app
   ```

3. **Run the Application**:
   - Create a `pdfs` directory and place your PDF files there.
   - Run the Streamlit app:
     ```bash
     streamlit run app.py
     ```
   - Open the provided URL (e.g., `http://localhost:8501`) in your browser.

4. **Usage**:
   - In the sidebar, enter the path to your PDF directory (default: `./pdfs`) and click "Process PDFs".
   - Once processed, enter a question in the main interface and click "Get Answer" to retrieve answers and source document snippets.

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
