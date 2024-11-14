Certainly! Here is a README for your code that you can use for your GitHub repository:

---

# Agentic RAG System

Agentic RAG System is a Streamlit application designed to assist users in querying information from uploaded PDF documents. The system leverages LangChain, Google Generative AI, and FAISS for document search and retrieval augmented generation (RAG).

## Features

- Upload multiple PDF documents and extract their text content.
- Create a vector store of document embeddings using Google Generative AI.
- Use a Retrieval-Augmented Generation (RAG) tool to answer user queries based on the uploaded documents.
- Maintain a chat history for better context during interactions.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/muditgaur-1009/Agentic-RAG.git
    cd Agentic-RAG
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Configure the Google API key:
    - Replace `"api-key-here"` in the script with your actual Google API key.

## Usage

1. Run the Streamlit application:
    ```sh
    streamlit run app.py
    ```

2. Open your web browser and navigate to the URL provided by Streamlit, typically `http://localhost:8501`.

3. Upload your PDF documents using the file uploader.

4. Ask questions about the content of your uploaded documents using the chat interface.

## Code Overview

- `app.py`: Main script containing the Streamlit application.
- Functions:
  - `create_agent(llm, tools)`: Creates an agent with the provided language model and tools.
  - `process_pdf(uploaded_file)`: Processes a PDF file and extracts its text content.
  - `create_vector_store(texts, embeddings)`: Creates a FAISS vector store from the given texts and embeddings.
  - `setup_rag_tool(vector_store)`: Creates a RAG tool for document search.
  - `main()`: Main function that initializes the Streamlit application.

## Dependencies

- Streamlit
- LangChain
- Google Generative AI
- FAISS
- PyPDF2

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For any questions or issues, please open an issue in this repository or contact at muditgaur1009@gmail.com.

---

Feel free to customize the README further to suit your preferences.
