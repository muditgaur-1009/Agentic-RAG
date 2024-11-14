import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, Tool
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain.document_loaders import PyPDFLoader
import google.generativeai as genai
import os
import tempfile
import PyPDF2
import logging

# Setup logging with more detail
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure the Google API key
os.environ["GOOGLE_API_KEY"] = "api-key-here"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.7)

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def create_agent(llm, tools):
    """Create an agent with the given LLM and tools."""
    try:
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=(
                "You are a helpful AI assistant that uses tools to find and provide relevant information. "
                "Always search through the available documents before providing an answer. "
                "When providing answers:\n"
                "1. Always cite specific sections from the searched documents\n"
                "2. If the search doesn't yield relevant information, acknowledge that\n"
                "3. Maintain a professional and informative tone\n"
                "4. Organize complex information in a structured way"
            )),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            HumanMessage(content="{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        llm_with_tools = llm.bind(
            functions=[format_tool_to_openai_function(t) for t in tools]
        )

        agent = (
            {
                "input": lambda x: x["input"],
                "chat_history": lambda x: x.get("chat_history", []),
                "agent_scratchpad": lambda x: format_to_openai_function_messages(
                    x["intermediate_steps"]
                ),
            }
            | prompt
            | llm_with_tools
            | OpenAIFunctionsAgentOutputParser()
        )

        executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        return executor
        
    except Exception as e:
        logger.error(f"Error creating agent: {str(e)}")
        raise

def process_pdf(uploaded_file):
    """Process a PDF file and extract its text content."""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Read PDF content with PyPDF2
        with open(tmp_file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)

            # Extract text using PyPDF2
            text_content = ""
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text_content += page.extract_text() + "\n\n"

        # Clean up
        os.unlink(tmp_file_path)

        # Verify we got some text
        if not text_content.strip():
            raise Exception("No text content extracted from PDF")

        return text_content

    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise

def create_vector_store(texts, embeddings):
    """Create a FAISS vector store."""
    try:
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        documents = text_splitter.split_text(texts)

        # Create vector store
        vector_store = FAISS.from_texts(documents, embeddings)
        
        # Verify vector store
        test_search = vector_store.similarity_search("test", k=1)
        
        return vector_store
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise

def setup_rag_tool(vector_store):
    """Create a RAG tool."""
    def rag_func(query: str) -> str:
        try:
            docs = vector_store.similarity_search(query, k=3)
            results = []
            for i, doc in enumerate(docs, 1):
                results.append(f"Result {i}:\n{doc.page_content}\n")
            return "\n".join(results)
        except Exception as e:
            logger.error(f"Error in RAG search: {str(e)}")
            return f"Error performing search: {str(e)}"
    
    return Tool(
        name="search_documents",
        description="Search through the documents for relevant information. Use this tool first before providing any answer.",
        func=rag_func
    )

def main():
    st.title("📚 Agentic RAG System")
    
    # Session state initialization
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload your PDF documents", 
        type=["pdf"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
        if new_files:
            try:
                all_text = []
                for file in new_files:
                    text = process_pdf(file)
                    if text.strip():  # Only add if we got some text
                        all_text.append(text)
                        st.session_state.processed_files.add(file.name)
                    else:
                        st.warning(f"No text extracted from {file.name}")
                
                if all_text:
                    combined_text = "\n\n".join(all_text)
                    st.session_state.vector_store = create_vector_store(combined_text, embeddings)
                    rag_tool = setup_rag_tool(st.session_state.vector_store)
                    st.session_state.agent = create_agent(llm, [rag_tool])
                    st.success("Processing complete! You can now ask questions.")
                else:
                    st.error("No text was extracted from any of the uploaded files.")
            except Exception as e:
                st.error(f"Error during processing: {str(e)}")

    # Display processed files
    if st.session_state.processed_files:
        with st.expander("Processed Documents"):
            for file_name in st.session_state.processed_files:
                st.write(f"✓ {file_name}")

    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your documents"):
        if not st.session_state.vector_store:
            st.error("Please upload some PDF documents first!")
            return

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                response = st.session_state.agent.invoke({
                    "input": prompt,
                    "chat_history": [(m["role"], m["content"]) for m in st.session_state.messages]
                })
                response_content = response["output"]
                st.markdown(response_content)
                st.session_state.messages.append({"role": "assistant", "content": response_content})
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                logger.error(f"Error generating response: {str(e)}")

if __name__ == "__main__":
    main()