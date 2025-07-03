from langchain_core.messages import HumanMessage, AIMessage
import os
from dotenv import load_dotenv
import streamlit as st
from pathlib import Path
from agent import agent, check_ollama_connection, get_available_ollama_models

load_dotenv()   

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")


# configure Streamlit
st.set_page_config(
    page_title="Leonardo RAG",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("Leonardo RAG")


# sidebar for LLM selection
with st.sidebar:
    st.logo("./aieng_log.jpeg", size="large")
    st.header("⚙️ Settings")

    # check OLLAMA connection
    ollama_available = check_ollama_connection()

    if ollama_available:
        st.success("✅ Ollama is running")
        
    else:

        st.error("❌ OLLAMA not running")



# PDF upload

st.markdown("⬆️ Upload the document for RAG processing.")

uploaded_file = st.file_uploader(
    "📄 Upload your PDF Document",
    type=["pdf"],
    help="Upload a PDF file to run semantic search on"
)

# Path del PDF salvato (di default None)

pdf_path = None
path_name = None

if uploaded_file is not None:
    # Definisci dove salvarlo
    save_dir = Path("./uploaded_pdfs")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Salva il PDF caricato
    pdf_path = save_dir / uploaded_file.name
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    path_name = Path(pdf_path).name

    pdf_path_str = pdf_path.as_posix()

    st.success(f"✅ Uploaded file saved as: `{pdf_path_str}`")


# Main content
initial_msg = f"""
#### Welcome!!! I am your RAG assistant chatbot 👨‍🦰
#### You can ask me any queries about Leonardo's Documents
> **NOTE:** Currently I have access to the **{"./uploaded_pdfs"}** directory where all uploaded PDFs are stored. Try to ask relevant queries only😇
"""
st.markdown(initial_msg)


# Initialize session state
if "store" not in st.session_state:
    st.session_state.store = []


store = st.session_state.store

# Display chat history
for message in store:
    if message.type == "ai":
        avatar = "👨‍🦰"
    else:
        avatar = "💬"

    with st.chat_message(message.type, avatar=avatar):
        st.markdown(message.content)


# Chat input
if prompt := st.chat_input("What is your query?"):
    # Display user message
    st.chat_message("user", avatar="💬").markdown(prompt)

    # Show thinking message
    thinking_placeholder = st.chat_message("assistant", avatar="👨‍🦰")
    thinking_placeholder.markdown("Thinking...")

    # Add user message to store
    store.append(HumanMessage(content=prompt))


    try:
        

        if ollama_available:
            response_content = agent(query=prompt, ollama_model="llama3.1:8b")

        
            response = AIMessage(content=response_content)

    except Exception as e:
        error_msg = f"Sorry, I encountered an error: {str(e)}"

        if "API" in str(e).upper():
            error_msg += "\n\nThis might be due to API limits. Try using OLLAMA for local processing."

        response = AIMessage(content=error_msg)

    
    # Add response to store
    store.append(response)

    # Update the thinking message with actual response
    thinking_placeholder.markdown(response.content)



# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 12px;'>
        💡 Tip: Use OLLAMA for unlimited local processing
    </div>
    """,
    unsafe_allow_html=True
)