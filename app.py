import os
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient, ServerlessSpec

# --- Pinecone setup ---
pc = PineconeClient(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "ragbot-index"

# Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,          # matches OpenAI embeddings
        metric="cosine",         # semantic similarity
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# --- LLM setup ---
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# --- Sidebar for document upload ---
st.sidebar.header("Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Choose files", type=["txt", "pdf", "docx"], accept_multiple_files=True
)

all_docs = []
embeddings = OpenAIEmbeddings()

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Save uploaded file temporarily
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Pick loader based on file type
        if uploaded_file.name.endswith(".txt"):
            loader = TextLoader(uploaded_file.name)
        elif uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(uploaded_file.name)
        elif uploaded_file.name.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(uploaded_file.name)
        else:
            st.warning(f"Unsupported file type: {uploaded_file.name}")
            continue

        docs = loader.load()
        all_docs.extend(docs)

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(all_docs)

    # Create embeddings and store in Pinecone
    vectorstore = Pinecone.from_documents(split_docs, embeddings, index_name=index_name)
    st.sidebar.success("Documents uploaded and indexed!")

else:
    # Connect to existing Pinecone index
    index = pc.Index(index_name)
    vectorstore = Pinecone(index, embeddings.embed_query, "text")

# --- Retrieval function ---
def retrieve_context(query, k=3):
    results = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in results])

# --- Main chat interface ---
st.title("📚 Multi‑Document RAG Chatbot")
user_input = st.text_input("Ask a question:")

if user_input:
    context = retrieve_context(user_input)
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided context when relevant."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_input}"}
    ]
    response = llm.invoke(messages)
    st.markdown("### AI Response")
    st.write(response.content)
