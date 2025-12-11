import os
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

docs = []
docs.extend(TextLoader("doc1.txt").load())
docs.extend(PyPDFLoader("doc2.pdf").load())
docs.extend(UnstructuredWordDocumentLoader("doc3.docx").load())


# IMPORTANT: Replace with your actual OpenAI API key
api_key = os.environ.get("OPEN_API_KEY")

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# Persistent directory for ChromaDB
persist_directory = "./chroma_db"

# Sidebar for document upload
st.sidebar.header("Upload Documents")
uploaded_files = st.sidebar.file_uploader("Choose files", type=["txt", "pdf", "docx"], accept_multiple_files=True)

if uploaded_files:
    all_docs = []
    for file in uploaded_files:
        # Save uploaded file temporarily
        with open(file.name, "wb") as f:
            f.write(file.getbuffer())
        loader = UnstructuredFileLoader(file.name)
        docs = loader.load()
        all_docs.extend(docs)

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(all_docs)

    # Create embeddings and store in persistent ChromaDB
    embeddings = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))
    vectorstore = Chroma.from_documents(split_docs, embeddings, persist_directory=persist_directory)
    vectorstore.persist()
    st.sidebar.success("Documents uploaded and indexed!")

else:
    # Load existing persistent DB if no new files uploaded
    embeddings = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))
    vectorstore = Chroma(embedding_function=embeddings, persist_directory=persist_directory)

# Retrieval function
def retrieve_context(query, k=3):
    results = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in results])

# Main chat interface
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



