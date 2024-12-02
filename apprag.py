import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.document_loaders import PyPDFLoader
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain



st.title("RAG App with Fallback to Model")

model_choice = st.selectbox(
    "Choose a model:",
    options=["mistral", "llama3.1", "llama2"],  
)

if model_choice == "llama3.1":
    embed_model = OllamaEmbeddings(model="llama3.1", base_url="http://127.0.0.1:11434")
    llm = Ollama(model="llama3.1", base_url="http://127.0.0.1:11434")
elif model_choice == "mistral":
    embed_model = OllamaEmbeddings(model="mistral", base_url="http://127.0.0.1:11434")
    llm = Ollama(model="mistral", base_url="http://127.0.0.1:11434")
elif model_choice == "llama2":
    embed_model = OllamaEmbeddings(model="llama2", base_url="http://127.0.0.1:11434")
    llm = Ollama(model="llama2", base_url="http://127.0.0.1:11434")

uploaded_file = st.file_uploader("Upload a PDF file to create a new vector database", type=["pdf"])
retriever = None
retrieval_chain = None

if uploaded_file:
    st.write("Processing uploaded file...")
    temp_file_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    try:
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        
        new_persist_directory = "./RAG/new_chroma_db"
        vector_store = Chroma.from_documents(docs, embed_model, persist_directory=new_persist_directory)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        
        retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        combined_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
        retrieval_chain = create_retrieval_chain(retriever, combined_docs_chain)
        
        st.success("New vector database created successfully!")
    except Exception as e:
        st.error(f"Error processing the file: {e}")
    finally:
        os.remove(temp_file_path)

question = st.text_input("Ask your question:")
if st.button("Get Answer"):
    if question:
        with st.spinner("Searching for the answer..."):
            if retriever:
                docs = retriever.get_relevant_documents(question)
                if docs:  # If relevant documents are found
                    response = retrieval_chain.invoke({"input": question})
                    st.write("**Answer (from context):**")
                    st.write(response["answer"])
                else:  # Fall back to the model directly
                    st.write("**Answer (directly from model):**")
                    response = llm(question)  # Pass question as a string
                    st.write(response)
            else:
                st.warning("No vector database found. Defaulting to model.")
                response = llm(question)  # Pass question as a string
                st.write("**Answer (directly from model):**")
                st.write(response)
    else:
        st.write("Please enter a question.")
