import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from project import load_Documents,embeddingmodel, retriver, normalize_input
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq


st.set_page_config(page_title="Chatbot")
st.title("AI-Powered Document Search & Knowledge Retrieval System")
st.subheader("Upload documents")
def build_vectorstore(file_paths):
    return embeddingmodel(file_paths)
model = RunnableLambda(normalize_input) | RunnableLambda(call_llm)
#model = RunnableLambda(normalize_input) |ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

if "file_paths" not in st.session_state:
    st.session_state.file_paths = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf","txt"],
    accept_multiple_files=True
)



if uploaded_files:
    st.session_state.file_paths = [] 
    for file in uploaded_files:
        path = os.path.join(UPLOAD_DIR, file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        st.session_state.file_paths.append(path)

if st.button("Load documents"):
    docs=load_Documents( st.session_state.file_paths)
    if not docs:
        st.warning("Please upload your documents ")
        st.stop()
    
    db = build_vectorstore(st.session_state.file_paths)
    st.session_state.chat_history = []
    st.session_state.chain = retriver(db)
    st.success("Document loaded. Ask your question below...")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask something from your documents...")

if user_input:
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    if st.session_state.chain is None:
        st.warning("Please load documents before asking questions.")
        st.stop() 
    usr_input=str(user_input)
    response = st.session_state.chain.invoke({
        "question": usr_input,
        "chat_history": st.session_state.chat_history
    })
    st.session_state.chat_history.append(AIMessage(content=response))

for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    else:
        st.chat_message("assistant").write(msg.content)
