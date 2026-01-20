import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda



load_dotenv()
persistent_directory="db/chroma_db"
COLLECTION_NAME = "my_collection"
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0)


def load_Documents(File_path):
    path=File_path
    if not path:
        print("Please upload your documents")
        return []
    
    documents=[]
    
    for path in File_path:
        ext = os.path.splitext(path)[1].lower()

        if ext == ".pdf":
            loader = PyPDFLoader(path)
        elif ext == ".txt":
            loader = TextLoader(path, encoding="utf-8")
        else:
            continue 


        documents.extend(loader.load())

    if len(documents) == 0:
        print(f"No files found .Please add your documents")
    
    return documents


def spliting(documents):
    splitter= RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=0
    )
    chunks=splitter.split_documents(documents)

    return chunks

    


def embeddingmodel(File_path):
    documments=load_Documents(File_path)
    chunk= spliting(documments)
    embeddings = GoogleGenerativeAIEmbeddings(model= 'gemini-embedding-001', dimension = 32)

    Chroma.from_documents(
    documents=chunk,
    embedding=embeddings,
    persist_directory=persistent_directory,
    collection_name=COLLECTION_NAME,
    collection_metadata={"hnsw:space":"cosine"}
    )

    

def retriver():


   
    embeddings= GoogleGenerativeAIEmbeddings(model= 'gemini-embedding-001', dimension = 32)

    vectorstore = Chroma(
        persist_directory=persistent_directory,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
    )

    model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0)
       
    
    retriever=vectorstore.as_retriever(search_kwargs={"k":8,"fetch_k": 20},
     search_type="mmr")
    
    
    prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer using the given context only."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
    ])

    parser = StrOutputParser()

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)
   
    

    chain = (
    {
        
        "context": RunnableLambda(lambda x: x["question"]) | retriever| format_docs,
        "question": RunnableLambda(lambda x: x["question"]),
        "chat_history": RunnableLambda(lambda x: x["chat_history"])
    }
    | prompt
    | model
    | parser
)
    
    
    return chain




    




