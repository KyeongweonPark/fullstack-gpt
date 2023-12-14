import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.callbacks.base import BaseCallbackHandler


st.set_page_config(
    page_title="DocumentGPT",
    page_icon="😄"
)

class ChatCallbackHandler(BaseCallbackHandler):

    message = ""

    def on_llm_start(self, *arg, **kwargs):
        self.message_box = st.empty()
            
    def on_llm_end(self, *arg, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token: str, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOpenAI(
    temperature=0.1, 
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ]
    )

if "messages" not in st.session_state:
    st.session_state["messages"] = []

@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=600,
        # separator="CALENDRIER"
    )

    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    for doc in docs:
        doc.page_content = doc.page_content.replace('\n', '')

    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever() 
    return retriever

def save_message(message, role):
    st.session_state["messages"].append({"message":message, "role":role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","""
        Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
        
        Context: {context}
        """),
        ("human","{question}")
    ]
)

st.title("Document GPT")

st.markdown("""
            Welcome! 
            \nUse this chatbot to ask questions to an AI about your files!
            \nUpload your files on the sidebar.
            """)

with st.sidebar:
    file = st.file_uploader("Upload a .txt .pdf or .dox file", type=["pdf", "txt", "xlsx", "docx"])

if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")

    if message:
        send_message(message, "human")
        # result = retriever.invoke(message)
        # st.write(result)
        chain = {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        } | prompt | llm
        with st.chat_message("ai"):
            response = chain.invoke(message)
else:
    st.session_state["messages"] = []