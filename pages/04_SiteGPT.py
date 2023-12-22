import streamlit as st
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer

st.set_page_config(
    page_title="SiteGPT",
    page_icon="😄"
)
html2text_transformer = Html2TextTransformer()

st.title("Site GPT")

with st.sidebar:
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
        )


if url:
    loader = AsyncChromiumLoader([url])
    docs = loader.load()
    transformed = html2text_transformer.transform_documents(docs)
    st.write(docs)
