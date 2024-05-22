import gradio as gr
import os
from llama_index.packs.raptor import RaptorRetriever
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore # type: ignore
import chromadb
from llama_index.packs.raptor import RaptorPack
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
import streamlit as st


os.environ["OPENAI_API_KEY"] = "sk-proj-cHgnNwpihYLM884UqVkDT3BlbkFJGw0AgW8etJSuz6s13Le1"

import asyncio


from llama_index.core import SimpleDirectoryReader

@st.cache_resource
def raptor_retriever():
    documents = SimpleDirectoryReader(input_files=["./Form Master Services Agreement (Outsourcing).DOCX"]).load_data()
    client = chromadb.PersistentClient(path="./raptor_paper_db")
    collection = client.get_or_create_collection("raptor")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    retriever = RaptorRetriever(
        [],
    embed_model=OpenAIEmbedding(
        model="text-embedding-3-small"
    ),  # used for embedding clusters
    llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1),  # used for generating summaries
    vector_store=vector_store,  # used for storage
    similarity_top_k=2,  # top k for each layer, or overall top-k for collapsed
    mode="tree_traversal")
    raptor_query_engine = RetrieverQueryEngine.from_args(
    retriever, llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1,  use_async=True))
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    return raptor_query_engine, query_engine

raptor_query_engine, query_engine = raptor_retriever()

input = st.text_area('Prompt', 'Enter input prompt')

def greet():
    print(input)
    response1 = raptor_query_engine.query(input)
    response2 = query_engine.query(input).response
    st.subheader('Raptor Output: ')
    st.write(str(response1))
    st.subheader('vanilla RAG Output: ')
    st.write(str(response2))

st.button('Run', on_click=greet)
