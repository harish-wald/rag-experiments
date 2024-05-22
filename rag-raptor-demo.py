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
import threading


import asyncio


from llama_index.core import SimpleDirectoryReader
documents = SimpleDirectoryReader(input_files=["./Form Master Services Agreement (Outsourcing).DOCX"]).load_data()



client = chromadb.PersistentClient(path="./raptor_paper_db")
collection = client.get_or_create_collection("raptor")

vector_store = ChromaVectorStore(chroma_collection=collection)

raptor_pack = RaptorPack(
    documents,
    embed_model=OpenAIEmbedding(
        model="text-embedding-3-small"
    ),  # used for embedding clusters
    llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1),  # used for generating summaries
    vector_store=vector_store,  # used for storage
    similarity_top_k=2,  # top k for each layer, or overall top-k for collapsed
    mode="collapsed",  # sets default mode
    transformations=[
        SentenceSplitter(chunk_size=400, chunk_overlap=50)
    ],  # transformations applied for ingestion
)


retriever = RaptorRetriever(
    [],
    embed_model=OpenAIEmbedding(
        model="text-embedding-3-small"
    ),  # used for embedding clusters
    llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1),  # used for generating summaries
    vector_store=vector_store,  # used for storage
    similarity_top_k=2,  # top k for each layer, or overall top-k for collapsed
    mode="tree_traversal",  # sets default mode
)


raptor_query_engine = RetrieverQueryEngine.from_args(
    retriever, llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1,  use_async=True)
)

# Normal RAG
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# raptor_query_engine.query("Test")

def greet(prompt):
    print(prompt)
    response1 = raptor_query_engine.query(prompt)
    # response2 = query_engine.query(prompt).response
    print(response1)
    return response1

demo = gr.Interface(fn=greet, inputs=["text"], 
                    outputs=["text"],
                    title="Raptor RAG",concurrency_limit=None)
demo.launch(share=True)