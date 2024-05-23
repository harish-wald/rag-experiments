from flask import Flask, request, jsonify
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
import time
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

from llama_index.core import SimpleDirectoryReader

#def load_documents():
#    return 
_documents = SimpleDirectoryReader(input_files=["./Form Master Services Agreement (Outsourcing).DOCX"]).load_data()

#def get_vector_store():
client = chromadb.PersistentClient(path="./raptor_paper_db")
collection = client.get_or_create_collection("raptor")
_vector_store = ChromaVectorStore(chroma_collection=collection)
#return ChromaVectorStore(chroma_collection=collection)

#def create_engines(_documents, _vector_store):
# raptor_pack = RaptorPack(
#     _documents,
#     embed_model=OpenAIEmbedding(model="text-embedding-3-small"),
#     llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1),
#     vector_store=_vector_store,
#     similarity_top_k=2,
#     mode="collapsed",
#     transformations=[SentenceSplitter(chunk_size=400, chunk_overlap=50)]
# )

retriever = RaptorRetriever(
    [],
    embed_model=OpenAIEmbedding(model="text-embedding-3-small"),
    llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1),
    vector_store=_vector_store,
    similarity_top_k=2,
    mode="tree_traversal"
)

raptor_query_engine = RetrieverQueryEngine.from_args(
    retriever, llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1, use_async=True)
)

index = VectorStoreIndex.from_documents(_documents)
query_engine = index.as_query_engine()

    #return raptor_query_engine, query_engine



# documents = load_documents()
# vector_store = get_vector_store()

# raptor_query_engine ,query_engine = create_engines(documents, vector_store)


def query_raptor(input_prompt):
    print(f'entering inside raptor query engine raptor_query_engine {raptor_query_engine}')
    start_time = time.time()
    retriever = RaptorRetriever(
    [],
    embed_model=OpenAIEmbedding(model="text-embedding-3-small"),
    llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1),
    vector_store=_vector_store,
    similarity_top_k=2,
    mode="tree_traversal"
)

    raptor_query_engine1 = RetrieverQueryEngine.from_args(
        retriever, llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1, use_async=True)
    )

    response1 = raptor_query_engine1.query(input_prompt)
    end_time = time.time()
    print(f"Time taken by Raptor: {end_time - start_time} seconds")
    return response1

def query_vanilla(input_prompt):
    
    start_time = time.time()
    response2 = query_engine.query(input_prompt).response
    end_time = time.time()
    print(f"Time taken by Vanilla: {end_time - start_time} seconds")
    return response2


@app.route('/raptor',methods = ['POST'])
def hello_world():
    print('inside line 16')
    
    question = (request.json)['query']
    print(question)
    response1 = query_vanilla(question)
    response2 = query_raptor(question)
    # with ThreadPoolExecutor() as executor:
    #     future1 = executor.submit(query_raptor, question)
    #     future2 = executor.submit(query_vanilla, question)

    #     response1 = future1.result()
    #     response2 = future2.result()

    combined_response = {
        'response1': response1,
        'response2': response2.response
    }
    print(combined_response)
    return jsonify(combined_response)



if __name__ == '__main__':
   app.run(debug = True,port = 8000)