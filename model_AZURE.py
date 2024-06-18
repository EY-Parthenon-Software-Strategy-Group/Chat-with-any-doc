#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import AzureOpenAI
import os
import argparse
import time

load_dotenv()

# embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
# model_path = os.environ.get('MODEL_PATH')
# model_path = "c:/Users/RV414CE/OneDrive - EY/Desktop/Programming/privateGPT/models/ggml-gpt4all-j-v1.3-groovy.bin"
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

from constants import CHROMA_SETTINGS

def main():
    # Parse the command line arguments
    args = parse_arguments()
    
    embeddings = OpenAIEmbeddings(
        deployment="SSG_embedding",
        openai_api_base="https://caedaoipocaoa1l.openai.azure.com",
        openai_api_version="2023-07-01-preview",
        openai_api_key="d873529160934ec19f1276e20e6bd94a",
        openai_api_type="azure",
        chunk_size = 16,
        max_retries= 50)
    
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    
    # Prepare the LLM
    # BASE_URL = "https://usedaoipocaoa08.openai.azure.com"
    # API_KEY = "5093bb833f7e481fad3c5743fb082a4d"
    # DEPLOYMENT_NAME = "Davinci"

    BASE_URL = "https://caedaoipocaoa1l.openai.azure.com"
    API_KEY = "d873529160934ec19f1276e20e6bd94a"
    DEPLOYMENT_NAME = "gpt-4"
    
    llm = AzureOpenAI(
        openai_api_base=BASE_URL,
        # openai_api_version="2023-05-15",
        openai_api_version="2023-07-01-preview",
        deployment_name=DEPLOYMENT_NAME,
        openai_api_key=API_KEY,
        openai_api_type="azure",
    )
        
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)

    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue

        # Get the answer from the chain
        start = time.time()
        res = qa(query)
        answer, docs = res['result'], [] if args.hide_source else res['source_documents']
        end = time.time()

        # Print the result
        print("\n\n> Question:")
        print(query)
        print(f"\n> Answer (took {round(end - start, 2)} s.):")
        print(answer)

        # Print the relevant sources used for the answer
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)

def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()


if __name__ == "__main__":
    main()
