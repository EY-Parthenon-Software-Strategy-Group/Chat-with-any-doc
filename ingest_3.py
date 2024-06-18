# sources:
# https://predictivehacks.com/get-started-with-chroma-db-and-retrieval-models-using-langchain/

# import the packages
import os
from typing import List
from multiprocessing import Pool
from tqdm import tqdm
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
import docx

# from langchain_community.document_loaders import (
#     CSVLoader,
#     EverNoteLoader,
#     PyMuPDFLoader,
#     TextLoader,
#     UnstructuredEmailLoader,
#     UnstructuredEPubLoader,
#     # UnstructuredHTMLLoader,''
#     # UnstructuredMarkdownLoader,
#     # UnstructuredODTLoader,
#     # UnstructuredPowerPointLoader,
#     # UnstructuredWordDocumentLoader,
# )

from langchain_community.document_loaders import (
    CSVLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    PyMuPDFLoader,
    TextLoader,
    Docx2txtLoader,
    # UnstructuredWordDocumentLoader,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.docstore.document import Document
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from constants import CHROMA_SETTINGS

load_dotenv()

# Load environment variables
persist_directory = os.environ.get('PERSIST_DIRECTORY')
source_directory = os.environ.get('SOURCE_DIRECTORY', 'privateGPT/source_documents')
chunk_size = 2000
chunk_overlap = 100

# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".docx": (Docx2txtLoader, {}),
    # ".doc": (UnstructuredWordDocumentLoader, {}),
    # ".docx": (UnstructuredWordDocumentLoader, {}),
    # ".enex": (EverNoteLoader, {}),
    # ".eml": (MyElmLoader, {}),
    # ".epub": (UnstructuredEPubLoader, {}),
    # ".html": (UnstructuredHTMLLoader, {}),
    # ".md": (UnstructuredMarkdownLoader, {}),
    # ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".xlsx": (UnstructuredExcelLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}

def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")

def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, docs in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                results.extend(docs)
                pbar.update()

    return results

def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split in chunks
    """
    print(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory, ignored_files)
    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {len(documents)} new documents from {source_directory}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    return texts

def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks if vectorstore exists
    """
    if os.path.exists(os.path.join(persist_directory, 'index')):
        if os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) and os.path.exists(os.path.join(persist_directory, 'chroma-embeddings.parquet')):
            list_index_files = glob.glob(os.path.join(persist_directory, 'index/*.bin'))
            list_index_files += glob.glob(os.path.join(persist_directory, 'index/*.pkl'))
            # At least 3 documents are needed in a working vectorstore
            if len(list_index_files) > 3:
                return True
    return False

def main():

    # Create embeddings
    embeddings = AzureOpenAIEmbeddings(
        deployment="SSG_embedding",
        azure_endpoint="https://caedaoipocaoa1l.openai.azure.com",
        openai_api_version="2023-07-01-preview",
        openai_api_key="d873529160934ec19f1276e20e6bd94a",
        openai_api_type="azure",
        chunk_size = 2000,
        max_retries= 50)
    
#     embeddings = embedding_functions.OpenAIEmbeddingFunction(
#         model_name="SSG_embedding",
#         api_key="d873529160934ec19f1276e20e6bd94a",
# )

    # if does_vectorstore_exist(persist_directory):
    #     # Update and store locally vectorstore
    #     print(f"Appending to existing vectorstore at {persist_directory}")
    #     # db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client=CHROMA_SETTINGS) # change accordingly
    #     # db = Chroma.from_embeddings(name="fuck", embedding_function=embeddings)
    #     db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
    #     collection = db.get()
    #     texts = process_documents([metadata['source'] for metadata in collection['metadatas']])
    #     print(f"Creating embeddings. May take some minutes...")
    #     db.add_documents(texts)
    # else:
    # Create and store locally vectorstore
    print("Creating new vectorstore")
    texts = process_documents()
    print(f"Creating embeddings. May take some minutes...")
    # db = Chroma(texts, embeddings, persist_directory=persist_directory, client=CHROMA_SETTINGS)
    # db = CHROMA_SETTINGS.create_collection(name="fuck", embedding_function=embeddings)
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
    # db = None
    db.persist()

    print(f"Ingestion complete! You can now run privateGPT.py to query your documents")


if __name__ == "__main__":
    main()