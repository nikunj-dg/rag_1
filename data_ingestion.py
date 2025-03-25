import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS


"""
SETUP API KEYS
"""

# Load environment variables from .env file
load_dotenv('config.env')

# Get the API key
if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")



"""
SETUP RELEVANT PATHS AND MODELS 
"""

if 'config' not in locals(): config = {}

config['DOCS_DIR'] = f""
config['VECTOR_STORE_PATH'] = f""

# Initialize the embedding model 
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")



"""
FUNCTION DEFINITIONS 
"""

def load_docs(directory: str, file_types=("*.txt", "*.html", "*.pdf")):
    """
    Reads all files in the given directory and extracts text.
    Supports filtering by file type.
    
    Args:
        directory (str): Path to the directory containing documents.
        file_types (tuple): File extensions to load (default: TXT, HTML, PDF).
    
    Returns:
        list: Loaded LangChain document objects.
    """

    documents = []
    
    for file_type in file_types:
        loader = DirectoryLoader(directory, glob=file_type, show_progress=True)
        documents.extend(loader.load())

    print(f"\nINFO - Documents Loaded, ({len(documents)} files):")
    for doc in documents:
        print(f"- {os.path.basename(doc.metadata['source'])}")
        # print(f"- {os.path.basename(doc.metadata['source'])}", type(doc), doc)

    return documents

def split_docs(docs, chunk_size=1000, chunk_overlap=200, add_start_index=True):
    """
    Splits text from documents into chunks and allows some overlap for continuity between chunks.
    
    Args:
        documents (list): List of LangChain document objects to be split.
        chunk_size (int): The maximum size (in characters) of each chunk (default: 1000).
        chunk_overlap (int): The number of overlapping characters between chunks (default: 20).
        add_start_index(bool): This tracks the starting index of each chunk within the original document, can be useful for reconstructing or referencing chunks later.
    
    Returns:
        list: A list of LangChain document chunks after splitting.
    """
    if not docs:
        raise ValueError("The input documents list is empty.")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=add_start_index,
    )

    docs_chunks = text_splitter.split_documents(documents)
    # print(f"\nINFO - Documents loaded: {len(documents)}, Total Splits: {len(docs_chunks)}")

    return docs_chunks

def embed_chunks(docs_chunks):
    """
    Embeds document chunks using OpenAI's API to create vector embeddings.

    Args:
        docs_chunks (list): List of document chunks (Document object) to embed.
    
    Returns:
        list: A list of embeddings (vectors) for each document chunk.
    """ 
    print("INFO - Using emedding model: GoogleGenerativeAI")
    chunk_content = [doc.page_content for doc in docs_chunks]
    embedded_chunks = embeddings.embed_documents(chunk_content)

    print(f"\nINFO - {len(embedded_chunks)} Chunks Embedded")

    return embedded_chunks

def create_vector_store(docs_chunks):
    """
    Creates a FAISS vector store using the document chunks.

    Args:
        embedded_chunks (list): List of vector embeddings.

    Returns:
        FAISS vector store containing the embeddings.
    """
    vector_store = FAISS.from_documents(docs_chunks, embeddings)
    vector_store.save_local(config['VECTOR_STORE_PATH'])
    print("INFO - FAISS vector_store created")
    print("INFO - vector store path: ", config['VECTOR_STORE_PATH'])

    return vector_store



"""
MAIN
"""

print("\nStep: Loading the documents")
documents = load_docs(config['DOCS_DIR']) ## loads docs from directory

print("\nStep: Chunking the documents")
docs_chunks = split_docs(documents, chunk_overlap=100) ## split docs in chunks 

print("\nStep: Embedding the chunks of documents")
print("\nStep: Skipped - Embedding the chunks of documents")
# embedded_chunks = embed_chunks(docs_chunks) ## embed the chunks 
# print(embedded_chunks)

print("\nStep 4: Creating a vector store for chunks of documents after embedding them and storing it locally")
vector_store = create_vector_store(docs_chunks) ## embed the chunks 
# print(vector_store)
