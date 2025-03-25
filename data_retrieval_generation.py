import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
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

# Initialize the llm model 
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)



"""
FUNCTION DEFINITIONS 
"""
def load_vector_store():
    """
    Load the vector_store stored locally at config['VECTOR_STORE_PATH']
    """
    try:
        vector_store = FAISS.load_local(
            folder_path=config['VECTOR_STORE_PATH'], 
            embeddings=embeddings, 
            allow_dangerous_deserialization=True  # Fix for pickle loading
        )

        return vector_store
    except Exception as e:
        print(f"Error loading vector_store: {e}")
        
        return None


def retrieve_data(query, vector_store):
    """
    Searches the vector store for data similar to the query.

    Args:
        query (str): Input query.

    Returns:
        List of similar data retrieved from the vector store.
    """
    try:
        retrieved_data = vector_store.similarity_search(query)

        return retrieved_data
    except Exception as e:
        print(f"Error retrieving data: {e}")
        
        return []


def generate_answer(query, retrieved_data):
  """
  Generates an answer using an LLM based on the user query and retrieved documents.

  Args:
      query (str): The user's question.
      retrieved_data (str): The most relevant retrieved information related to the query.

  Returns:
      str: The AI-generated response.
  """
  messages = [
        (
            "system",
            """You are an AI assistant that provides helpful responses using retrieved documents and user queries. 
            Try to give answers that are true and verified. If you don't know the answer, say so. Keep the asnwer 
            concise and to the point""",
        ),
        (
            "human",
            f"User Query: {query}\n\nRetrieved Documents:\n{retrieved_data}",
        ),
    ]

    try:
        response = llm.invoke(messages)

        return response
    except Exception as e:
        return f"An error occurred while generating a response: {e}" 



"""
MAIN
"""

print("\nStep: Loading the vector_store")
vector_store = load_vector_store() ## loads vector_store

query = input("Enter the query: ") # Get the query from the user 
# query = "What are the types of electric vehicles ?"

print("\nStep: Retrieving similar data from the vector_store")
retrieved_data = retrieve_data(query, vector_store) ## loads similar data from vector store 
retrieved_text = "\n".join([doc.page_content for doc in retrieved_data])
print("Text:", retrieved_text)

print("\nStep: Generating a response using LLM")
response = generate_answer(query, retrieved_data) ## loads similar data from vector store 
print("Response:", response.content)



