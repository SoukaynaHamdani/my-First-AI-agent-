  
import os
import requests
 
from langchain.agents import AgentType, initialize_agent, Tool
from langchain_community.agent_toolkits import load_tools
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# --- Imports for a FREE Embedding Model and the OpenAI LLM ---
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAI

# --- API Key Configuration ---
os.environ["OPENAI_API_KEY"] = "PASTE_YOUR_OPENAI_API_KEY_HERE"
os.environ["SERPAPI_API_KEY"] = "PASTE_YOUR_SERPAPI_KEY_HERE"

print("✅ Step 1: Keys are set.")

# --- Part 1: The Local Knowledge Tool ---

# Download the PDF if it doesn't exist
pdf_url = "https://arxiv.org/pdf/1706.03762.pdf"
pdf_path = "attention_is_all_you_need.pdf"
if not os.path.exists(pdf_path ):
    print("Downloading knowledge base (PDF)...")
    response = requests.get(pdf_url)
    with open(pdf_path, 'wb') as f:
        f.write(response.content)
    print("Knowledge base downloaded.")

# Load and process the PDF
loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()
print(f"✅ Step 2: Knowledge base loaded with {len(pages)} pages.")

# Use the FREE HuggingFace Embedding Model
print("Initializing free embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Embedding model loaded.")

# Create the vector store using the free embeddings
vector_store = FAISS.from_documents(pages, embeddings)
print("✅ Step 3: Created fast index from knowledge base.")

# Create the retrieval chain
retriever = vector_store.as_retriever()
pdf_qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),
    chain_type="stuff",
    retriever=retriever
)
print("✅ Step 4: 'Local Knowledge' tool is ready.")


# --- Part 2: The Web Search Tool ---
# =================================================================
# THE FIX IS HERE: We remove the llm=OpenAI() parameter from this call.
# The agent will use the main llm we define later.
# =================================================================
search_tools = load_tools(["google-serp"])
print("✅ Step 5: 'Google Web Search' tool is ready.")


# --- Part 3: Assembling the Agent ---

# Combine the tools
tools = [
    Tool(
        name="Local Knowledge Base",
        func=pdf_qa_chain.run,
        description="Use this for any questions about machine learning, transformers, attention mechanisms, or the 'Attention Is All You Need' paper. This is your primary source for deep technical questions."
    )
] + search_tools

# Initialize the agent's brain
llm = OpenAI(temperature=0)

# Build the agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
print("\n✅ Step 6: Agent assembled and ready to work!")
print("===================================================")

# --- Part 4: Running the Agent ---
question1 = "What is the self-attention mechanism?"
print(f"Executing Task 1: {question1}")
agent.run(question1)

print("\n===================================================\n")

question2 = "What is the latest news about the company Apple?"
print(f"Executing Task 2: {question2}")
agent.run(question2)

