 import os  
import requests 
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain import hub
from langchain.tools import Tool

# Using a FREE model for embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
# Using the RELIABLE OpenAI model for the agent's brain
from langchain_openai import OpenAI
# Using SerpAPI for the search tool
from langchain_community.utilities import SerpAPIWrapper

# --- API Key Configuration ---
# A working OpenAI key with credits is required for the agent's brain.
os.environ["OPENAI_API_KEY"] = "PASTE_YOUR_OPENAI_API_KEY_HERE"
os.environ["SERPAPI_API_KEY"] = "PASTE_YOUR_SERPAPI_KEY_HERE"

print("✅ Step 1: Keys are set.")

# --- Part 1: The Local Knowledge Tool ---

pdf_url = "https://arxiv.org/pdf/1706.03762.pdf"
pdf_path = "attention_is_all_you_need.pdf"
if not os.path.exists(pdf_path ):
    print("Downloading knowledge base (PDF)...")
    response = requests.get(pdf_url)
    with open(pdf_path, 'wb') as f:
        f.write(response.content)
    print("Knowledge base downloaded.")

loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()
print(f"✅ Step 2: Knowledge base loaded with {len(pages)} pages.")

print("Initializing free embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Embedding model loaded.")

vector_store = FAISS.from_documents(pages, embeddings)
print("✅ Step 3: Created fast index from knowledge base.")

# Using the reliable OpenAI LLM for the agent's brain.
llm = OpenAI(temperature=0)
print("✅ Using reliable OpenAI LLM for agent's brain.")

# The QA chain for the PDF tool
pdf_qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever()
)
print("✅ Step 4: 'Local Knowledge' tool is ready.")


# --- Part 2: The Web Search Tool ---
search = SerpAPIWrapper()
print("✅ Step 5: 'Google Web Search' tool is ready.")


# --- Part 3: Assembling the Agent ---

tools = [
    Tool(
        name="Local Knowledge Base",
        func=pdf_qa_chain.run,
        description="Use this for any questions about machine learning, transformers, attention mechanisms, or the 'Attention Is All You Need' paper. This is your primary source for deep technical questions."
    ),
    Tool(
        name="Current Events Search",
        func=search.run,
        description="Use this for any general questions, recent events, or topics outside of the local knowledge base. If you don't know something, search the web."
    )
]

# Get the prompt template that is guaranteed to work with the OpenAI LLM
prompt = hub.pull("hwchase17/react")

# Create the modern ReAct agent
agent = create_react_agent(llm, tools, prompt)

# Create the executor to run the agent
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
print("\n✅ Step 6: Modern Agent assembled and ready to work!")
print("===================================================")

# --- Part 4: Running the Agent ---
question1 = "What is the self-attention mechanism?"
print(f"Executing Task 1: {question1}")
agent_executor.invoke({"input": question1})

print("\n===================================================\n")

question2 = "What is the latest news about the company Apple?"
print(f"Executing Task 2: {question2}")
agent_executor.invoke({"input": question2})
