# my-First-AI-agent-
# Smart Research Agent v2.0 🧠

![AI Agent Demo](https://img.shields.io/badge/AI-Agent-blueviolet )
![Python Version](https://img.shields.io/badge/Python-3.9+-blue )
![Framework](https://img.shields.io/badge/Framework-LangChain-yellow )
![Status](https://img.shields.io/badge/Status-Optimized-brightgreen )

An advanced, multi-tool AI agent designed to provide accurate and context-aware answers by intelligently choosing between a private knowledge base and a live web search. This version has been optimized for cost-efficiency and updated with the latest libraries.

## The Problem

Standard Large Language Models (LLMs) face two major limitations:
1.  **Knowledge Cutoff:** They lack information about events that occurred after their training date.
2.  **Lack of Specialization:** They cannot provide deep, authoritative answers based on private or domain-specific documents.
3.  **API Costs:** Foundational tasks like creating text embeddings can be cost-prohibitive for large documents, quickly exhausting API quotas.

This project solves all three problems by creating an agent that can reason about a user's query, select the appropriate tool, and leverage free, local, open-source models for heavy-lifting tasks.

## How It Works

The Smart Research Agent is built on a "Zero-shot ReAct" framework, allowing it to reason and decide on a sequence of actions. It has access to two specialized tools:

1.  **Local Knowledge Base (The Expert):**
    *   **Technology:** This tool is a Retrieval-Augmented Generation (RAG) pipeline.
    *   **Process:** It ingests a domain-specific PDF (the seminal AI paper "Attention Is All You Need"), processes it into a searchable vector index using FAISS, and uses it to answer deep technical questions.
    *   **Key Optimization:** To eliminate API costs and quota limits, this pipeline uses a high-performance, open-source `HuggingFaceEmbeddings` model (`all-MiniLM-L6-v2`) that runs locally.

2.  **Google Web Search (The Journalist):**
    *   **Technology:** This tool leverages the Google Search API via SerpAPI.
    *   **Process:** It takes a query, searches the live internet, and returns up-to-date information.
    *   **Use Case:** Perfect for questions about recent events, current affairs, or any topic outside the scope of the local knowledge base.

The agent's "brain" (powered by OpenAI's LLM) analyzes the user's question and writes a "thought" about which tool is best suited to answer it before taking action.

## Key Features & Technical Highlights

*   **Agentic Logic:** The agent autonomously decides which tool to use based on the query's context.
*   **Hybrid Model Approach:** Strategically uses a paid OpenAI LLM for high-level reasoning and a free, local embedding model for the data-intensive task of document indexing, creating a cost-effective and powerful solution.
*   **Modern Codebase:** All `langchain` imports have been updated to reflect the latest `langchain-community` and `langchain-openai` standards, resolving all deprecation warnings.
*   **Retrieval-Augmented Generation (RAG):** Demonstrates the ability to build a system that grounds LLM responses in factual, verifiable document sources.
*   **Verbose Reasoning:** The agent is set to `verbose=True` to print its chain of thought, providing full transparency into its decision-making process.

## Getting Started

### Prerequisites

*   Python 3.9+
*   An OpenAI API Key (for the agent's brain)
*   A SerpAPI API Key (for Google Search)

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/My-AI-Agent.git
    cd My-AI-Agent
    ```

2.  **Create and populate `requirements.txt`:**
    Create a file named `requirements.txt` and paste the following into it:
    ```
    langchain
    langchain-openai
    langchain-community
    pypdf
    faiss-cpu
    tiktoken
    google-search-results
    requests
    sentence-transformers
    ```

3.  **Install the required Python libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your API keys:**
    Open the `agent.py` file and replace the placeholder text with your actual API keys:
    ```python
    os.environ["OPENAI_API_KEY"] = "PASTE_YOUR_OPENAI_API_KEY_HERE"
    os.environ["SERPAPI_API_KEY"] = "PASTE_YOUR_SERPAPI_KEY_HERE"
    ```

5.  **Run the agent:**
    ```bash
    python agent.py
    ```
    The first time you run the script, it may take a minute to download the local embedding model. The agent will then run two example questions to demonstrate its capabilities. Feel free to edit the questions at the bottom of `agent.py` to ask it anything you want!
