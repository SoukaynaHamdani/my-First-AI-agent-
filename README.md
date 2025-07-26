# my-First-AI-agent-
# Smart Research Agent 🧠
 

![Status](https://img.shields.io/badge/Status-Working_&_Optimized-brightgreen )
![Python Version](https://img.shields.io/badge/Python-3.9+-blue )
![Framework](https://img.shields.io/badge/Framework-LangChain-yellow )
![Architecture](https://img.shields.io/badge/Architecture-Hybrid_Model-blueviolet )

This repository contains the code for an advanced, multi-tool AI agent designed to provide accurate and context-aware answers by intelligently choosing between a private knowledge base and a live web search.

The project's key feature is its **hybrid model architecture**, which strategically uses a free, local embedding model for data-intensive tasks and a powerful, reliable OpenAI LLM for high-level reasoning. This creates a solution that is both cost-effective and highly capable.

## The Problem

Standard Large Language Models (LLMs) are powerful but face key limitations for practical use:
1.  **Knowledge Cutoff:** They lack information about events that occurred after their training date.
2.  **Lack of Specialization:** They cannot provide deep, authoritative answers based on private or domain-specific documents.
3.  **API Costs:** Foundational tasks like creating text embeddings for large documents can be cost-prohibitive and quickly exhaust API quotas.

This project solves all three problems by creating an agent that can reason about a user's query, select the appropriate tool, and leverage the best model for each part of the task.

## Architecture and Design

The Smart Research Agent is built using the modern **ReAct (Reasoning and Acting)** framework within LangChain. It has access to two specialized tools:

1.  **Local Knowledge Base :**
    *   **Technology:** This tool is a Retrieval-Augmented Generation (RAG) pipeline.
    *   **Process:** It ingests a domain-specific PDF (the seminal AI paper "Attention Is All You Need"), processes it into a searchable vector index using FAISS, and uses it to answer deep technical questions.
    *   **Key Optimization:** To eliminate API costs for data processing, this pipeline uses a high-performance, open-source `HuggingFaceEmbeddings` model (`all-MiniLM-L6-v2`) that runs locally on the user's machine.

2.  **Google Web Search :**
    *   **Technology:** This tool leverages the Google Search API via `SerpAPIWrapper`.
    *   **Process:** It takes a query, searches the live internet, and returns up-to-date information.
    *   **Use Case:** Perfect for questions about recent events, current affairs, or any topic outside the scope of the local knowledge base.

The agent's "brain" is powered by **OpenAI's LLM**, which is the industry standard for reliable, high-quality reasoning required for agentic decision-making. The agent analyzes the user's question and writes a "thought" about which tool is best suited to answer it before taking action.

## Key Features & Technical Highlights

*   **Agentic Logic:** The agent autonomously decides which tool to use based on the query's context.
*   **Hybrid Model Strategy:** Demonstrates a sophisticated understanding of AI application design by using a free, local model for embeddings and a powerful cloud model (OpenAI) for reasoning, optimizing for both cost and performance.
*   **Modern Agent Framework:** Built with the latest `create_react_agent` and `AgentExecutor` classes, ensuring compatibility and robustness.
*   **Retrieval-Augmented Generation (RAG):** Shows the ability to ground LLM responses in factual, verifiable document sources.
*   **Verbose Reasoning:** The agent is set to `verbose=True` to print its chain of thought, providing full transparency into its decision-making process.

## Getting Started

### Prerequisites

*   Python 3.9+
*   An OpenAI API Key  
*   A SerpAPI API Key.

 .  **Set up your API keys:**
    Open the `agent.py` file and replace the placeholder text with your actual API keys:
    ```python
    os.environ["OPENAI_API_KEY"] = "PASTE_YOUR_OPENAI_API_KEY_HERE"
    os.environ["SERPAPI_API_KEY"] = "PASTE_YOUR_SERPAPI_KEY_HERE"
    ```

 .  **Run the agent:**
    ```bash
    python agent.py
    ```
    The first time you run the script, it may take a minute to download the local embedding model. The agent will then run two example questions to demonstrate its capabilities. Feel free to edit the questions at the bottom of `agent.py` to ask it anything you want!
