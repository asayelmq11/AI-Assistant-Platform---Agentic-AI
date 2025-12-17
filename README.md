# AI Assistant Platform 🤖

A modular Agentic AI platform built with Streamlit, demonstrating practical LLM workflows such as tool-augmented agents, Retrieval-Augmented Generation (RAG), vector search, and Model Context Protocol (MCP) integration.

---

## Overview

This project showcases how to build **intelligent AI assistants** using LangChain and LangGraph.
It focuses on real-world patterns including document-based question answering, external tool usage, and structured prompt engineering.

The platform is designed as a hands-on reference for developers exploring modern Agentic AI systems.

---

## Project Structure
```
├── project_code/                  # Core Streamlit application  
│   ├── .streamlit/                # Streamlit configuration  
│   │   └── config.toml  
│   ├── pages/                     # Application pages (agents & features)  
│   │   ├── 1_Prompt_Generator.py  # Prompt Generator agent  
│   │   ├── 2_Chatbot_Agent.py     # Tool-using chatbot agent  
│   │   ├── 3_Chat_with_your_Data.py # Agentic RAG (PDF chat + vector store)  
│   │   └── 4_MCP_Agent.py         # MCP-connected agent  
│   ├── tmp/                       # Temporary workspace for uploaded files  
│   ├── ui/                        # Shared UI layer  
│   │   ├── __pycache__/           # Python cache  
│   │   └── layout.py              # Layout, theme, and UI helpers  
│   ├── Home.py                    # Application entry point  
│   ├── requirements.txt           # Python dependencies  
│   └── README.md                  # Project documentation  
```
---

## Running the Project

```bash
cd project_code/
pip install -r requirements.txt
streamlit run Home.py

```
## Key Features

- Prompt Generator for structured and optimized prompt creation  
- Chatbot Agent with external tool usage (search, Wikipedia, reasoning)  
- Agentic RAG for querying uploaded PDF documents  
- Vector Store Visualization with clustering and chunk inspection  
- MCP Agent for connecting LLMs to external services via MCP  
- Streamlit UI Modular, multi-page  

---

## Built-in Applications 

- **Prompt Generator** – Guided prompt engineering assistant  
- **Chatbot Agent** – Conversational agent with integrated tools  
- **Chat with Your Data** – PDF-based Agentic RAG system  
- **MCP Agent** – LLM connected to MCP-compatible tools and services  

---

## Tech Stack

### Languages
- Python  

### Frameworks & Libraries
- Streamlit  
- LangChain  
- LangGraph  
- FAISS  
- Plotly  
- scikit-learn  

### APIs & Tools
- OpenAI API  
- Model Context Protocol (MCP) - zapier  
- Wikipedia  
- ArXiv  
- Tavily  
- DuckDuckGo Search
