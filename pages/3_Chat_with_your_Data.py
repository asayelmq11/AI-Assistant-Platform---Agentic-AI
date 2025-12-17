"""
Chat with your Data - Agentic RAG SOLUTION

This solution adds "agentic" capabilities to RAG:
- Agent decides if it needs to search documents
- Agent grades if retrieved documents are relevant
- Agent can rewrite questions for better results

This is called "Agentic RAG" - RAG with decision-making!
"""


# =========================================================
# IMPORTS (Libraries we need)
# =========================================================

# Streamlit: Framework for building web apps with Python
import streamlit as st
from ui.layout import apply_dark_theme, app_header, sidebar_panel

# os: For file operations and environment variables
import os

# ChatOpenAI: Connects to OpenAI's GPT models (like ChatGPT)
# OpenAIEmbeddings: Converts text to vectors (numbers) for similarity search
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ChatPromptTemplate: Template for formatting messages to the AI
from langchain_core.prompts import ChatPromptTemplate

# PyPDFLoader: Loads and reads PDF files
from langchain_community.document_loaders import PyPDFLoader

# FAISS: Fast similarity search database (stores document chunks as vectors)
from langchain_community.vectorstores import FAISS

# RecursiveCharacterTextSplitter: Splits long documents into smaller chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ✨ NEW: LangGraph tools for building agentic workflows
# StateGraph: Tool for building workflows with state management
# START, END: Special markers for workflow beginning and end
from langgraph.graph import StateGraph, START, END

# ✨ NEW: TypedDict for defining state structure
from typing_extensions import TypedDict

# ✨ NEW: Literal for specifying exact string values
from typing import Literal

import numpy as np
import pandas as pd
import plotly.express as px


# =========================================================
# PAGE SETUP
# =========================================================

st.set_page_config(
    page_title="Agentic RAG",
    page_icon="📚",
    layout="wide",  # Use full width of browser
    initial_sidebar_state="expanded"
)

apply_dark_theme()
app_header(
    title="Chat with your Data (Agentic RAG)",
    caption="AI agent that intelligently searches and answers from your documents",
    icon="📚"
)


# =========================================================
# SESSION STATE
# =========================================================

if "openai_key" not in st.session_state:
    st.session_state.openai_key = ""  # Store OpenAI API key

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None  # Store document database

if "llm" not in st.session_state:
    st.session_state.llm = None  # Store language model instance

if "rag_messages" not in st.session_state:
    st.session_state.rag_messages = []  # Store chat history

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []  # Track which files we've processed

# ✨ NEW: Store the agentic RAG workflow
if "rag_agent" not in st.session_state:
    st.session_state.rag_agent = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "chunk_text_preview" not in st.session_state:
    st.session_state.chunk_text_preview = []
if "chunk_source" not in st.session_state:
    st.session_state.chunk_source = []
if "chunk_page" not in st.session_state:
    st.session_state.chunk_page = []
if "retrieved_docs" not in st.session_state:
    st.session_state.retrieved_docs = []

visualize_vector_store = False
use_tsne = False
cluster_k = 6


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.subheader("🔑 API Keys")
    
    if st.session_state.openai_key:
        st.success("✅ OpenAI Connected")
        
        # ✨ NEW: Show agent capabilities
        if st.session_state.rag_agent:
            st.divider()
            st.subheader("🤖 Agent Capabilities")
            st.write("✅ **Search Documents**")
            st.write("✅ **Grade Relevance**")
            st.write("✅ **Rewrite Questions**")
            st.write("✅ **Generate Answers**")
        
        if st.session_state.rag_messages:
            st.divider()
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.rag_messages = []
                st.rerun()
        
        if st.button("Change API Keys", use_container_width=True):
            # Reset everything to start fresh
            st.session_state.openai_key = ""
            st.session_state.vector_store = None
            st.session_state.rag_messages = []
            st.session_state.rag_agent = None  # ✨ NEW: Reset agent
            st.rerun()
        
        st.divider()
        toggle_label = "📈 Visualize Vector Store"
        if hasattr(st.sidebar, "toggle"):
            visualize_vector_store = st.sidebar.toggle(toggle_label, value=False)
        else:
            visualize_vector_store = st.sidebar.checkbox(toggle_label, value=False)

        if visualize_vector_store:
            cluster_k = st.sidebar.slider("Number of clusters (k)", min_value=2, max_value=12, value=6, step=1)
            tsne_label = "Use t-SNE (slow)"
            if hasattr(st.sidebar, "toggle"):
                use_tsne = st.sidebar.toggle(tsne_label, value=False)
            else:
                use_tsne = st.sidebar.checkbox(tsne_label, value=False)
    else:
        st.warning("⚠️ Not Connected")


# =========================================================
# API KEY INPUT
# =========================================================

if not st.session_state.openai_key:
    # Show input form for API key
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",  # Hide the key as user types
        placeholder="sk-proj-..."
    )
    
    if st.button("Connect"):
        # Validate key format
        if api_key and api_key.startswith("sk-"):
            st.session_state.openai_key = api_key
            st.rerun()
        else:
            st.error("❌ Invalid API key format")
    
    st.stop()  # Don't show rest of app until connected


# =========================================================
# PDF UPLOAD AND PROCESSING
# =========================================================

uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True  # Allow multiple PDFs
)

if uploaded_files:
    current_files = [f.name for f in uploaded_files]
    
    # Check if we need to process (avoid reprocessing on every rerun)
    if st.session_state.processed_files != current_files:
        
        with st.spinner("Processing documents..."):
            # Load PDFs
            documents = []
            os.makedirs("tmp", exist_ok=True)
            
            for file in uploaded_files:
                # Save to temporary file (PyPDFLoader needs file path)
                file_path = os.path.join("tmp", file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getvalue())
                
                # Load PDF
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            
            # Split into chunks (long documents → smaller pieces)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,      # Each chunk max 1500 characters
                chunk_overlap=200     # 200 character overlap between chunks
            )
            chunks = text_splitter.split_documents(documents)
            st.session_state.chunks = chunks
            st.session_state.chunk_text_preview = [
                (c.page_content[:300] + ("..." if len(c.page_content) > 300 else ""))
                for c in chunks
            ]
            st.session_state.chunk_source = [
                (c.metadata.get("source") or c.metadata.get("file_name") or "unknown")
                for c in chunks
            ]
            st.session_state.chunk_page = [
                c.metadata.get("page")
                or c.metadata.get("page_number")
                or c.metadata.get("page_index")
                or c.metadata.get("page_num")
                for c in chunks
            ]
            st.session_state.retrieved_docs = []
            
            # Create vector store (searchable database)
            # Converts text chunks to vectors (numbers) for similarity search
            embeddings = OpenAIEmbeddings(api_key=st.session_state.openai_key)
            st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)

            
            # Create language model
            st.session_state.llm = ChatOpenAI(
                model="gpt-4o-mini",  # Use GPT-4o-mini (fast and cheap)
                temperature=0,  # 0 = deterministic, 1 = creative
                api_key=st.session_state.openai_key
            )
            
            # Reset chat and update processed files
            st.session_state.rag_messages = []
            st.session_state.processed_files = current_files
            st.session_state.rag_agent = None  # ✨ NEW: Reset agent when docs change
            
            st.success(f"✅ Processed {len(uploaded_files)} document(s)!")

# =========================================================
# 📈 OPTIONAL: VISUALIZE VECTOR STORE (Plotly + Clusters)
# =========================================================

if st.session_state.vector_store and visualize_vector_store:
    try:
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        if use_tsne:
            from sklearn.manifold import TSNE
        try:
            from streamlit_plotly_events import plotly_events
        except ImportError:
            plotly_events = None
    except ImportError:
        st.error("scikit-learn or plotly extras missing. Install with `pip install scikit-learn plotly streamlit-plotly-events`.")
    else:
        faiss_index = st.session_state.vector_store.index
        total_vectors = faiss_index.ntotal

        if total_vectors < 2:
            st.info("Need at least 2 embeddings to plot.")
        else:
            if total_vectors > 800:
                rng = np.random.default_rng(42)
                sample_indices = rng.choice(total_vectors, size=800, replace=False)
                vectors = [faiss_index.reconstruct(int(i)) for i in sample_indices]
            else:
                sample_indices = np.arange(total_vectors)
                vectors = faiss_index.reconstruct_n(0, total_vectors)

            X = np.array(vectors)

            if X.shape[0] < 2:
                st.info("Need at least 2 embeddings to plot.")
            else:
                cluster_count = min(cluster_k, X.shape[0])
                kmeans = KMeans(n_clusters=cluster_count, n_init=10, random_state=42)
                cluster_labels = kmeans.fit_predict(X)

                if use_tsne:
                    perplexity = min(30, max(1, X.shape[0] - 1))
                    reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
                    X_2d = reducer.fit_transform(X)
                    title = "Vector Store (t-SNE)"
                else:
                    reducer = PCA(n_components=2, random_state=42)
                    X_2d = reducer.fit_transform(X)
                    title = "Vector Store (PCA)"

                chunk_indices = sample_indices.tolist()
                # Safe metadata lookup
                sources = [
                    st.session_state.chunk_source[i] if i < len(st.session_state.chunk_source) else "unknown"
                    for i in chunk_indices
                ]
                pages = [
                    st.session_state.chunk_page[i] if i < len(st.session_state.chunk_page) else None
                    for i in chunk_indices
                ]
                previews = [
                    st.session_state.chunk_text_preview[i] if i < len(st.session_state.chunk_text_preview) else ""
                    for i in chunk_indices
                ]

                df = pd.DataFrame({
                    "x": X_2d[:, 0],
                    "y": X_2d[:, 1],
                    "cluster": cluster_labels.astype(int),
                    "chunk_index": chunk_indices,
                    "source": sources,
                    "page": pages,
                    "preview": previews,
                })

                fig = px.scatter(
                    df,
                    x="x",
                    y="y",
                    color=df["cluster"].astype(str),
                    hover_data={
                        "chunk_index": True,
                        "source": True,
                        "page": True,
                        "preview": True,
                        "cluster": True,
                        "x": False,
                        "y": False,
                    },
                    labels={"x": "Dim 1", "y": "Dim 2", "color": "Cluster"},
                    title=title,
                )
                fig.update_traces(marker=dict(size=8, opacity=0.8))
                fig.update_layout(height=520, legend_title_text="Cluster")

                cluster_counts = df["cluster"].value_counts().sort_index()
                st.caption(
                    f"{title} • points: {len(df)} • clusters: "
                    + ", ".join([f"{c}: {cluster_counts[c]}" for c in cluster_counts.index])
                )

                selected_chunk_idx = None
                if plotly_events:
                    selected_points = plotly_events(
                        fig,
                        click_event=True,
                        hover_event=False,
                        select_event=False,
                        override_height=560,
                        override_width="100%",
                    )
                    if selected_points:
                        point_idx = selected_points[0].get("pointIndex")
                        if point_idx is not None and point_idx < len(df):
                            selected_chunk_idx = int(df.iloc[point_idx]["chunk_index"])
                else:
                    st.plotly_chart(fig, use_container_width=True)
                    if len(df) > 0:
                        selected_point = st.selectbox(
                            "Select a point to view chunk text",
                            options=list(range(len(df))),
                            format_func=lambda i: f"#{i} | cluster {df.iloc[i]['cluster']} | {df.iloc[i]['source']} p{df.iloc[i]['page']}",
                        )
                        selected_chunk_idx = int(df.iloc[selected_point]["chunk_index"])

                if selected_chunk_idx is not None and selected_chunk_idx < len(st.session_state.chunks):
                    chunk = st.session_state.chunks[selected_chunk_idx]
                    meta = chunk.metadata or {}
                    source = meta.get("source") or meta.get("file_name") or "unknown"
                    page = meta.get("page") or meta.get("page_number") or meta.get("page_index") or meta.get("page_num")
                    with st.expander("Selected chunk", expanded=True):
                        st.markdown(f"**Source:** {source} | **Page:** {page} | **Index:** {selected_chunk_idx}")
                        st.write(chunk.page_content)


# =========================================================
# ✨ NEW BLOCK: CREATE AGENTIC RAG WORKFLOW
# =========================================================
# This is the new code that adds agentic capabilities

if st.session_state.vector_store and not st.session_state.rag_agent:
    
    # Define state structure (what information flows through the workflow)
    class AgentState(TypedDict):
        question: str  # User's question
        documents: list  # Retrieved documents
        retrieved_docs: list  # Top retrieved docs for UI
        generation: str  # Generated answer
        steps: list  # Track what the agent does
        rewrite_count: int  # ✅ NEW
    
    # Node 1: Retrieve documents
    def retrieve_documents(state: AgentState):
        """Search documents for relevant information."""
        question = state["question"]
        retriever = st.session_state.vector_store.as_retriever()
        docs = retriever.invoke(question)
        
        return {
            "documents": docs,
            "retrieved_docs": docs[:5],
            "steps": state.get("steps", []) + ["📚 Retrieved documents"]
        }
    
    # Node 2: Grade document relevance
    def grade_documents(state: AgentState) -> Literal["generate", "rewrite"]:
        """Check if retrieved documents are actually relevant."""
        question = state["question"]
        docs = state["documents"]

        if state.get("rewrite_count", 0) >= 2:
            return "generate"
        
        if not docs:
            return "generate"
        
        # Simple relevance check using LLM
        prompt = f"""Are these documents relevant to the question: "{question}"?
        
Documents: {docs[0].page_content[:500]}

Answer with just 'yes' or 'no'."""
        
        response = st.session_state.llm.invoke(prompt)
        is_relevant = "yes" in response.content.lower()
        
        return "generate" if is_relevant else "rewrite"
    
    # Node 3: Rewrite question
    def rewrite_question(state: AgentState):
        """Rewrite question for better search results."""
        question = state["question"]
        
        rewrite_prompt = f"Rewrite this question to be more specific and searchable: {question}"
        new_question = st.session_state.llm.invoke(rewrite_prompt).content
        
        return {
            "question": new_question,
            "retrieved_docs": state.get("retrieved_docs", state.get("documents", [])),
            "documents": state.get("documents", []),
            "rewrite_count": state.get("rewrite_count", 0) + 1,  # ✅ NEW
            "steps": state["steps"] + [f"🔄 Rewrote question: {new_question}"]
        }
    
    # Node 4: Generate answer
    def generate_answer(state: AgentState):
        """Generate final answer from documents."""
        question = state["question"]
        docs = state["documents"]
        
        if not docs:
            return {
                "generation": "I couldn't find relevant information in the documents.",
                "steps": state["steps"] + ["❌ No relevant documents found"]
            }
        
        # Combine documents into context
        context = "\n\n---\n\n".join(doc.page_content for doc in docs[:5])
        
        # Generate answer
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the question using ONLY the provided context. Be concise and accurate."),
            ("human", "Question: {question}\n\nContext: {context}\n\nAnswer:")
        ])
        
        response = st.session_state.llm.invoke(
            prompt.format_messages(question=question, context=context)
        )
        
        return {
            "generation": response.content,
            "documents": docs,
            "retrieved_docs": state.get("retrieved_docs", docs),
            "steps": state["steps"] + ["💬 Generated answer"]
        }
    
    # Build the workflow graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("grade", grade_documents)
    workflow.add_node("rewrite", rewrite_question)
    workflow.add_node("generate", generate_answer)
    
    # Define the flow
    workflow.add_edge(START, "retrieve")
    workflow.add_conditional_edges("retrieve", grade_documents)
    workflow.add_edge("rewrite", "retrieve")  # After rewrite, retrieve again
    workflow.add_edge("generate", END)
    
    # Compile and save
    st.session_state.rag_agent = workflow.compile()


# =========================================================
# CHAT INTERFACE
# =========================================================

if st.session_state.vector_store:
    
    # Display chat history
    for idx, message in enumerate(st.session_state.rag_messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Handle user input
    user_input = st.chat_input("Ask a question about your documents...")
    
    if user_input:
        # Save user message
        st.session_state.rag_messages.append({
            "role": "user",
            "content": user_input
        })
        
        with st.chat_message("user"):
            st.write(user_input)
        
        # ✨ MODIFIED: Generate response using agentic workflow
        with st.chat_message("assistant"):
            with st.spinner("🤖 Agent is working..."):
                
                # Run the agentic workflow
                result = st.session_state.rag_agent.invoke({
                     "question": user_input,
                     "documents": [],
                     "retrieved_docs": [],
                     "generation": "",
                     "steps": [],
                     "rewrite_count": 0
                })
                
                # Display final answer as normal chat text
                response_text = result["generation"]
                st.markdown(response_text)
                st.session_state.retrieved_docs = result.get("retrieved_docs", [])

                retrieved_docs = st.session_state.retrieved_docs or []
                if retrieved_docs:
                    with st.expander("🔎 Retrieved chunks (top 5)", expanded=False):
                        for idx, doc in enumerate(retrieved_docs):
                            meta = doc.metadata or {}
                            source = meta.get("source") or meta.get("file_name") or "unknown"
                            page = meta.get("page") or meta.get("page_number") or meta.get("page_index") or meta.get("page_num")
                            score = meta.get("score") or meta.get("distance") or meta.get("similarity")
                            header = f"**Chunk {idx + 1}** — source: {source}"
                            if page is not None:
                                header += f" | page: {page}"
                            if score is not None:
                                header += f" | score: {score}"
                            st.markdown(header)
                            preview_text = doc.page_content[:300]
                            if len(doc.page_content) > 300:
                                preview_text += "..."
                            st.markdown(preview_text)
                            with st.expander("Show full chunk", expanded=False):
                                st.write(doc.page_content)
                
                # Save assistant response
                st.session_state.rag_messages.append({
                    "role": "assistant",
                    "content": response_text
                })

else:
    st.info("📄 Please upload PDF documents to start chatting.") 