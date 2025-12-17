"""
Chatbot Agent with Multiple Tools - SOLUTION

This solution adds Wikipedia and ArXiv tools to the agent.
The agent can now search the web, look up encyclopedia articles, and find academic papers.
"""


# =========================================================
# IMPORTS (Libraries we need)
# =========================================================

# Streamlit: Framework for building web apps with Python
import streamlit as st
from ui.layout import apply_dark_theme, app_header, sidebar_panel

# os: For setting environment variables (API keys)
import os

# ChatOpenAI: Connects to OpenAI's GPT models (like ChatGPT)
from langchain_openai import ChatOpenAI

# TavilySearchResults: Tool for searching the web
from langchain_community.tools.tavily_search import TavilySearchResults

# ✨ NEW: Wikipedia and ArXiv tools
# WikipediaQueryRun: Tool for searching Wikipedia encyclopedia
# ArxivQueryRun: Tool for searching academic papers on ArXiv
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun

# ✨ NEW: API wrappers for Wikipedia and ArXiv
# WikipediaAPIWrapper: Handles Wikipedia API calls
# ArxivAPIWrapper: Handles ArXiv API calls
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import re

# create_react_agent: Creates an agent that can reason and use tools
from langgraph.prebuilt import create_react_agent




# =========================================================
# PAGE SETUP
# =========================================================

st.set_page_config(
    page_title="Multi-Tool Chatbot Agent",
    page_icon="🔍",
    layout="wide",  # Use full width of browser
    initial_sidebar_state="expanded"
)

apply_dark_theme()
app_header(
    title="Multi-Tool Chatbot Agent",
    caption="AI agent with web search, Wikipedia, and ArXiv capabilities",
    icon="🔍"
)


# =========================================================
# SESSION STATE
# =========================================================

if "openai_key" not in st.session_state:
    st.session_state.openai_key = ""  # Store OpenAI API key

if "tavily_key" not in st.session_state:
    st.session_state.tavily_key = ""  # Store Tavily API key

if "agent" not in st.session_state:
    st.session_state.agent = None  # Store the agent instance

if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []  # Store chat history

RECENCY_SYSTEM_PROMPT = (
    # Fresh, grounded answers: search first, then answer with sources
    "You are a grounded, recency-aware assistant. For every request, you must call Tavily web search "
    "before answering. Do not answer from prior knowledge. If Tavily returns no results, reply "
    "exactly: 'No reliable web source found.' If research papers are requested, also use arXiv. "
    "Prefer recent sources automatically (no hardcoded dates), and include publication dates when available. "
    "Provide a normal, helpful answer, then add a short 'Sources' section with 2–4 links from the Tavily results. "
    "If multiple sources disagree, present each version; do not reconcile. If insufficient recent results, say so."
)

# Expanded recency detection (EN + AR)
RECENT_QUERY_REGEX = re.compile(
    r"(latest|recent|update|updates|news|this month|this year|current|"
    r"آخر|اخر|أحدث|احدث|جديد|تحديث|تحديثات|مستجدات|اليوم|هذا الشهر|هذه السنة)",
    re.IGNORECASE,
)

# Non-capturing year pattern to return full years
YEAR_REGEX = re.compile(r"\b(?:19|20)\d{2}\b")


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.subheader("🔑 API Keys")
    
    if st.session_state.openai_key:
        st.success("✅ OpenAI Connected")
    else:
        st.warning("⚠️ OpenAI Not Connected")
    
    if st.session_state.tavily_key:
        st.success("✅ Tavily Connected")
    else:
        st.warning("⚠️ Tavily Not Connected")
    
    # ✨ NEW: Show available tools
    if st.session_state.openai_key and st.session_state.tavily_key:
        st.divider()
        st.subheader("🛠️ Available Tools")
        st.write("✅ **Tavily Search** - Web search")
        st.write("✅ **Wikipedia** - Encyclopedia")
        st.write("✅ **ArXiv** - Research papers")
    
    if st.session_state.openai_key or st.session_state.tavily_key:
        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.agent_messages = []
            st.rerun()
        
        if st.button("Change API Keys", use_container_width=True):
            # Reset everything to start fresh
            st.session_state.openai_key = ""
            st.session_state.tavily_key = ""
            st.rerun()


# =========================================================
# API KEYS INPUT
# =========================================================

# Check which keys we still need
keys_needed = []
if not st.session_state.openai_key:
    keys_needed.append("openai")
if not st.session_state.tavily_key:
    keys_needed.append("tavily")

if keys_needed:
    openai_key = st.session_state.openai_key
    tavily_key = st.session_state.tavily_key
    
    if "openai" in keys_needed:
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",  # Hide the key as user types
            placeholder="sk-proj-..."
        )
    
    if "tavily" in keys_needed:
        tavily_key = st.text_input(
            "Tavily API Key",
            type="password",  # Hide the key as user types
            placeholder="tvly-..."
        )
    
    if st.button("Connect"):
        valid = True
        
        # Validate OpenAI key format
        if "openai" in keys_needed:
            if not openai_key or not openai_key.startswith("sk-"):
                valid = False
        
        # Validate Tavily key format
        if "tavily" in keys_needed:
            if not tavily_key or not tavily_key.startswith("tvly-"):
                valid = False
        
        if valid:
            if "openai" in keys_needed:
                st.session_state.openai_key = openai_key
            if "tavily" in keys_needed:
                st.session_state.tavily_key = tavily_key
            st.rerun()  # Restart to show connected state
        else:
            st.error("❌ Invalid API key format")
    
    st.stop()  # Don't show chat interface until connected


# =========================================================
# CREATE AGENT
# =========================================================

if not st.session_state.agent:
    # Set API keys as environment variables (required by some tools)
    os.environ["OPENAI_API_KEY"] = st.session_state.openai_key
    os.environ["TAVILY_API_KEY"] = st.session_state.tavily_key
    
    # Create language model
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # Use GPT-4o-mini (fast and cheap)
        temperature=0.5 # 0 = deterministic, 1 = creative
    )
    
    # Create Tavily search tool (for web search)
    search_tool = TavilySearchResults(max_results=2)
    
    # ✨ NEW: Create Wikipedia tool (for encyclopedia articles)
    wikipedia = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(
            top_k_results=2,  # Return top 2 results
            doc_content_chars_max=500  # Limit content length
        ),
        name="wikipedia",
        description="""Search Wikipedia for encyclopedia articles, historical information, 
        biographies, and general knowledge. Best for: 'Who was...', 'What is...', 
        'History of...', 'Explain...' queries."""
    )
    
    # ✨ NEW: Create ArXiv tool (for academic papers)
    arxiv = ArxivQueryRun(
        api_wrapper=ArxivAPIWrapper(
            top_k_results=2,  # Return top 2 results
            doc_content_chars_max=500  # Limit content length
        ),
        name="arxiv",
        description="""Search ArXiv for academic papers, research articles, and scientific 
        publications. Best for: 'Latest research on...', 'Papers about...', 
        'Scientific studies on...' queries."""
    )
    
    # ✨ MODIFIED: Create agent with all three tools
    tools = [search_tool, wikipedia, arxiv]
    st.session_state.agent = create_react_agent(llm, tools)


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for idx, message in enumerate(st.session_state.agent_messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# =========================================================
# HANDLE USER INPUT
# =========================================================

user_input = st.chat_input("Ask me anything...")

if user_input:
    # Add user message to chat history
    st.session_state.agent_messages.append({
        "role": "user",
        "content": user_input
    })
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Generate response using agent
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching and thinking..."):
            def to_lc_messages(msgs):
                lc = []
                for m in msgs:
                    if m.get("role") == "user":
                        lc.append(HumanMessage(content=m.get("content", "")))
                    else:
                        lc.append(AIMessage(content=m.get("content", "")))
                return lc

            def is_recency_query(text: str) -> bool:
                return bool(RECENT_QUERY_REGEX.search(text))

            def contains_old_year(text: str) -> bool:
                years = [int(m.group(0)) for m in YEAR_REGEX.finditer(text)]
                return any(y < 2024 for y in years)

            def has_source_link(text: str) -> bool:
                return "http" in text

            history_tail = st.session_state.agent_messages[-8:]
            lc_messages = [SystemMessage(content=RECENCY_SYSTEM_PROMPT)] + to_lc_messages(history_tail)

            response = st.session_state.agent.invoke(
                {"messages": lc_messages},
                config={"recursion_limit": 8}
            )
            response_text = response["messages"][-1].content

            # Reliability guard: retry once if no sources/links or (recency ask and stale years)
            need_retry = (not has_source_link(response_text)) or (is_recency_query(user_input) and contains_old_year(response_text))
            if need_retry:
                guard_msg = SystemMessage(
                    content=(
                        "Your previous answer lacked web sources or used stale data. "
                        "Call Tavily search now (and arXiv if papers) before answering. "
                        "Provide a normal helpful answer, then a 'Sources' section with 2–4 links. "
                        "If Tavily returns no results, reply exactly: 'No reliable web source found.' "
                        "Do not rely on prior knowledge."
                    )
                )
                retry_messages = lc_messages + [guard_msg]
                response = st.session_state.agent.invoke(
                    {"messages": retry_messages},
                    config={"recursion_limit": 8}
                )
                response_text = response["messages"][-1].content

            # Display response as normal chat text
            st.markdown(response_text)
            
            # Add assistant response to chat history
            st.session_state.agent_messages.append({
                "role": "assistant",
                "content": response_text
            })
