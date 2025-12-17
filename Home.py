import streamlit as st
import sys, os
sys.path.append(os.path.dirname(__file__))

from ui.layout import apply_dark_theme, sidebar_panel


st.set_page_config(
    page_title="AI Assistant Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_dark_theme()

# Check for navigation flag and switch page
if "go_to" in st.session_state:
    target_page = st.session_state.pop("go_to")
    st.switch_page(target_page)

# Make header minimal but keep sidebar toggle accessible
st.markdown("""
    <style>
    /* Keep header present but minimal - sidebar toggle must remain visible */
    header[data-testid="stHeader"],
    .stApp > header {
        visibility: visible !important;
        display: flex !important;
        height: 3rem !important;
        min-height: 3rem !important;
        padding: 0.5rem 1rem !important;
        background: transparent !important;
        border-bottom: none !important;
        opacity: 0.15 !important;
        transition: opacity 0.2s ease !important;
    }
    
    /* Ensure ALL buttons in header (including sidebar toggle) are always visible and clickable */
    header[data-testid="stHeader"] button,
    header[data-testid="stHeader"] > div button,
    .stApp > header button,
    .stApp > header > div button {
        opacity: 1 !important;
        visibility: visible !important;
        display: block !important;
        z-index: 1000 !important;
        pointer-events: auto !important;
    }
    
    /* Style sidebar toggle button to be more visible */
    header[data-testid="stHeader"] button[data-testid*="baseButton-header"],
    header[data-testid="stHeader"] button[aria-label*="menu"],
    header[data-testid="stHeader"] button[aria-label*="Menu"] {
        background: rgba(107, 45, 138, 0.4) !important;
        border: 1px solid rgba(139, 77, 170, 0.5) !important;
        border-radius: 6px !important;
        padding: 0.4rem 0.6rem !important;
    }
    
    /* Hover effect to make header more visible when needed */
    header[data-testid="stHeader"]:hover,
    .stApp > header:hover {
        opacity: 0.5 !important;
    }
    
    /* Minimize top spacing */
    .main .block-container {
        padding-top: 0.5rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar - hide tools status
sidebar_panel(
    openai_connected=False,
    tools_status=None,
    show_tools_status=False
)

# Hero Section - compact with all text
hero_html = """
<div class="hero-section">
    <h1 class="hero-title">AI Assistant Platform</h1>
    <p class="hero-tagline">Powerful AI agents designed to enhance your productivity and creativity</p>
</div>
"""
st.markdown(hero_html, unsafe_allow_html=True)

# Create cards for each assistant with updated titles
assistants = [
    {
        "icon": "🎯",
        "title": "Prompt Generator",
        "description": "Intelligent prompt engineering that gathers requirements and generates custom prompts through conversation",
        "page": "1_Basic_Chatbot"
    },
    {
        "icon": "🔍",
        "title": "Search Agent",
        "description": "Advanced research capabilities with web search, Wikipedia, and ArXiv integration for comprehensive information gathering",
        "page": "2_Chatbot_Agent"
    },
    {
        "icon": "📚",
        "title": "Agentic RAG",
        "description": "Chat with your documents using intelligent retrieval, relevance grading, and question rewriting for accurate answers",
        "page": "3_Chat_with_your_Data"
    },
    {
        "icon": "🔧",
        "title": "MCP Agent",
        "description": "Connect to external tools and services via Model Context Protocol for extended functionality",
        "page": "4_MCP_Agent"
    }
]

# Display cards in responsive grid (3 columns on wide screens, 2 on medium, 1 on small)
st.markdown("<div class='section-heading'>Choose Your AI Assistant</div>", unsafe_allow_html=True)

# Use 3 columns for better layout
cols = st.columns(3)
for idx, assistant in enumerate(assistants):
    col_idx = idx % 3
    with cols[col_idx]:
        # Card container - entire card is clickable
        page_path = f"pages/{assistant['page']}.py"
        
        # Card HTML with clickable class
        card_html = f"""
        <div class="premium-card card-animated clickable-card" data-page="{page_path}">
            <div class="premium-card-icon">{assistant['icon']}</div>
            <h3 class="premium-card-title">{assistant['title']}</h3>
            <p class="premium-card-description">{assistant['description']}</p>
            <div class="premium-card-cta-container">
                <span class="premium-card-cta-text">→ Explore Assistant</span>
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

# Add spacing at bottom
st.markdown("<br><br>", unsafe_allow_html=True)
