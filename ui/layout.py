"""
Shared UI components for the Streamlit app
Provides consistent header, sidebar, and styling across all pages
"""

import streamlit as st
import os


def apply_dark_theme():
    """Apply dark mode CSS styling"""
    st.markdown(r"""
    <style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Main dark theme */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a0a1a 50%, #0f0a0f 100%);
        color: #e0e0e0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Main content area - minimal top padding for home page */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 3rem;
    }
    
    /* Remove extra spacing on home page */
    .main .block-container:first-child {
        padding-top: 0.5rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a0a1a 0%, #0f0a0f 100%);
        border-right: 1px solid #2a1a2a;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #d0d0d0;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #e8d4f0 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6b2d8a 0%, #4a1a5a 100%);
        color: #ffffff;
        border: 1px solid #8b4daa;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #7b3d9a 0%, #5a2a6a 100%);
        border-color: #9b5dba;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(139, 77, 170, 0.3);
    }
    
    /* Text inputs */
    .stTextInput > div > div > input {
        background: #1a0a1a;
        color: #e0e0e0;
        border: 1px solid #3a2a3a;
        border-radius: 6px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #6b2d8a;
        box-shadow: 0 0 0 2px rgba(107, 45, 138, 0.2);
    }
    
    /* File uploader */
    .stFileUploader {
        background: #1a0a1a;
        border: 1px dashed #3a2a3a;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Chat messages - enhanced styling */
    [data-testid="stChatMessage"] {
        padding: 1.25rem;
        margin: 0.75rem 0;
        border-radius: 16px;
        background: linear-gradient(135deg, #1a0a2a 0%, #2a1a3a 100%);
        border-left: 4px solid #6b2d8a;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        transition: all 0.2s ease;
    }
    
    [data-testid="stChatMessage"]:hover {
        box-shadow: 0 4px 12px rgba(107, 45, 138, 0.2);
        transform: translateX(2px);
    }
    
    /* User messages - distinct styling */
    [data-testid="stChatMessage"]:nth-child(odd) {
        background: linear-gradient(135deg, #2a1a3a 0%, #3a2a4a 100%);
        border-left-color: #8b4daa;
    }
    
    /* Assistant messages */
    [data-testid="stChatMessage"]:nth-child(even) {
        background: linear-gradient(135deg, #1a0a2a 0%, #2a1a3a 100%);
        border-left-color: #6b2d8a;
    }
    
    /* Chat message icon/avatar colors - elegant purple gradients */
    /* Remove red/orange/blue/pink colors and apply purple gradients */
    
    /* User message icon - elegant gradient (#3A145C → #4B1E78) */
    [data-testid="stChatMessage"] [data-testid="stAvatar"],
    [data-testid="stChatMessage"] [data-testid="stAvatar"] > div,
    [data-testid="stChatMessage"] [data-testid="stAvatar"] > span {
        background: linear-gradient(135deg, #3A145C, #4B1E78) !important;
        background-color: #3A145C !important;
        box-shadow: 0 0 8px rgba(111, 59, 184, 0.25) !important;
    }
    
    /* User message avatar container - elegant gradient */
    [data-testid="stChatMessage"] > div:first-child,
    [data-testid="stChatMessage"] > div:first-child > div {
        background: linear-gradient(135deg, #3A145C, #4B1E78) !important;
        background-color: #3A145C !important;
    }
    
    /* Override inline styles with red/orange/blue/pink colors - apply user gradient */
    [data-testid="stChatMessage"] [style*="rgb(255, 87"],
    [data-testid="stChatMessage"] [style*="rgb(255, 152"],
    [data-testid="stChatMessage"] [style*="rgb(255, 193"],
    [data-testid="stChatMessage"] [style*="rgb(255, 87, 34"],
    [data-testid="stChatMessage"] [style*="rgb(255, 152, 0"],
    [data-testid="stChatMessage"] [style*="rgb(255, 193, 7"],
    [data-testid="stChatMessage"] [style*="rgb(33, 150, 243"],
    [data-testid="stChatMessage"] [style*="rgb(63, 81, 181"],
    [data-testid="stChatMessage"] [style*="rgb(233, 30, 99"],
    [data-testid="stChatMessage"] [style*="rgb(156, 39, 176"] {
        background: linear-gradient(135deg, #3A145C, #4B1E78) !important;
        background-color: #3A145C !important;
    }
    
    /* Assistant message avatars - elegant gradient (#5A2D91 → #6F3BB8) */
    /* Use nth-child to target assistant messages (typically even) */
    [data-testid="stChatMessage"]:nth-child(even) [data-testid="stAvatar"],
    [data-testid="stChatMessage"]:nth-child(even) [data-testid="stAvatar"] > div,
    [data-testid="stChatMessage"]:nth-child(even) [data-testid="stAvatar"] > span,
    [data-testid="stChatMessage"]:nth-child(even) > div:first-child,
    [data-testid="stChatMessage"]:nth-child(even) > div:first-child > div {
        background: linear-gradient(135deg, #5A2D91, #6F3BB8) !important;
        background-color: #5A2D91 !important;
        box-shadow: 0 0 8px rgba(111, 59, 184, 0.25) !important;
    }
    
    /* Style SVG icons in avatars - user gradient (lighter end) */
    [data-testid="stChatMessage"] svg,
    [data-testid="stChatMessage"] [data-testid="stAvatar"] svg {
        color: #4B1E78 !important;
        fill: #4B1E78 !important;
    }
    
    /* Assistant SVG icons - assistant gradient (lighter end) */
    [data-testid="stChatMessage"]:nth-child(even) svg,
    [data-testid="stChatMessage"]:nth-child(even) [data-testid="stAvatar"] svg {
        color: #6F3BB8 !important;
        fill: #6F3BB8 !important;
    }
    
    /* Chat input */
    [data-testid="stChatInput"] {
        background: #1a0a1a;
        border-top: 1px solid #2a1a2a;
    }
    
    [data-testid="stChatInput"] textarea {
        background: #0f0a0f;
        color: #e0e0e0;
        border: 1px solid #3a2a3a;
        border-radius: 8px;
    }
    
    /* Cards and containers - premium hover effects */
    .card {
        background: linear-gradient(135deg, #1a0a1a 0%, #2a1a2a 100%);
        border: 1px solid #3a2a3a;
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    
    .card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(107, 45, 138, 0.1), transparent);
        transition: left 0.5s ease;
    }
    
    .card:hover::before {
        left: 100%;
    }
    
    .card:hover {
        border-color: #6b2d8a;
        transform: translateY(-6px) scale(1.02);
        box-shadow: 0 12px 32px rgba(107, 45, 138, 0.4);
        background: linear-gradient(135deg, #2a1a2a 0%, #3a2a3a 100%);
    }
    
    /* Success/Info/Warning/Error messages */
    .stSuccess {
        background: linear-gradient(135deg, #1a3a1a 0%, #2a4a2a 100%);
        border-left: 3px solid #4a8a4a;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #1a2a3a 0%, #2a3a4a 100%);
        border-left: 3px solid #4a6a8a;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #3a2a1a 0%, #4a3a2a 100%);
        border-left: 3px solid #8a6a4a;
    }
    
    .stError {
        background: linear-gradient(135deg, #3a1a1a 0%, #4a2a2a 100%);
        border-left: 3px solid #8a4a4a;
    }
    
    /* Spinner - enhanced styling */
    .stSpinner > div {
        border-color: #6b2d8a transparent transparent transparent;
        border-width: 3px;
    }
    
    /* Loading text styling */
    .loading-text {
        color: #b0b0b0;
        font-style: italic;
        margin-left: 0.5rem;
    }
    
    /* Expander */
    [data-testid="stExpander"] {
        background: #1a0a1a;
        border: 1px solid #3a2a3a;
        border-radius: 8px;
    }
    
    /* Divider */
    hr {
        border-color: #3a2a3a;
    }
    
    /* Caption */
    .stCaption {
        color: #b0b0b0;
    }
    
    /* Markdown text */
    .stMarkdown {
        color: #d0d0d0;
    }
    
    /* Links */
    a {
        color: #8b4daa;
    }
    
    a:hover {
        color: #9b5dba;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0f0a0f;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #3a2a3a;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #4a3a4a;
    }
    
    /* Copy button styling */
    .copy-btn {
        background: rgba(107, 45, 138, 0.1);
        border: 1px solid #3a2a3a;
        color: #b0b0b0;
        border-radius: 6px;
        padding: 0.4rem 0.8rem;
        font-size: 0.85rem;
        transition: all 0.2s ease;
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
    }
    
    .copy-btn:hover {
        border-color: #6b2d8a;
        color: #e8d4f0;
        background: rgba(107, 45, 138, 0.2);
        transform: scale(1.05);
    }
    
    /* Typography improvements */
    h1, h2, h3, h4, h5, h6 {
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    
    /* Input fields - enhanced */
    .stTextInput > div > div > input::placeholder {
        color: #6a5a6a;
    }
    
    /* Better spacing */
    .element-container {
        margin-bottom: 1.5rem;
    }
    
    /* Hero section styling - left-aligned, premium product feel */
    .hero-section {
        text-align: left;
        padding: 0;
        margin-top: 0;
        margin-bottom: 0;
        animation: fadeInUp 0.8s ease-out;
        max-width: 100%;
    }
    
    .hero-title {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #EADFFF 0%, #C7A6FF 50%, #9F7BFF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        color: transparent;
        margin-top: 0;
        margin-bottom: 0.5rem;
        padding-bottom: 0;
        letter-spacing: -0.04em;
        line-height: 1.1;
        text-shadow: 0 0 30px rgba(234, 223, 255, 0.15);
    }
    
    .hero-tagline {
        font-size: 1.1rem;
        color: #9a8aaa;
        font-weight: 400;
        margin: 0;
        padding-top: 0;
        line-height: 1.5;
        letter-spacing: 0.01em;
    }
    
    /* Section heading - minimal divider style */
    .section-heading {
        text-align: left;
        color: #7a6a8a;
        font-weight: 600;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 1.5rem;
        margin-bottom: 0;
        padding-top: 0;
        padding-bottom: 0;
    }
    
    /* Remove spacing after section heading */
    .section-heading + * {
        margin-top: 0;
    }
    
    /* Entrance animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
        }
        to {
            opacity: 1;
        }
    }
    
    .card-animated {
        animation: fadeInUp 0.6s ease-out;
        animation-fill-mode: both;
    }
    
    .card-animated:nth-child(1) { animation-delay: 0.1s; }
    .card-animated:nth-child(2) { animation-delay: 0.2s; }
    .card-animated:nth-child(3) { animation-delay: 0.3s; }
    .card-animated:nth-child(4) { animation-delay: 0.4s; }
    .card-animated:nth-child(5) { animation-delay: 0.5s; }
    
    /* Ensure Streamlit columns have equal heights for card alignment */
    [data-testid="column"] {
        display: flex !important;
        flex-direction: column !important;
        align-items: stretch !important;
    }
    
    [data-testid="column"] > div {
        display: flex !important;
        flex-direction: column !important;
        flex: 1 !important;
        height: 100% !important;
    }
    
    /* Premium card styling for home page */
    .premium-card {
        background: linear-gradient(135deg, #1a0a1a 0%, #2a1a2a 100%);
        border: 1px solid #3a2a3a;
        border-radius: 20px;
        padding: 2.5rem;
        margin: 1.5rem 0;
        margin-top: 1rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        position: relative;
        overflow: hidden;
        height: 100%;
        min-height: 22rem;
        display: flex;
        flex-direction: column;
        flex: 1;
    }
    
    .premium-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(107, 45, 138, 0.15), transparent);
        transition: left 0.6s ease;
    }
    
    .premium-card:hover::before {
        left: 100%;
    }
    
    .premium-card:hover {
        border-color: #6b2d8a;
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 16px 40px rgba(107, 45, 138, 0.5);
        background: linear-gradient(135deg, #2a1a2a 0%, #3a2a3a 100%);
    }
    
    .premium-card-icon {
        font-size: 3.5rem;
        margin-bottom: 1.5rem;
        filter: drop-shadow(0 4px 8px rgba(107, 45, 138, 0.4));
        transition: transform 0.3s ease;
    }
    
    .premium-card:hover .premium-card-icon {
        transform: scale(1.1) rotate(5deg);
    }
    
    .premium-card-title {
        color: #e8d4f0;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        letter-spacing: -0.01em;
    }
    
    .premium-card-description {
        color: #b0b0b0;
        line-height: 1.7;
        font-size: 1rem;
        flex-grow: 1;
        margin-bottom: 1.5rem;
    }
    
    .premium-card-cta {
        color: #8b4daa;
        font-size: 0.9rem;
        font-weight: 500;
        margin-top: auto;
        padding-top: 1.5rem;
        border-top: 1px solid #3a2a3a;
        transition: color 0.3s ease;
    }
    
    .premium-card:hover .premium-card-cta {
        color: #9b5dba;
    }
    
    /* CTA container inside card - bottom position */
    .premium-card-cta-container {
        margin-top: auto;
        padding-top: 1.5rem;
        border-top: 1px solid #3a2a3a;
        position: relative;
    }
    
    /* Inline text styling */
    .premium-card-cta-text {
        color: #7a6a8a;
        font-size: 0.85rem;
        font-weight: 400;
        transition: all 0.2s ease;
        display: inline-block;
    }
    
    /* Hover effect on card affects text */
    .premium-card:hover .premium-card-cta-text {
        color: #9b5dba;
        text-decoration: underline;
        text-decoration-color: #9b5dba;
        text-underline-offset: 3px;
    }
    
    /* Make clickable cards have pointer cursor */
    .clickable-card {
        cursor: pointer;
        position: relative;
    }
    
    /* Invisible button that covers entire card - use negative margin to overlay */
    .premium-card + div[data-testid="stButton"] {
        margin-top: -20rem !important;
        margin-bottom: 0 !important;
        padding: 0 !important;
        position: relative;
        z-index: 10 !important;
        height: 0 !important;
    }
    
    .premium-card + div[data-testid="stButton"] > button {
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        width: 100% !important;
        height: 20rem !important;
        background: transparent !important;
        border: none !important;
        color: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
        cursor: pointer !important;
        box-shadow: none !important;
        z-index: 10 !important;
        font-size: 0 !important;
        line-height: 0 !important;
    }
    
    .premium-card + div[data-testid="stButton"] > button:hover,
    .premium-card + div[data-testid="stButton"] > button:focus,
    .premium-card + div[data-testid="stButton"] > button:active {
        background: transparent !important;
        border: none !important;
        color: transparent !important;
        transform: none !important;
        box-shadow: none !important;
    }
    
    /* Darker purple button for Connect */
    .stButton > button.connect-btn {
        background: linear-gradient(3F1A6B, #4a1a5a 0%, #3a0a4a 100%);
        border-color: #5a2a6a;
    }
    
    .stButton > button.connect-btn:hover {
        background: linear-gradient(3F1A6B, #5a2a6a 0%, #4a1a5a 100%);
        border-color: #6a3a7a;
    }
    </style>
    """, unsafe_allow_html=True)


def app_header(title: str, caption: str = "", icon: str = ""):
    """Create a consistent app header"""
    apply_dark_theme()
    
    # Build header with raw HTML to completely bypass markdown parsing
    icon_html = f'<span>{icon}</span> ' if icon else ""
    title_html = f'<h1 style="color: #e8d4f0; margin: 0; font-size: 2.5rem; font-weight: 700; padding: 2rem 0 0.5rem 0;">{icon_html}{title}</h1>'
    caption_html = f'<p style="color: #b0b0b0; margin: 0.5rem 0 0 0; font-size: 1.1rem; padding-bottom: 1rem;">{caption}</p>' if caption else '<div style="padding-bottom: 1rem;"></div>'
    border_html = '<div style="border-bottom: 1px solid #2a1a2a; margin-bottom: 2rem;"></div>'
    
    # Use a single markdown call with all HTML to minimize parsing
    header_html = f'<div>{title_html}{caption_html}</div>{border_html}'
    st.markdown(header_html, unsafe_allow_html=True)


def sidebar_panel(openai_connected: bool = False, tools_status: dict = None, show_tools_status: bool = True):
    """Create a consistent sidebar panel"""
    with st.sidebar:
        st.markdown("### 🔑 Connection Status")
        
        if openai_connected:
            st.success("✅ OpenAI Connected")
        else:
            st.warning("⚠️ OpenAI Not Connected")
        
        # Only show tools status if explicitly requested
        if tools_status and show_tools_status:
            st.divider()
            st.markdown("### 🛠️ Tools Status")
            for tool_name, status in tools_status.items():
                if status:
                    st.markdown(f"✅ **{tool_name}**")
                else:
                    st.markdown(f"❌ **{tool_name}**")
        
        st.markdown("---")


def get_or_set_api_key(key_name: str, env_var: str = None, validate_func=None):
    """
    Get API key from session_state or environment variable.
    If not found, prompt user and persist to both session_state and environment.
    
    Args:
        key_name: Name of the key in session_state (e.g., "openai_key")
        env_var: Environment variable name (e.g., "OPENAI_API_KEY")
        validate_func: Optional function to validate key format
    
    Returns:
        The API key string or None if not set
    """
    # Initialize if not exists
    if key_name not in st.session_state:
        st.session_state[key_name] = ""
    
    # Check environment variable first
    if env_var and not st.session_state[key_name]:
        env_key = os.environ.get(env_var, "")
        if env_key:
            st.session_state[key_name] = env_key
            # Also set in environment for libraries that read from env
            os.environ[env_var] = env_key
    
    return st.session_state.get(key_name, "")


def set_api_key(key_name: str, value: str, env_var: str = None):
    """
    Set API key in both session_state and environment variable.
    
    Args:
        key_name: Name of the key in session_state
        value: The API key value
        env_var: Environment variable name
    """
    st.session_state[key_name] = value
    if env_var:
        os.environ[env_var] = value

