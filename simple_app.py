"""
Simple LinkedIn Profile Optimizer - Streamlit Chat Interface
"""
import streamlit as st
import uuid
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from graph import create_linkedin_optimization_graph
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="LinkedIn Profile Optimizer",
    page_icon="💼",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #0077B5;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #f0f2f6;
        border-left: 4px solid #0077B5;
    }
    .assistant-message {
        background-color: #ffffff;
        border-left: 4px solid #00A0B0;
        border: 1px solid #e0e0e0;
    }
    .status-success {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .status-info {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state for the app"""
    # Create unique thread ID for this session
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
        logger.info(f"Created new thread: {st.session_state.thread_id}")
    
    # Initialize the graph
    if "graph" not in st.session_state:
        try:
            st.session_state.graph = create_linkedin_optimization_graph()
            logger.info("LinkedIn optimization graph initialized successfully")
        except Exception as e:
            st.error(f"Failed to initialize graph: {str(e)}")
            st.session_state.graph = None
            return False
    
    # Configuration for LangGraph checkpointer
    if "config" not in st.session_state:
        st.session_state.config = {
            "configurable": {"thread_id": st.session_state.thread_id}
        }
    
    # Initialize messages list for display
    if "display_messages" not in st.session_state:
        st.session_state.display_messages = []
    
    # User profile tracking
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = {}
    
    return True


def display_chat_message(message, is_user=False):
    """Display a chat message with proper styling"""
    css_class = "user-message" if is_user else "assistant-message"
    sender = "You" if is_user else "AI Assistant"
    
    st.markdown(f"""
    <div class="chat-message {css_class}">
        <strong>{sender}:</strong><br>
        {message}
    </div>
    """, unsafe_allow_html=True)


def process_user_message(user_input: str):
    """Process user message through the LangGraph backend"""
    if not st.session_state.graph:
        st.error("Graph not initialized. Please refresh the page.")
        return
    
    try:
        # Add user message to display
        st.session_state.display_messages.append({"content": user_input, "is_user": True})
        
        # Prepare state for LangGraph
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "user_profile": st.session_state.user_profile,
            "linkedin_url": None,
            "job_role": None,
            "user_query": user_input,
            "profile_data": None,
            "profile_scraped": False,
            "analysis_results": None,
            "job_description": None,
            "job_fit_report": None,
            "rewritten_section": None,
            "next_action": None
        }
        
        # Show processing indicator
        with st.spinner("🤖 AI is processing your request..."):
            # Execute the graph
            result = st.session_state.graph.graph.invoke(
                initial_state,
                config=st.session_state.config
            )
            
            # Extract AI response from the messages
            ai_response = None
            if result.get("messages"):
                # Get the last AI message
                for msg in reversed(result["messages"]):
                    if isinstance(msg, AIMessage):
                        ai_response = msg.content
                        break
            
            # Fallback to other result fields if no AI message found
            if not ai_response:
                if result.get("analysis_results"):
                    ai_response = result["analysis_results"]
                elif result.get("job_fit_report"):
                    ai_response = result["job_fit_report"]
                elif result.get("rewritten_section"):
                    ai_response = result["rewritten_section"]
                else:
                    ai_response = "I've processed your request. How else can I help you optimize your LinkedIn profile?"
            
            # Add AI response to display
            if ai_response:
                st.session_state.display_messages.append({"content": ai_response, "is_user": False})
            
            # Update user profile if extracted
            if result.get("user_profile"):
                st.session_state.user_profile.update(result["user_profile"])
        
        # Force rerun to update display
        st.rerun()
        
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        st.error(f"An error occurred: {str(e)}")


def display_sidebar():
    """Display sidebar with app info and system status"""
    with st.sidebar:
        st.markdown("## 💼 LinkedIn Optimizer")
        
        st.markdown("### 🚀 Features")
        st.markdown("""
        - 📊 **Profile Analysis** - Get detailed feedback on your LinkedIn profile
        - 🎯 **Job Fit Analysis** - See how you match specific roles
        - ✍️ **Content Enhancement** - Improve your profile sections
        - 💬 **Interactive Chat** - Ask questions and get personalized advice
        """)
        
        st.markdown("---")
        
        # System status
        st.markdown("### ⚙️ System Status")
        
        # Check graph initialization
        if st.session_state.get("graph"):
            st.markdown('<div class="status-success">✅ Backend Connected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-error">❌ Backend Error</div>', unsafe_allow_html=True)
        
        # Check environment
        apify_status = "✅ Connected" if os.getenv("APIFY_API_TOKEN") else "⚠️ Not configured"
        st.markdown(f"**Apify:** {apify_status}")
        
        ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        st.markdown(f"**Model:** {ollama_model}")
        
        # Memory info
        if st.session_state.get("thread_id"):
            thread_short = st.session_state.thread_id[:8] + "..."
            st.markdown(f"**Session:** {thread_short}")
        
        # User profile info
        if st.session_state.user_profile:
            st.markdown("---")
            st.markdown("### 👤 User Profile")
            if st.session_state.user_profile.get("name"):
                st.markdown(f"**Name:** {st.session_state.user_profile['name']}")
            if st.session_state.user_profile.get("current_role"):
                st.markdown(f"**Role:** {st.session_state.user_profile['current_role']}")
            if st.session_state.user_profile.get("company"):
                st.markdown(f"**Company:** {st.session_state.user_profile['company']}")


def main():
    """Main application function"""
    st.markdown('<h1 class="main-header">💼 LinkedIn Profile Optimizer</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if not initialize_session_state():
        st.stop()
    
    # Display sidebar
    display_sidebar()
    
    # Main chat interface
    st.markdown("### 💬 Chat with Your LinkedIn Coach")
    
    # Display conversation history
    if st.session_state.display_messages:
        for message in st.session_state.display_messages:
            display_chat_message(message["content"], message["is_user"])
    else:
        # Welcome message
        st.markdown("""
        <div class="status-info">
        👋 <strong>Welcome!</strong> I'm your AI LinkedIn optimization assistant. 
        I can help you:
        <ul>
        <li>Analyze your LinkedIn profile (paste your profile text or URL)</li>
        <li>Compare your profile against specific job roles</li>
        <li>Rewrite and enhance profile sections</li>
        <li>Provide personalized career advice</li>
        </ul>
        Start by telling me about yourself or sharing your LinkedIn profile!
        </div>
        """, unsafe_allow_html=True)
    
    # Chat input
    user_input = st.chat_input("Type your message here... (e.g., 'Analyze my profile' or paste your LinkedIn profile text)")
    
    if user_input:
        process_user_message(user_input)


if __name__ == "__main__":
    main()
