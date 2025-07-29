"""
LinkedIn Profile Optimizer - Fixed Streamlit App
"""
import streamlit as st
import uuid
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="LinkedIn Profile Optimizer",
    page_icon="💼",
    layout="wide"
)

def safe_import_backend():
    """Safely import backend components with error handling"""
    try:
        from langchain_core.messages import HumanMessage, AIMessage
        from graph import create_linkedin_optimization_graph
        return True, create_linkedin_optimization_graph, HumanMessage, AIMessage
    except Exception as e:
        st.error(f"Backend import failed: {e}")
        return False, None, None, None

def main():
    """Main application"""
    st.title("💼 LinkedIn Profile Optimizer")
    
    # Initialize session state
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "backend_loaded" not in st.session_state:
        st.session_state.backend_loaded = False
    
    # Load backend
    if not st.session_state.backend_loaded:
        with st.spinner("🔄 Loading backend..."):
            success, graph_creator, HumanMessage, AIMessage = safe_import_backend()
            
            if success:
                try:
                    st.session_state.graph = graph_creator()
                    st.session_state.config = {"configurable": {"thread_id": st.session_state.thread_id}}
                    st.session_state.HumanMessage = HumanMessage
                    st.session_state.AIMessage = AIMessage
                    st.session_state.backend_loaded = True
                    st.success("✅ Backend loaded successfully!")
                except Exception as e:
                    st.error(f"Backend initialization failed: {e}")
                    return
            else:
                st.error("❌ Could not load backend components")
                return
    
    # Display status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Backend", "✅ Ready" if st.session_state.backend_loaded else "❌ Error")
    with col2:
        st.metric("Session", st.session_state.thread_id[:8] + "...")
    with col3:
        st.metric("Messages", len(st.session_state.messages))
    
    # Chat interface
    st.subheader("💬 Chat with Your LinkedIn Coach")
    
    # Display messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # Welcome message
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.write("""
            👋 **Welcome to LinkedIn Profile Optimizer!**
            
            I can help you:
            - 📊 Analyze your LinkedIn profile
            - 🎯 Compare against job requirements  
            - ✍️ Enhance your profile content
            - 💼 Provide career advice
            
            Start by introducing yourself or sharing your LinkedIn profile!
            """)
    
    # Chat input
    if prompt := st.chat_input("Type your message..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # Process with backend
        if st.session_state.backend_loaded:
            with st.chat_message("assistant"):
                with st.spinner("🤖 Thinking..."):
                    try:
                        # Prepare state
                        initial_state = {
                            "messages": [st.session_state.HumanMessage(content=prompt)],
                            "user_profile": {},
                            "linkedin_url": None,
                            "job_role": None,
                            "user_query": prompt,
                            "profile_data": None,
                            "profile_scraped": False,
                            "analysis_results": None,
                            "job_description": None,
                            "job_fit_report": None,
                            "rewritten_section": None,
                            "next_action": None
                        }
                        
                        # Execute graph
                        result = st.session_state.graph.graph.invoke(
                            initial_state,
                            config=st.session_state.config
                        )
                        
                        # Extract response
                        ai_response = "I'm processing your request. How can I help you optimize your LinkedIn profile?"
                        
                        if result.get("messages"):
                            for msg in reversed(result["messages"]):
                                if isinstance(msg, st.session_state.AIMessage):
                                    ai_response = msg.content
                                    break
                        
                        st.write(ai_response)
                        st.session_state.messages.append({"role": "assistant", "content": ai_response})
                        
                    except Exception as e:
                        error_msg = f"Sorry, I encountered an error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            st.error("Backend not loaded. Please refresh the page.")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 💼 LinkedIn Optimizer")
        st.markdown("### 🚀 Features")
        st.markdown("""
        - Profile Analysis
        - Job Fit Assessment  
        - Content Enhancement
        - Career Advice
        """)
        
        st.markdown("---")
        st.markdown("### ⚙️ System Status")
        
        if st.session_state.backend_loaded:
            st.success("✅ All systems operational")
        else:
            st.error("❌ Backend loading...")
        
        apify_status = "✅" if os.getenv("APIFY_API_TOKEN") else "❌"
        st.markdown(f"Apify API: {apify_status}")
        
        if st.button("🔄 Reset Chat"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()
