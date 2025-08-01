
#---------------------------------------------
import streamlit as st
import uuid
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

def main():

    try:
        from graph import create_linkedin_optimization_graph
        BACKEND_AVAILABLE = True
    except ImportError as e:
        BACKEND_AVAILABLE = False
        st.error(f"Failed to import backend: {e}")

    st.set_page_config(page_title="LinkedIn Profile Optimizer", layout="wide")
    st.title("💼 AI LinkedIn Profile Optimizer")

    if "graph" not in st.session_state and BACKEND_AVAILABLE:
        with st.spinner("Initializing AI Engine... This may take a moment."):
            st.session_state.graph = create_linkedin_optimization_graph()
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.config = {"configurable": {"thread_id": st.session_state.thread_id}}
            st.session_state.messages = []
            st.success("AI Engine Initialized!")

    if "messages" in st.session_state:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if "messages" in st.session_state and not st.session_state.messages:
        welcome_message = "👋 Welcome! To begin, please paste your full **LinkedIn Profile** text."
        with st.chat_message("assistant"):
            st.markdown(welcome_message)
            st.session_state.messages.append({"role": "assistant", "content": welcome_message})

    if BACKEND_AVAILABLE:
        if prompt := st.chat_input("Paste your LinkedIn URL or ask a follow-up question..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("🤖 Thinking..."):
                    input_for_graph = {"messages": [HumanMessage(content=prompt)]}
                    final_response = "I'm sorry, I encountered an issue. Please try again."
                    for chunk in st.session_state.graph.stream(input_for_graph, config=st.session_state.config):
                        for key in chunk:
                            if "messages" in chunk[key]:
                                final_response = chunk[key]["messages"][-1].content
                    st.markdown(final_response)
                    st.session_state.messages.append({"role": "assistant", "content": final_response})
    else:
        st.error("Backend could not be loaded. Please check the terminal for errors.")


if __name__ == "__main__":
    main()
