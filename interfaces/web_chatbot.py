"""
Streamlit Web Interface for Medical Chatbot
Interactive web UI for the medical chatbot.
"""

import streamlit as st
import sys
from pathlib import Path
import json
from datetime import datetime

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from medical_chatbot_engine import ConversationalChatbot

# Page config
st.set_page_config(
    page_title="Medical Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196F3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .stMetric {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    """Load the chatbot engine (cached)."""
    engine = ConversationalChatbot(model_path="/home/shivam/pikky/models/merged_model")
    if engine.load_model():
        return engine
    return None


def format_message(role: str, content: str) -> None:
    """Format and display a message."""
    css_class = "user-message" if role == "user" else "assistant-message"
    role_label = "👤 You" if role == "user" else "🤖 Chatbot"

    st.markdown(
        f"""
        <div class="chat-message {css_class}">
            <strong>{role_label}</strong><br>
            {content}
        </div>
        """,
        unsafe_allow_html=True,
    )


def export_conversation(history: list) -> str:
    """Export conversation as formatted text."""
    text_parts = [
        "=" * 80,
        "MEDICAL CHATBOT CONVERSATION",
        f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80,
        "",
    ]

    for i, msg in enumerate(history, 1):
        role = "QUESTION" if msg["role"] == "user" else "ANSWER"
        text_parts.append(f"[{i}] {role}")
        text_parts.append(msg["content"])
        text_parts.append("-" * 80)

    return "\n".join(text_parts)


def main():
    """Main Streamlit app."""

    # Title
    st.markdown("# 🏥 Medical Chatbot")
    st.markdown(
        "Fine-tuned Llama-3.2-1B for medical Q&A | "
        "[GitHub](https://github.com) | "
        "[Documentation](#)"
    )

    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        # Model info
        st.markdown("### Model Information")
        engine = load_model()

        if engine:
            model_info = engine.get_model_info()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Model", "Llama-3.2-1B")
                st.metric("Device", model_info.get("device", "Unknown"))
            with col2:
                st.metric("Status", "✓ Loaded")
                st.metric("Vocab Size", model_info.get("vocab_size", "Unknown"))
        else:
            st.error("❌ Model failed to load")
            st.stop()

        # Generation parameters
        st.markdown("### Generation Parameters")
        max_length = st.slider(
            "Max Response Length",
            min_value=50,
            max_value=512,
            value=256,
            step=50,
            help="Maximum number of tokens in response",
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher = more creative, Lower = more deterministic",
        )
        # top_p = st.slider(
        #     "Top P (Nucleus Sampling)",
        #     min_value=0.0,
        #     max_value=1.0,
        #     value=0.95,
        #     step=0.05,
        #     help="Probability threshold for token selection",
        # )

        # Conversation management
        st.markdown("### Conversation Management")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.messages = []
                st.success("History cleared!")

        with col2:
            if st.button("📊 Show Stats", use_container_width=True):
                st.session_state.show_stats = True

        with col3:
            if st.button("📥 Export", use_container_width=True):
                st.session_state.show_export = True

        # About
        st.markdown("---")
        st.markdown("### About")
        st.info(
            "This medical chatbot is fine-tuned on 193K medical Q&A samples "
            "using QLoRA (Quantized LoRA) for efficient adaptation. "
            "It provides medical information but should not replace professional medical advice."
        )

    # Main chat area
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display statistics if requested
    if st.session_state.get("show_stats", False):
        st.markdown("---")
        st.markdown("### 📊 Conversation Statistics")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Total Messages",
                len(st.session_state.messages),
            )
        with col2:
            user_msgs = sum(1 for m in st.session_state.messages if m["role"] == "user")
            st.metric("Questions", user_msgs)
        with col3:
            assistant_msgs = sum(
                1 for m in st.session_state.messages if m["role"] == "assistant"
            )
            st.metric("Answers", assistant_msgs)

        st.session_state.show_stats = False

    # Display export if requested
    if st.session_state.get("show_export", False):
        st.markdown("---")
        st.markdown("### 📥 Export Conversation")

        if st.session_state.messages:
            exported_text = export_conversation(st.session_state.messages)
            st.download_button(
                label="📥 Download as Text",
                data=exported_text,
                file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
            )

            exported_json = json.dumps(
                st.session_state.messages, indent=2, ensure_ascii=False
            )
            st.download_button(
                label="📥 Download as JSON",
                data=exported_json,
                file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )
        else:
            st.info("No conversation to export yet.")

        st.session_state.show_export = False

    # Display chat history
    st.markdown("---")
    st.markdown("### 💬 Conversation")

    if st.session_state.messages:
        for message in st.session_state.messages:
            format_message(message["role"], message["content"])
    else:
        st.info("👋 Start a conversation by asking a medical question below!")

    # User input
    st.markdown("---")
    user_input = st.text_area(
        "Ask a medical question:",
        placeholder="e.g., What are the symptoms of diabetes?",
        height=100,
        label_visibility="collapsed",
    )

    col1, col2 = st.columns([4, 1])

    with col1:
        if st.button("📤 Send Question", use_container_width=True, type="primary"):
            if user_input.strip():
                # Add user message
                st.session_state.messages.append(
                    {"role": "user", "content": user_input}
                )

                # Generate response
                with st.spinner("🤔 Thinking..."):
                    response = engine.chat(
                        user_input,
                        max_length=max_length,
                        temperature=temperature,
                    )

                # Add assistant message
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )

                st.rerun()
            else:
                st.warning("Please enter a question!")

    with col2:
        if st.button("🔄 Reload", use_container_width=True):
            st.rerun()

    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray; font-size: 0.8rem;'>"
        "💡 Disclaimer: This chatbot provides medical information for educational purposes. "
        "Always consult with healthcare professionals for medical advice."
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
