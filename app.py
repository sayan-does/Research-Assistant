import streamlit as st
import tempfile
import os
from rag_utils import extract_text_from_pdf, chunk_text, EmbeddingModel, VectorDB, process_pdf_to_faiss
from llm_utils import OllamaClient, build_rag_prompt, DEFAULT_MODEL

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = DEFAULT_MODEL
    # Remove temperature from session state if present
    if "temperature" in st.session_state:
        del st.session_state["temperature"]

def main():
    st.set_page_config(
        page_title="Ollama Chat Client",
        page_icon="🤖",
        layout="wide"
    )
    
    initialize_session_state()
    
    # Initialize Ollama client
    ollama_client = OllamaClient()

    # --- RAG Pipeline State ---
    if "vectordb" not in st.session_state:
        st.session_state.vectordb = None
    if "embed_model" not in st.session_state:
        st.session_state.embed_model = None
    if "uploaded_chunks" not in st.session_state:
        st.session_state.uploaded_chunks = []
    
    # Sidebar configuration
    with st.sidebar:
        st.title("🤖 Ollama Chat")
        
        # Check Ollama connection
        if ollama_client.is_ollama_running():
            st.success("✅ Ollama server connected")
        else:
            st.error("❌ Ollama server not running")
            st.info("Make sure Ollama is running with: `ollama serve`")
            return
        
        # Model selection
        available_models = ollama_client.get_available_models()
        if available_models:
            selected_model = st.selectbox(
                "Select Model",
                available_models,
                index=0 if DEFAULT_MODEL not in available_models else available_models.index(DEFAULT_MODEL)
            )
            st.session_state.selected_model = selected_model
        else:
            st.warning("No models available. Pull a model first:")
            st.code("ollama pull qwen3:1.7b")
            return
        
        # No temperature slider
        
        # Clear chat button
        if st.button("Clear Chat", type="secondary"):
            st.session_state.messages = []
            st.rerun()
        
        # --- Multi-PDF Upload & RAG Setup ---
        st.divider()
        st.subheader("📄 Upload up to 5 Research Papers (PDF)")
        uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True, key="multi_pdf")
        if uploaded_files:
            if len(uploaded_files) > 5:
                st.warning("You can upload up to 5 PDFs only.")
            else:
                if st.button("Process & Index PDFs"):
                    st.session_state.embed_model = EmbeddingModel()
                    st.session_state.vectordb = VectorDB(dim=384)
                    all_chunks = []
                    for uploaded_file in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            tmp_path = tmp_file.name
                        chunks = process_pdf_to_faiss(tmp_path, st.session_state.embed_model, st.session_state.vectordb)
                        all_chunks.extend(chunks)
                        os.unlink(tmp_path)
                    st.session_state.uploaded_chunks = all_chunks
                    st.success(f"Processed and indexed {len(all_chunks)} chunks from {len(uploaded_files)} PDFs.")
        if st.session_state.uploaded_chunks:
            st.info(f"{len(st.session_state.uploaded_chunks)} chunks available for retrieval from {len(uploaded_files) if uploaded_files else 0} PDFs.")
    
    # Main chat interface (ChatGPT-like)
    st.title("Personal Research Assistant (RAG)")

    # Suggest common research questions after PDF upload
    if st.session_state.uploaded_chunks:
        st.markdown("**💡 Example questions you can ask:**")
        st.markdown("- What is the main contribution of these papers?")
        st.markdown("- Summarize the key findings.")
        st.markdown("- What methods or algorithms are proposed?")
        st.markdown("- List the key concepts discussed.")
        st.markdown("- What datasets or experiments are used?")
        st.markdown("- What are the limitations or future work mentioned?")
        st.markdown("- Provide a citation for the main paper.")

    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt := st.chat_input("Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Retrieval-augmented context (if PDFs uploaded)
        context = ""
        if st.session_state.uploaded_chunks:
            # Retrieve top 5 relevant chunks
            query_emb = st.session_state.embed_model.encode([prompt])[0]
            top_chunks = st.session_state.vectordb.search(query_emb, top_k=5)
            context = "\n".join(top_chunks)

        # Compose prompt for LLM using system prompt
        llm_prompt = build_rag_prompt(prompt, context)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for chunk in ollama_client.chat_stream(
                model=st.session_state.selected_model,
                messages=[{"role": "user", "content": llm_prompt}]
            ):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()