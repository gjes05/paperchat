import streamlit as st
from rag_pipeline import PaperChatRAG


st.set_page_config(
    page_title="PaperChat",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

  /* Global */
  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0f14;
    color: #e8e6e0;
  }

  # /* Hide Streamlit branding */
  # #MainMenu, footer { visibility: hidden; }
  # header { visibility: visible; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #13151c;
    border-right: 1px solid #1e2130;
  }
  section[data-testid="stSidebar"] * {
    color: #e8e6e0 !important;
  }

  /* Logo / title */
  .logo {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #e8e6e0;
  }
  .logo span { color: #7c6af7; }

  .tagline {
    font-size: 0.78rem;
    color: #5a5f72;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: -6px;
    margin-bottom: 24px;
  }

  /* File uploader */
  [data-testid="stFileUploader"] {
    background: #1a1d27;
    border: 1.5px dashed #2a2d3e;
    border-radius: 12px;
    padding: 8px;
    transition: border-color 0.2s;
  }
  [data-testid="stFileUploader"]:hover {
    border-color: #7c6af7;
  }

  /* Buttons */
  .stButton > button {
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    background: #7c6af7;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    width: 100%;
    cursor: pointer;
    transition: background 0.2s, transform 0.1s;
    letter-spacing: 0.02em;
  }
  .stButton > button:hover {
    background: #6a58e0;
    transform: translateY(-1px);
  }
  .stButton > button:active { transform: translateY(0); }

  /* Chat messages */
  .msg-user {
    background: #1a1d27;
    border: 1px solid #2a2d3e;
    border-radius: 14px 14px 4px 14px;
    padding: 14px 18px;
    margin: 8px 0 8px 60px;
    font-size: 0.92rem;
    line-height: 1.6;
  }
  .msg-ai {
    background: #16192a;
    border: 1px solid #7c6af720;
    border-left: 3px solid #7c6af7;
    border-radius: 4px 14px 14px 14px;
    padding: 14px 18px;
    margin: 8px 60px 8px 0;
    font-size: 0.92rem;
    line-height: 1.6;
  }
  .msg-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 6px;
  }
  .msg-label.user { color: #5a5f72; }
  .msg-label.ai   { color: #7c6af7; }

  /* Source citations */
  .source-card {
    background: #13151c;
    border: 1px solid #1e2130;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.8rem;
    color: #7a7f94;
  }
  .source-card strong { color: #b0adc8; font-family: 'Syne', sans-serif; font-size: 0.75rem; }

  /* Stats pills */
  .stat-pill {
    display: inline-block;
    background: #1a1d27;
    border: 1px solid #2a2d3e;
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 0.75rem;
    color: #7a7f94;
    margin: 3px 3px 3px 0;
  }
  .stat-pill b { color: #7c6af7; }

  /* Chat input */
  .stChatInput textarea, [data-testid="stChatInput"] textarea {
    background: #1a1d27 !important;
    border: 1px solid #2a2d3e !important;
    border-radius: 12px !important;
    color: #e8e6e0 !important;
    font-family: 'DM Sans', sans-serif !important;
  }
  .stChatInput textarea:focus, [data-testid="stChatInput"] textarea:focus {
    border-color: #7c6af7 !important;
  }

  /* Spinner */
  .stSpinner > div { border-top-color: #7c6af7 !important; }

  /* Divider */
  hr { border-color: #1e2130; }

  /* Alert / info boxes */
  .stAlert { background: #1a1d27; border-color: #2a2d3e; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


def init_state():
    defaults = {
        "rag":          None,
        "chat_history": [],   # list of {"role": "user"|"ai", "content": str, "sources": list}
        "ingested":     False,
        "file_stats":   {}
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# Sidebar

with st.sidebar:
    st.markdown('<div class="logo">Paper<span>Chat</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="tagline">RAG-Powered Document Q&A</div>', unsafe_allow_html=True)

    st.markdown("### Upload Documents")
    uploaded_files = st.file_uploader(
        "Drop your PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        if st.button("⚡ Ingest Documents"):
            with st.spinner("Processing & embedding documents..."):
                try:
                    if st.session_state.rag is None:
                        st.session_state.rag = PaperChatRAG()

                    stats = st.session_state.rag.ingest_documents(uploaded_files)
                    st.session_state.ingested   = True
                    st.session_state.file_stats = stats
                    st.session_state.chat_history = []  # reset chat on new upload
                    st.success("Documents ready!")
                except Exception as e:
                    st.error(f"Error: {e}")

    # Show ingestion stats
    if st.session_state.ingested and st.session_state.file_stats:
        st.markdown("---")
        st.markdown("### 📄 Ingested Files")
        stats = st.session_state.file_stats
        st.markdown(f'<span class="stat-pill">Total chunks: <b>{stats["total_chunks"]}</b></span>', unsafe_allow_html=True)
        for fname, info in stats["files"].items():
            with st.expander(f"📄 {fname}"):
                st.markdown(f'<span class="stat-pill">Pages: <b>{info["pages"]}</b></span> <span class="stat-pill">Chunks: <b>{info["chunks"]}</b></span>', unsafe_allow_html=True)

    # Reset conversation
    if st.session_state.ingested:
        st.markdown("---")
        if st.button("🔄 Reset Conversation"):
            st.session_state.rag.reset()
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.72rem; color:#3a3f52; line-height:1.7;">
    <b style="color:#5a5f72">Stack</b><br>
    LangChain · Gemini 2.5 Flash<br>
    Pinecone · Streamlit<br><br>
    <b style="color:#5a5f72">How it works</b><br>
    1. Upload PDFs<br>
    2. Chunks are embedded & stored in Pinecone<br>
    3. Your question retrieves top-K relevant chunks<br>
    4. Gemini synthesizes a grounded answer
    </div>
    """, unsafe_allow_html=True)


# Main Area

if not st.session_state.ingested:
    # Empty state
    st.markdown("""
    <div style="text-align:center; padding: 80px 20px;">
      <div style="font-family:'Syne',sans-serif; font-size:3.5rem; font-weight:800; letter-spacing:-0.04em; color:#e8e6e0;">
        Ask anything about<br><span style="color:#7c6af7;">your documents.</span>
      </div>
      <div style="font-size:1rem; color:#5a5f72; margin-top:16px; font-weight:300;">
        Upload PDFs in the sidebar to get started.
      </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # Chat history display
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("""
            <div style="text-align:center; padding:40px; color:#3a3f52; font-size:0.9rem;">
                Documents ingested ✓ — Ask your first question below
            </div>
            """, unsafe_allow_html=True)

        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="msg-user">
                  <div class="msg-label user">You</div>
                  {msg["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="msg-ai">
                  <div class="msg-label ai">PaperChat</div>
                  {msg["content"]}
                </div>
                """, unsafe_allow_html=True)

                # Show sources
                if msg.get("sources"):
                    with st.expander(f"📎 {len(msg['sources'])} source(s) cited"):
                        for s in msg["sources"]:
                            st.markdown(f"""
                            <div class="source-card">
                              <strong>📄 {s['file']} — Page {int(s['page']) + 1}</strong><br>
                              {s['snippet']}…
                            </div>
                            """, unsafe_allow_html=True)

    # Chat input
    question = st.chat_input("Ask a question about your documents...")

    if question:
        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.spinner("Thinking..."):
            try:
                result = st.session_state.rag.query(question)
                st.session_state.chat_history.append({
                    "role": "ai",
                    "content": result["answer"],
                    "sources": result["sources"]
                })
            except Exception as e:
                st.session_state.chat_history.append({
                    "role": "ai",
                    "content": f"⚠️ Error: {e}",
                    "sources": []
                })

        st.rerun()