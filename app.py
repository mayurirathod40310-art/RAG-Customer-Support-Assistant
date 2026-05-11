import streamlit as st
from backend import get_response

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="RAG Customer Support Assistant",
    page_icon="🤖",
    layout="centered"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #1E3A8A;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 30px;
}

.stTextInput > div > div > input {
    border-radius: 10px;
    padding: 12px;
    border: 2px solid #d1d5db;
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
    background-color: #2563eb;
    color: white;
    font-size: 16px;
    font-weight: bold;
    padding: 10px;
}

.stButton > button:hover {
    background-color: #1d4ed8;
    color: white;
}

.response-box {
    background-color: #1e1e1e;
    color: white;
    padding: 20px;
    border-radius: 12px;
    border-left: 5px solid #2563eb;
    margin-top: 20px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.3);
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown(
    "<div class='title'>🤖 RAG Customer Support Assistant</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Powered by LangGraph + ChromaDB + HITL</div>",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "📄 Upload PDF Document",
    type=["pdf"]
)
# -----------------------------
# INPUT
# -----------------------------
query = st.text_input(
    "🔍 Enter your question"
)

# -----------------------------
# BUTTON
# -----------------------------
if st.button("Generate Response"):

    if query.strip() != "":

        with st.spinner("Generating response..."):

            response = get_response(query)

        st.markdown(
            f"""
            <div class='response-box'>
            <b>📄 Retrieved Response:</b><br><br>
            {response}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.warning("Please enter a question.")