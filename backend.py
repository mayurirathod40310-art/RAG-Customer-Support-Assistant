from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from typing import TypedDict
from langgraph.graph import StateGraph, END


# -----------------------------
# LOAD PDF
# -----------------------------
loader = PyPDFLoader("Bank FAQ.pdf")
documents = loader.load()


# -----------------------------
# CHUNKING
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

texts = text_splitter.split_documents(documents)


# -----------------------------
# EMBEDDINGS + CHROMADB
# -----------------------------
embedding = HuggingFaceEmbeddings()

db = Chroma.from_documents(
    texts,
    embedding,
    persist_directory="chroma_db"
)


# -----------------------------
# RETRIEVER
# -----------------------------
retriever = db.as_retriever(search_kwargs={"k": 1})


# -----------------------------
# RAG FUNCTION
# -----------------------------
def rag_pipeline(query):

    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    if not context.strip():
        return "I could not find the answer in the documents."

    return f"Relevant Information Retrieved From Documents:\n\n{context}"

# -----------------------------
# LANGGRAPH STATE
# -----------------------------
class GraphState(TypedDict):
    query: str
    response: str


# -----------------------------
# PROCESS QUERY
# -----------------------------
def process_query(state):

    query = state["query"]

    response = rag_pipeline(query)

    return {
        "query": query,
        "response": response
    }


# -----------------------------
# HITL CHECK
# -----------------------------
def check_escalation(state):

    response = state["response"]

    if len(response) < 50:
        return "escalate"

    return "complete"


# -----------------------------
# HUMAN ESCALATION
# -----------------------------
def human_escalation(state):

    return {
        "query": state["query"],
        "response": "Query escalated to human support agent."
    }


# -----------------------------
# FINAL WORKFLOW
# -----------------------------
workflow = StateGraph(GraphState)

workflow.add_node("process", process_query)
workflow.add_node("human", human_escalation)

workflow.set_entry_point("process")

workflow.add_conditional_edges(
    "process",
    check_escalation,
    {
        "escalate": "human",
        "complete": END
    }
)

workflow.add_edge("human", END)

app = workflow.compile()


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def get_response(query):

    result = app.invoke({
        "query": query
    })

    return result["response"]