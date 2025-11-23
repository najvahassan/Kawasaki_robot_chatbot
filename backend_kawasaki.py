# ----------------------------- Imports -----------------------------
import os
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
# Import your query intelligence functions
from query_intelligence import expand_query, suggest_queries, relevance_score

load_dotenv()  # loads the .env file

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


VECTORSTORE_DIR = "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ----------------------------- Embeddings -----------------------------
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# ----------------------------- Load Vectorstore -----------------------------
def load_vectorstore():
    """
    Load the FAISS vectorstore with embeddings.
    """
    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore, embeddings

# ----------------------------- Create QA Chain -----------------------------
def create_qa_chain(vectorstore):
    """
    Create a conversational retrieval chain using ChatGroq and the FAISS retriever.
    """
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True, 
        output_key="answer"
    )

    PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a Kawasaki robot assistant. Answer the question using the provided context. 
If the answer is not in the context, say "I don't have the information".

Context:
{context}

Question: {question}
Answer:
"""
    )

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.0
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True,
        output_key="answer"
    )

    return qa_chain, memory

# ----------------------------- Handle User Query -----------------------------
def handle_query(user_query: str, qa_chain, debug=False):
    """
    Handles a query through the QA chain and query intelligence.
    Returns:
        response (str): LLM answer
        suggestions (list): Related suggested queries
        relevance (float): Relevance score
        sources (list): Retrieved documents from FAISS
    """
    # ----------------- Step 1: Retrieve relevant documents -----------------
    result = qa_chain({"question": user_query})
    response = result["answer"]
    sources = result.get("source_documents", [])

    if debug:
        print("=== Retrieved documents ===")
        if not sources:
            print("No documents retrieved from FAISS.")
        for i, doc in enumerate(sources):
            print(f"Doc {i} ({doc.metadata.get('source', 'unknown')}):")
            print(doc.page_content[:500], "\n---")

    # ----------------- Step 2: Query intelligence -----------------
    # Suggestions & relevance
    suggestions = suggest_queries(user_query)
    relevance = relevance_score(user_query)

    return response, suggestions, relevance, sources


