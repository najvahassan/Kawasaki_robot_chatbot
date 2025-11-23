# ----------------------------- Imports -----------------------------
import re
from sentence_transformers import SentenceTransformer, util
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
import os
from dotenv import load_dotenv

# ----------------------------- Environment -----------------------------
load_dotenv()  # loads the .env file

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# ----------------------------- Load Embedding Model -----------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------- Robot Metadata -----------------------------
robot_docs = [
    {"model": "RS007N", "text": "RS007N payload, reach, description, axes, repeatability, manuals"},
    {"model": "BX200L", "text": "BX200L payload, reach, description, axes, repeatability, manuals"},
    {"model": "ZD130S / ZDE130S", "text": "ZD130S payload, reach, description, axes, repeatability, manuals"},
    {"model": "MX350L", "text": "MX350L payload, reach, description, axes, repeatability, manuals"},
    # Add more robots here
]

# Precompute embeddings
robot_texts = [r["text"] for r in robot_docs]
robot_embeddings = embedding_model.encode(robot_texts)

# ----------------------------- Robot Normalization -----------------------------
robot_map = {
    r"\brs007n\b": "RS007N",
    r"\brs013n\b": "RS013N",
    r"\brs015x\b": "RS015X",
    r"\brs020n\b": "RS020N",
    r"\brs080n\b": "RS080N",
    r"\bbx100s\b": "BX100S",
    r"\bbx100n\b": "BX100N",
    r"\bbx200l\b": "BX200L",
    r"\bbx200x\b": "BX200X",
    r"\bzd130s\b": "ZD130S / ZDE130S",
    r"\bmx350l\b": "MX350L",
}

# ----------------------------- Semantic Retrieval -----------------------------
def find_relevant_robots(query: str, top_k=3):
    """
    Return the top_k relevant robot models for a query using embeddings similarity.
    """
    query_emb = embedding_model.encode(query)
    cos_scores = util.cos_sim(query_emb, robot_embeddings)
    top_idx = cos_scores.argsort(descending=True)[0][:top_k]  # tensor
    top_idx = top_idx.tolist()  # convert to Python list
    return [robot_docs[i]["model"] for i in top_idx]

# ----------------------------- Query Normalization -----------------------------
def expand_query(query: str) -> str:
    """
    Normalize robot names and add context for QA chain.
    """
    query_lower = query.lower()
    for pattern, full_name in robot_map.items():
        query_lower = re.sub(pattern, full_name.lower(), query_lower)
    return query_lower + " robot specifications"




# ----------------------------- Relevance Score -----------------------------
def relevance_score(query: str) -> float:
    """
    Simple heuristic to compute a relevance score based on presence of robot model.
    """
    query_lower = query.lower()
    for pattern in robot_map.keys():
        if re.search(pattern, query_lower):
            return 0.9  # High relevance if robot mentioned
    return 0.2  # Low relevance if no robot mentioned

# ----------------------------- Suggest Queries via ChatGroq -----------------------------
def suggest_queries(query: str, max_suggestions=2):
    # --- Step 1: Calculate relevance ---
    rel = relevance_score(query)

    # --- Step 2: Find robots using embeddings ---
    relevant_robots = find_relevant_robots(query)
    robots_str = ", ".join(relevant_robots)

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)

    # --- Case A: High relevance → generate deep technical suggestions ---
    if rel >= 0.7:
        prompt = f"""
User query: "{query}"
Relevant robots: {robots_str}

Generate {max_suggestions} extremely relevant follow-up questions.
Rules:
- They MUST relate to the robot(s) referenced.
- They MUST be technical (payload, reach, axes, manuals, repeatability, etc.)
- No generic questions.
- Output ONLY the questions, one per line.
"""
    
    # --- Case B: Medium relevance → general but robotics-related ---
    elif 0.3 <= rel < 0.7:
        prompt = f"""
User query: "{query}"

Generate {max_suggestions} robotics-related follow-up questions.
Rules:
- Broader robotics topics allowed.
- Avoid referencing specific robot models.
- Output ONLY the questions, one per line.
"""
    
    # --- Case C: Low relevance → suggest clarifying questions ---
    else:
        return [
            "Which Kawasaki robot model are you asking about?",
            "Do you want specifications, manuals, or part numbers?"
        ]

    messages = [
        SystemMessage(content="You generate highly relevant follow-up questions."),
        HumanMessage(content=prompt)
    ]

    response = llm.invoke(messages).content
    suggestions = [s.strip() for s in response.split("\n") if s.strip()]

    return suggestions[:3]
