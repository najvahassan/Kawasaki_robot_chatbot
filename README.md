# Kawasaki Robot Chatbot

A **question-answering chatbot** for Kawasaki robots using **LangChain, FAISS, and ChatGroq**.  
It allows users to query specifications, manuals, and features of various Kawasaki robot models.

---

##  Features

- Conversational interface with Streamlit (ChatGPT-style UI)
- Semantic search using FAISS vector store
- Query expansion and suggestion generation with ChatGroq LLM
- Handles multiple Kawasaki robot models: BX200L, RS007N, ZD130S, MX350L, etc.
- Suggestions for follow-up queries
- Keeps sensitive keys like `GROQ_API_KEY` in a `.env` file (not committed to Git)

---

##  Project Structure

