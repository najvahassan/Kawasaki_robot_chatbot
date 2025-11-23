# Kawasaki Robot Chatbot

A **question-answering chatbot** for Kawasaki robots using **LangChain, FAISS, and ChatGroq**.  
It allows users to query specifications, manuals, and features of various Kawasaki robot models.

---

##  Features

- Conversational interface with Streamlit 
- Semantic search using FAISS vector store
- Query expansion and suggestion generation with ChatGroq LLM
- Handles multiple Kawasaki robot models: BX200L, RS007N, ZD130S, MX350L, etc.
- Suggestions for follow-up queries
- Keeps sensitive keys like `GROQ_API_KEY` in a `.env` file (not committed to Git)

---

##  Project Structure


Kawasaki_chatbot/
│
├─ backend_kawasaki.py # Handles vectorstore loading and QA chain
├─ query_intelligence.py # Query expansion and suggestion generation
├─ streamlit_ui.py #  UI for user interaction
|- text_splitting_embedding.py #  for converting text to embeddings
|- data_csv_file.py  #  creating the csv file to download data
|- download_dta.py  # for downloading the data
├─ vectorstore/ # FAISS vector store (tracked in Git)
├─ README.md # Project documentation
├─ requirements.txt # Python dependencies
|- .env #consists of GROQ-API key,not tracked in git
|- kawasaki_manual  #  consists of the data downloaded



---

##  Setup

1. **Clone the repository:**


git clone https://github.com/<najvahassan>/Kawasaki_robot_chatbot.git
cd Kawasaki_robot_chatbot

2. Install dependencies:

pip install -r requirements.txt


3.Create .env file:

GROQ_API_KEY=your_groq_api_key_here


4.Run the Streamlit UI:

streamlit run ir_app.py

