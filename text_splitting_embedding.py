import pandas as pd
import fitz  # PyMuPDF
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

# --------------------------
# Paths
# --------------------------
csv_path = r"d:\Projects\ai_projects\Kawasaki_chatbot\kawasaki_robot_data.csv"
pdf_dir = r"d:\Projects\ai_projects\Kawasaki_chatbot\kawasaki_manuals"
vectorstore_dir = r"d:\Projects\ai_projects\Kawasaki_chatbot\vectorstore"

# --------------------------
# Load CSV metadata
# --------------------------
df = pd.read_csv(csv_path)
df["Robot Model"] = df["Robot Model"].str.upper()

# --------------------------
# Load PDFs
# --------------------------
pdfs = list(Path(pdf_dir).glob("*.pdf"))
docs = []

# --------------------------
# Initialize text splitter and embeddings
# --------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --------------------------
# Process PDFs
# --------------------------
for pdf in pdfs:
    # Extract robot model from filename
    model = pdf.stem.upper().split('_')[0]

    # Lookup metadata
    meta = df[df["Robot Model"].str.contains(model)]
    if not meta.empty:
        md = meta.iloc[0].to_dict()
    else:
        md = {"Robot Model": model}

    # Read PDF text
    doc = fitz.open(pdf)
    text = "".join([page.get_text() for page in doc])
    doc.close()

    # Split text into chunks
    chunks = splitter.split_text(text)
    for i, ch in enumerate(chunks):
        md2 = md.copy()
        md2.update({"source": pdf.name, "chunk": i})
        docs.append(Document(page_content=ch, metadata=md2))

# --------------------------
# Create FAISS vector store
# --------------------------
vs = FAISS.from_documents(docs, embeddings)
vs.save_local(vectorstore_dir)

print(f"Indexed {len(docs)} chunks → {vectorstore_dir}")
