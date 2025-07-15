import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Path where documents are stored
DOCS_PATH = "data"
DB_FAISS_PATH = "vectorstore/db_faiss"

# Load documents
documents = []
for filename in os.listdir(DOCS_PATH):
    filepath = os.path.join(DOCS_PATH, filename)
    if filename.endswith(".txt"):
        loader = TextLoader(filepath)
    elif filename.endswith(".pdf"):
        loader = PyPDFLoader(filepath)
    else:
        continue
    documents.extend(loader.load())

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build FAISS vectorstore
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)


db.save_local(DB_FAISS_PATH)

print(f"✅ FAISS vectorstore rebuilt and saved at: {DB_FAISS_PATH}")
