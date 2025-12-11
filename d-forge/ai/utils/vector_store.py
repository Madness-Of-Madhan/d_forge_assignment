from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from utils.pdf_processor import get_text_chunks
import os

# Use free local embeddings - NO API KEY NEEDED!
def get_embeddings():
    """Get HuggingFace embeddings model (runs locally, completely free)."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


def create_vector_store(text, index_path):
    """Create FAISS vector store from text using free local embeddings."""
    try:
        print("📄 Splitting text into chunks...")
        # Split text into chunks
        chunks = get_text_chunks(text)
        print(f"✓ Created {len(chunks)} chunks")
        
        if not chunks:
            raise ValueError("No text chunks created from PDF")
        
        print("🤖 Loading embeddings model (first time may take 30 seconds)...")
        # Create embeddings (downloads model on first run only)
        embeddings = get_embeddings()
        
        print("💾 Creating vector store...")
        # Create and save vector store
        vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
        vectorstore.save_local(index_path)
        
        print(f"✅ Vector store created successfully with {len(chunks)} chunks!")
        return len(chunks)
        
    except Exception as e:
        print(f"❌ Error creating vector store: {str(e)}")
        raise Exception(f"Error creating vector store: {str(e)}")


def query_vector_store(index_path, query, k=5):
    """Query the FAISS vector store using free local embeddings."""
    try:
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Vector store not found at {index_path}")
        
        print(f"🔍 Loading vector store from {index_path}...")
        
        # Use same embeddings model
        embeddings = get_embeddings()
        
        # Load vector store
        vectorstore = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        print(f"🔎 Searching for top {k} relevant documents...")
        docs = vectorstore.similarity_search(query, k=k)
        
        print(f"✅ Found {len(docs)} relevant documents")
        return docs
        
    except Exception as e:
        print(f"❌ Error querying vector store: {str(e)}")
        raise Exception(f"Error querying vector store: {str(e)}")