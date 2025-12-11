from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def extract_text_from_pdf(pdf_path):
    """Extract text from a single PDF file."""
    text = ""
    try:
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    except Exception as e:
        raise Exception(f"Error reading PDF {pdf_path}: {str(e)}")
    return text


def process_pdfs(pdf_paths):
    """Extract text from multiple PDF files."""
    combined_text = ""
    for pdf_path in pdf_paths:
        text = extract_text_from_pdf(pdf_path)
        combined_text += text
    return combined_text


def get_text_chunks(text):
    """Split text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks