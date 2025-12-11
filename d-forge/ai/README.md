# PDF Chat API with Google Gemini

A Flask REST API for chatting with PDF documents using Google Gemini AI.

## Features

- Upload multiple PDF files
- Extract and process text from PDFs
- Create vector embeddings using FAISS
- Q&A conversations with documents
- Generate quiz questions from content
- Summarize documents
- Session management

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create `.env` file with your Google API key:
```
GOOGLE_API_KEY=your_key_here
```

3. Run the application:
```bash
python app.py
```

## API Endpoints

### 1. Health Check
```
GET /api/health
```

### 2. Create Session
```
POST /api/session/create
```

### 3. Upload Files
```
POST /api/upload
Content-Type: multipart/form-data

Form Data:
- session_id: string
- files: file[] (PDF files)
```

### 4. Process Documents
```
POST /api/process
Content-Type: application/json

Body:
{
  "session_id": "uuid"
}
```

### 5. Chat/Query
```
POST /api/chat
Content-Type: application/json

Body:
{
  "session_id": "uuid",
  "question": "What is the main topic?",
  "type": "qa",  // Options: qa, quiz, summary
  "num_questions": 5  // Optional, for quiz type
}
```

### 6. Get Session Info
```
GET /api/session/{session_id}
```

### 7. Delete Session
```
DELETE /api/session/{session_id}
```

## Usage Example
```python
import requests

BASE_URL = "http://localhost:5000/api"

# 1. Create session
response = requests.post(f"{BASE_URL}/session/create")
session_id = response.json()['session_id']

# 2. Upload files
files = [('files', open('document.pdf', 'rb'))]
data = {'session_id': session_id}
requests.post(f"{BASE_URL}/upload", files=files, data=data)

# 3. Process documents
requests.post(f"{BASE_URL}/process", json={'session_id': session_id})

# 4. Ask question
response = requests.post(f"{BASE_URL}/chat", json={
    'session_id': session_id,
    'question': 'What is the main topic?',
    'type': 'qa'
})
print(response.json()['answer'])

# 5. Generate quiz
response = requests.post(f"{BASE_URL}/chat", json={
    'session_id': session_id,
    'question': 'Generate quiz questions',
    'type': 'quiz',
    'num_questions': 5
})
print(response.json()['answer'])
```

## Production Deployment

For production, use Gunicorn:
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Notes

- Maximum file size: 50MB
- Supported formats: PDF only
- Sessions are stored in memory (use Redis/DB for production)