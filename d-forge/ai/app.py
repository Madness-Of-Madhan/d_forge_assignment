from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import uuid
from dotenv import load_dotenv
import google.generativeai as genai
from utils.pdf_processor import process_pdfs
from utils.vector_store import create_vector_store, query_vector_store
from utils.chains import get_conversational_chain, get_quiz_chain, get_summary_chain

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Initialize Flask app
app = Flask(__name__)

# CRITICAL FIX: Properly configure CORS with explicit settings
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000", "http://127.0.0.1:3000"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Type"],
        "supports_credentials": True,
        "max_age": 3600
    }
})

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['FAISS_FOLDER'] = 'faiss_indexes'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['FAISS_FOLDER'], exist_ok=True)

# Store session data (in production, use Redis or database)
sessions = {}


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'message': 'PDF Chat API is running'
    }), 200


@app.route('/api/session/create', methods=['POST', 'OPTIONS'])
def create_session():
    """Create a new chat session."""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            'files': [],
            'processed': False,
            'index_path': None
        }
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Session created successfully'
        }), 201
    except Exception as e:
        print(f"Error creating session: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/upload', methods=['POST', 'OPTIONS'])
def upload_files():
    """Upload PDF files for processing."""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        # Get session ID
        session_id = request.form.get('session_id')
        if not session_id or session_id not in sessions:
            return jsonify({
                'success': False,
                'error': 'Invalid or missing session_id'
            }), 400

        # Check if files are present
        if 'files' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No files provided'
            }), 400

        files = request.files.getlist('files')
        
        if not files or files[0].filename == '':
            return jsonify({
                'success': False,
                'error': 'No files selected'
            }), 400

        uploaded_files = []
        
        # Create session folder
        session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        os.makedirs(session_folder, exist_ok=True)

        # Save uploaded files
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(session_folder, filename)
                file.save(filepath)
                uploaded_files.append(filepath)
            else:
                return jsonify({
                    'success': False,
                    'error': f'Invalid file type: {file.filename}. Only PDF files are allowed.'
                }), 400

        # Update session
        sessions[session_id]['files'] = uploaded_files

        return jsonify({
            'success': True,
            'message': f'{len(uploaded_files)} file(s) uploaded successfully',
            'files': [os.path.basename(f) for f in uploaded_files]
        }), 200

    except Exception as e:
        print(f"Upload error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/process', methods=['POST', 'OPTIONS'])
def process_documents():
    """Process uploaded PDF documents and create vector store."""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        session_id = data.get('session_id')

        if not session_id or session_id not in sessions:
            return jsonify({
                'success': False,
                'error': 'Invalid or missing session_id'
            }), 400

        session = sessions[session_id]
        
        if not session['files']:
            return jsonify({
                'success': False,
                'error': 'No files uploaded for this session'
            }), 400

        # Process PDFs
        print(f"Processing {len(session['files'])} files...")
        raw_text = process_pdfs(session['files'])
        
        if not raw_text.strip():
            return jsonify({
                'success': False,
                'error': 'No text could be extracted from PDFs'
            }), 400

        # Create vector store
        print("Creating vector store...")
        index_path = os.path.join(app.config['FAISS_FOLDER'], session_id)
        num_chunks = create_vector_store(raw_text, index_path)

        # Update session
        session['processed'] = True
        session['index_path'] = index_path

        print(f"Processing complete. Created {num_chunks} chunks.")
        return jsonify({
            'success': True,
            'message': 'Documents processed successfully',
            'chunks_created': num_chunks
        }), 200

    except Exception as e:
        print(f"Processing error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat():
    """Handle chat queries - Q&A, Quiz, or Summary."""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        question = data.get('question')
        query_type = data.get('type', 'qa')

        # Validation
        if not session_id or session_id not in sessions:
            return jsonify({
                'success': False,
                'error': 'Invalid or missing session_id'
            }), 400

        if not question:
            return jsonify({
                'success': False,
                'error': 'Question is required'
            }), 400

        session = sessions[session_id]
        
        if not session['processed']:
            return jsonify({
                'success': False,
                'error': 'Please process documents first'
            }), 400

        # Perform similarity search
        print(f"🔍 Searching for: {question}")
        docs = query_vector_store(session['index_path'], question, k=5)

        if not docs:
            return jsonify({
                'success': False,
                'error': 'No relevant documents found'
            }), 404

        # Auto-detect query type from question
        question_lower = question.lower()
        if any(keyword in question_lower for keyword in ['quiz', 'mcq', 'test', 'questions']):
            query_type = 'quiz'
        elif any(keyword in question_lower for keyword in ['summary', 'summarize', 'overview']):
            query_type = 'summary'

        # Import the retry function
        from utils.chains import call_chain_with_retry

        # Generate response based on query type with retry logic
        print(f"🤖 Generating {query_type} response...")
        
        if query_type == 'quiz':
            num_questions = data.get('num_questions', 5)
            chain = get_quiz_chain()
            response = call_chain_with_retry(
                chain,
                {
                    "input_documents": docs,
                    "num_questions": num_questions
                }
            )
        elif query_type == 'summary':
            chain = get_summary_chain()
            response = call_chain_with_retry(
                chain,
                {
                    "input_documents": docs,
                    "question": question
                }
            )
        else:  # qa
            chain = get_conversational_chain()
            response = call_chain_with_retry(
                chain,
                {
                    "input_documents": docs,
                    "question": question
                }
            )

        print("✅ Response generated successfully")
        return jsonify({
            'success': True,
            'type': query_type,
            'question': question,
            'answer': response['output_text']
        }), 200

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Chat error: {error_msg}")
        
        # Check if it's a quota error
        if '429' in error_msg or 'quota' in error_msg.lower():
            return jsonify({
                'success': False,
                'error': 'API quota exceeded. Please wait a few minutes and try again, or use a different API key.'
            }), 429
        
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500

    except Exception as e:
        print(f"Chat error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/session/<session_id>', methods=['GET', 'OPTIONS'])
def get_session_info(session_id):
    """Get information about a session."""
    if request.method == 'OPTIONS':
        return '', 200
    
    if session_id not in sessions:
        return jsonify({
            'success': False,
            'error': 'Session not found'
        }), 404

    session = sessions[session_id]
    return jsonify({
        'success': True,
        'session_id': session_id,
        'files': [os.path.basename(f) for f in session['files']],
        'processed': session['processed']
    }), 200


@app.route('/api/session/<session_id>', methods=['DELETE', 'OPTIONS'])
def delete_session(session_id):
    """Delete a session and its associated files."""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        if session_id not in sessions:
            return jsonify({
                'success': False,
                'error': 'Session not found'
            }), 404

        session = sessions[session_id]

        # Delete uploaded files
        session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        if os.path.exists(session_folder):
            import shutil
            shutil.rmtree(session_folder)

        # Delete FAISS index
        if session['index_path'] and os.path.exists(session['index_path']):
            import shutil
            shutil.rmtree(session['index_path'])

        # Remove from sessions
        del sessions[session_id]

        return jsonify({
            'success': True,
            'message': 'Session deleted successfully'
        }), 200

    except Exception as e:
        print(f"Delete session error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 50MB.'
    }), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


if __name__ == '__main__':
    print("=" * 50)
    print("Starting PDF Chat API Server")
    print("=" * 50)
    print("API Base URL: http://127.0.0.1:5000/api")
    print("Health Check: http://127.0.0.1:5000/api/health")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)