# PDF Chat Assistant

An interactive, AI-powered PDF document analysis application with a beautiful 3D interface built using React, TypeScript, Three.js, and Tailwind CSS.

## Features

- **Interactive 3D Background**: Engaging particle system and animated torus created with Three.js
- **Drag & Drop Upload**: Easy PDF file upload with drag-and-drop support
- **Multiple Query Types**:
  - Q&A: Ask questions about your documents
  - Quiz: Generate multiple-choice questions
  - Summary: Get document summaries
- **Optimistic UI**: Smooth, responsive interface with optimistic updates
- **Session Management**: Automatic session handling with localStorage persistence
- **Responsive Design**: Mobile-first design that works on all devices
- **Real-time Chat**: Interactive chat interface with message history

## Prerequisites

- Node.js 18+
- Flask backend server running (see backend setup)

## Installation

1. Install dependencies:
```bash
npm install
```

2. Create a `.env` file:
```bash
cp .env.example .env
```

3. Update the `.env` file with your Flask API URL:
```
VITE_API_URL=http://localhost:5000/api
```

## Development

Run the development server:
```bash
npm run dev
```

The application will be available at `http://localhost:5173`

## Building for Production

```bash
npm run build
```

The production-ready files will be in the `dist` directory.

## Backend Setup

This frontend requires the Flask backend to be running. Make sure your Flask server is started and accessible at the URL specified in your `.env` file.

The backend should provide the following endpoints:
- `POST /api/session/create` - Create a new session
- `POST /api/upload` - Upload PDF files
- `POST /api/process` - Process uploaded documents
- `POST /api/chat` - Send chat queries
- `GET /api/session/:id` - Get session info
- `DELETE /api/session/:id` - Delete session

## Usage

1. **Upload Documents**: Drag and drop PDF files or click to browse
2. **Process**: Click "Process Documents" to create vector embeddings
3. **Chat**: Select query type (Q&A, Quiz, or Summary) and start asking questions

## Technology Stack

- **React** - UI library
- **TypeScript** - Type safety
- **Three.js** - 3D graphics and animations
- **Tailwind CSS** - Styling
- **Vite** - Build tool
- **Lucide React** - Icons

## Architecture

- `src/services/api.ts` - API service layer for Flask backend
- `src/context/SessionContext.tsx` - Session state management
- `src/components/ThreeBackground.tsx` - 3D animated background
- `src/components/FileUpload.tsx` - File upload interface
- `src/components/ChatInterface.tsx` - Chat UI
- `src/components/Header.tsx` - App header with session info

## License

MIT
