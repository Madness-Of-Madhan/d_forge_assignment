// services/api.ts
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000/api';

export interface Session {
  session_id: string;
  files: string[];
  processed: boolean;
}

export interface ChatResponse {
  success: boolean;
  type: string;
  question: string;
  answer: string;
  error?: string;
}

export interface UploadResponse {
  success: boolean;
  message: string;
  files: string[];
  error?: string;
}

export class ApiService {
  static async createSession(): Promise<{ session_id: string }> {
    const response = await fetch(`${API_BASE_URL}/session/create`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error('Failed to create session');
    }

    const data = await response.json();
    return data;
  }

  static async uploadFiles(sessionId: string, files: File[]): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('session_id', sessionId);

    files.forEach(file => {
      formData.append('files', file);
    });

    const response = await fetch(`${API_BASE_URL}/upload`, {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || 'Failed to upload files');
    }

    return data;
  }

  static async processDocuments(sessionId: string): Promise<{ chunks_created: number }> {
    const response = await fetch(`${API_BASE_URL}/process`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ session_id: sessionId }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || 'Failed to process documents');
    }

    return data;
  }

  static async chat(
    sessionId: string,
    question: string,
    type: 'qa' | 'quiz' | 'summary' = 'qa',
    numQuestions?: number
  ): Promise<ChatResponse> {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        session_id: sessionId,
        question,
        type,
        num_questions: numQuestions,
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || 'Failed to get response');
    }

    return data;
  }

  static async getSessionInfo(sessionId: string): Promise<Session> {
    const response = await fetch(`${API_BASE_URL}/session/${sessionId}`);

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || 'Failed to get session info');
    }

    return data;
  }

  static async deleteSession(sessionId: string): Promise<void> {
    const response = await fetch(`${API_BASE_URL}/session/${sessionId}`, {
      method: 'DELETE',
    });

    if (!response.ok) {
      const data = await response.json();
      throw new Error(data.error || 'Failed to delete session');
    }
  }

  static async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      return response.ok;
    } catch {
      return false;
    }
  }
}