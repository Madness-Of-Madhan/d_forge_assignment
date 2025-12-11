import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { ApiService } from '../services/api';

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  queryType?: 'qa' | 'quiz' | 'summary';
  timestamp: Date;
  optimistic?: boolean;
}

interface SessionContextType {
  sessionId: string | null;
  isProcessed: boolean;
  uploadedFiles: string[];
  messages: Message[];
  isLoading: boolean;
  error: string | null;
  createSession: () => Promise<void>;
  uploadFiles: (files: File[]) => Promise<void>;
  processDocuments: () => Promise<void>;
  sendMessage: (message: string, type?: 'qa' | 'quiz' | 'summary') => Promise<void>;
  clearError: () => void;
  resetSession: () => void;
}

const SessionContext = createContext<SessionContextType | undefined>(undefined);

export function SessionProvider({ children }: { children: ReactNode }) {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isProcessed, setIsProcessed] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<string[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const storedSessionId = localStorage.getItem('sessionId');
    if (storedSessionId) {
      setSessionId(storedSessionId);
      loadSessionInfo(storedSessionId);
    }
  }, []);

  const loadSessionInfo = async (id: string) => {
    try {
      const info = await ApiService.getSessionInfo(id);
      setUploadedFiles(info.files);
      setIsProcessed(info.processed);
    } catch {
      localStorage.removeItem('sessionId');
      setSessionId(null);
    }
  };

  const createSession = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const response = await ApiService.createSession();
      setSessionId(response.session_id);
      localStorage.setItem('sessionId', response.session_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create session');
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  const uploadFiles = async (files: File[]) => {
    if (!sessionId) {
      throw new Error('No active session');
    }

    try {
      setIsLoading(true);
      setError(null);
      const response = await ApiService.uploadFiles(sessionId, files);
      setUploadedFiles(response.files);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to upload files');
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  const processDocuments = async () => {
    if (!sessionId) {
      throw new Error('No active session');
    }

    try {
      setIsLoading(true);
      setError(null);
      await ApiService.processDocuments(sessionId);
      setIsProcessed(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to process documents');
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  const sendMessage = async (message: string, type: 'qa' | 'quiz' | 'summary' = 'qa') => {
    if (!sessionId || !isProcessed) {
      throw new Error('Please upload and process documents first');
    }

    const userMessage: Message = {
      id: `user-${Date.now()}`,
      type: 'user',
      content: message,
      timestamp: new Date(),
    };

    const optimisticAssistantMessage: Message = {
      id: `assistant-${Date.now()}`,
      type: 'assistant',
      content: '...',
      queryType: type,
      timestamp: new Date(),
      optimistic: true,
    };

    setMessages(prev => [...prev, userMessage, optimisticAssistantMessage]);

    try {
      setError(null);
      const response = await ApiService.chat(sessionId, message, type);

      setMessages(prev =>
        prev.map(msg =>
          msg.id === optimisticAssistantMessage.id
            ? {
                ...msg,
                content: response.answer,
                queryType: response.type as 'qa' | 'quiz' | 'summary',
                optimistic: false,
              }
            : msg
        )
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to send message');
      setMessages(prev => prev.filter(msg => msg.id !== optimisticAssistantMessage.id));
      throw err;
    }
  };

  const clearError = () => {
    setError(null);
  };

  const resetSession = () => {
    if (sessionId) {
      ApiService.deleteSession(sessionId).catch(console.error);
      localStorage.removeItem('sessionId');
    }
    setSessionId(null);
    setIsProcessed(false);
    setUploadedFiles([]);
    setMessages([]);
    setError(null);
  };

  return (
    <SessionContext.Provider
      value={{
        sessionId,
        isProcessed,
        uploadedFiles,
        messages,
        isLoading,
        error,
        createSession,
        uploadFiles,
        processDocuments,
        sendMessage,
        clearError,
        resetSession,
      }}
    >
      {children}
    </SessionContext.Provider>
  );
}

export function useSession() {
  const context = useContext(SessionContext);
  if (context === undefined) {
    throw new Error('useSession must be used within a SessionProvider');
  }
  return context;
}
