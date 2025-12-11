import { useEffect, useState } from 'react';
import { SessionProvider, useSession } from './context/SessionContext';
import ThreeBackground from './components/ThreeBackground';
import Header from './components/Header';
import FileUpload from './components/FileUpload';
import ChatInterface from './components/ChatInterface';
import { Loader2 } from 'lucide-react';

function AppContent() {
  const { sessionId, createSession, isProcessed } = useSession();
  const [initializing, setInitializing] = useState(true);

  useEffect(() => {
    const initialize = async () => {
      if (!sessionId) {
        try {
          await createSession();
        } catch (error) {
          console.error('Failed to create session:', error);
        }
      }
      setInitializing(false);
    };

    initialize();
  }, []);

  if (initializing) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center space-y-4">
          <Loader2 className="w-12 h-12 text-blue-500 animate-spin mx-auto" />
          <p className="text-slate-400">Initializing session...</p>
        </div>
      </div>
    );
  }

  return (
    <>
      <ThreeBackground />
      <div className="relative min-h-screen flex flex-col">
        <Header />

        <main className="flex-1 container mx-auto p-4 md:p-6">
          {!isProcessed ? (
            <div className="flex items-center justify-center min-h-[calc(100vh-200px)]">
              <FileUpload />
            </div>
          ) : (
            <div className="h-[calc(100vh-140px)] bg-slate-900/40 backdrop-blur rounded-2xl border border-slate-800 overflow-hidden">
              <ChatInterface />
            </div>
          )}
        </main>
      </div>
    </>
  );
}

function App() {
  return (
    <SessionProvider>
      <AppContent />
    </SessionProvider>
  );
}

export default App;
