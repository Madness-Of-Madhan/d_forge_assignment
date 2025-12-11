import { FileText, RefreshCw, AlertCircle } from 'lucide-react';
import { useSession } from '../context/SessionContext';

export default function Header() {
  const { sessionId, uploadedFiles, isProcessed, error, clearError, resetSession } = useSession();

  return (
    <header className="bg-slate-900/80 backdrop-blur border-b border-slate-800">
      <div className="container mx-auto px-4 py-4">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between space-y-3 md:space-y-0">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-blue-600 rounded-lg">
              <FileText className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl md:text-2xl font-bold text-slate-100">
                PDF Chat Assistant
              </h1>
              <p className="text-xs md:text-sm text-slate-400">
                AI-powered document analysis and Q&A
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            {sessionId && (
              <>
                <div className="hidden md:flex items-center space-x-4 text-sm">
                  <div className="flex items-center space-x-2">
                    <div
                      className={`w-2 h-2 rounded-full ${
                        isProcessed ? 'bg-green-400' : 'bg-yellow-400'
                      }`}
                    />
                    <span className="text-slate-300">
                      {uploadedFiles.length} document{uploadedFiles.length !== 1 ? 's' : ''}
                    </span>
                  </div>
                  <span className="text-slate-500">|</span>
                  <span className="text-slate-400">
                    {isProcessed ? 'Ready' : 'Not processed'}
                  </span>
                </div>

                <button
                  onClick={resetSession}
                  className="flex items-center space-x-2 px-3 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg transition-colors text-sm"
                >
                  <RefreshCw className="w-4 h-4" />
                  <span className="hidden sm:inline">New Session</span>
                </button>
              </>
            )}
          </div>
        </div>

        {error && (
          <div className="mt-3 flex items-start space-x-2 p-3 bg-red-500/20 border border-red-500/50 rounded-lg">
            <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <p className="text-sm text-red-300">{error}</p>
            </div>
            <button
              onClick={clearError}
              className="text-red-400 hover:text-red-300 text-sm font-medium"
            >
              Dismiss
            </button>
          </div>
        )}
      </div>
    </header>
  );
}
