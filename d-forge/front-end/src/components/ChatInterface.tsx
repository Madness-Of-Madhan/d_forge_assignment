import { useState, useRef, useEffect } from 'react';
import { Send, Sparkles, FileQuestion, ListChecks } from 'lucide-react';
import { useSession } from '../context/SessionContext';

export default function ChatInterface() {
  const { messages, sendMessage, isProcessed } = useSession();
  const [input, setInput] = useState('');
  const [queryType, setQueryType] = useState<'qa' | 'quiz' | 'summary'>('qa');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !isProcessed) return;

    const message = input.trim();
    setInput('');

    try {
      await sendMessage(message, queryType);
    } catch (error) {
      console.error('Failed to send message:', error);
    }
  };

  const queryTypes = [
    { value: 'qa', label: 'Q&A', icon: FileQuestion, color: 'blue' },
    { value: 'quiz', label: 'Quiz', icon: ListChecks, color: 'green' },
    { value: 'summary', label: 'Summary', icon: Sparkles, color: 'purple' },
  ] as const;

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto space-y-4 p-4 md:p-6">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center space-y-4">
            <div className="p-6 bg-slate-800/50 rounded-full">
              <Sparkles className="w-12 h-12 text-blue-400" />
            </div>
            <div>
              <h2 className="text-2xl font-semibold text-slate-200 mb-2">
                Ready to Chat!
              </h2>
              <p className="text-slate-400 max-w-md">
                Ask questions about your documents, generate quizzes, or request summaries.
              </p>
            </div>
          </div>
        ) : (
          <>
            {messages.map(message => (
              <div
                key={message.id}
                className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[85%] md:max-w-[75%] rounded-2xl px-4 py-3 ${
                    message.type === 'user'
                      ? 'bg-blue-600 text-white'
                      : 'bg-slate-800/70 backdrop-blur text-slate-200 border border-slate-700'
                  } ${message.optimistic ? 'animate-pulse' : ''}`}
                >
                  {message.queryType && message.type === 'assistant' && (
                    <div className="flex items-center space-x-1 text-xs text-slate-400 mb-2">
                      {message.queryType === 'quiz' && (
                        <ListChecks className="w-3 h-3" />
                      )}
                      {message.queryType === 'summary' && (
                        <Sparkles className="w-3 h-3" />
                      )}
                      {message.queryType === 'qa' && (
                        <FileQuestion className="w-3 h-3" />
                      )}
                      <span className="capitalize">{message.queryType}</span>
                    </div>
                  )}
                  <div className="whitespace-pre-wrap break-words">
                    {message.content}
                  </div>
                  <div className="text-xs opacity-70 mt-2">
                    {message.timestamp.toLocaleTimeString()}
                  </div>
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      <div className="border-t border-slate-700 bg-slate-900/80 backdrop-blur p-4">
        <div className="flex flex-wrap gap-2 mb-3">
          {queryTypes.map(({ value, label, icon: Icon, color }) => (
            <button
              key={value}
              onClick={() => setQueryType(value)}
              className={`flex items-center space-x-2 px-3 py-1.5 rounded-full text-sm font-medium transition-all ${
                queryType === value
                  ? `bg-${color}-600 text-white shadow-lg scale-105`
                  : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
              }`}
            >
              <Icon className="w-4 h-4" />
              <span>{label}</span>
            </button>
          ))}
        </div>

        <form onSubmit={handleSubmit} className="flex space-x-2">
          <input
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder={
              isProcessed
                ? 'Ask a question about your documents...'
                : 'Please upload and process documents first'
            }
            disabled={!isProcessed}
            className="flex-1 px-4 py-3 bg-slate-800 border border-slate-700 rounded-xl text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
          />
          <button
            type="submit"
            disabled={!input.trim() || !isProcessed}
            className="px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-700 disabled:cursor-not-allowed text-white rounded-xl transition-colors font-medium flex items-center space-x-2"
          >
            <Send className="w-5 h-5" />
            <span className="hidden sm:inline">Send</span>
          </button>
        </form>
      </div>
    </div>
  );
}
