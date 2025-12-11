import { useState, useRef } from 'react';
import { Upload, File, X, CheckCircle2 } from 'lucide-react';
import { useSession } from '../context/SessionContext';

export default function FileUpload() {
  const { uploadFiles, processDocuments, uploadedFiles, isProcessed, isLoading } = useSession();
  const [dragActive, setDragActive] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [uploadComplete, setUploadComplete] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const files = Array.from(e.dataTransfer.files).filter(
      file => file.type === 'application/pdf'
    );

    if (files.length > 0) {
      setSelectedFiles(prev => [...prev, ...files]);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files = Array.from(e.target.files);
      setSelectedFiles(prev => [...prev, ...files]);
    }
  };

  const removeFile = (index: number) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleUpload = async () => {
    if (selectedFiles.length === 0) return;

    try {
      await uploadFiles(selectedFiles);
      setUploadComplete(true);
      setSelectedFiles([]);
    } catch (error) {
      console.error('Upload failed:', error);
    }
  };

  const handleProcess = async () => {
    try {
      await processDocuments();
    } catch (error) {
      console.error('Processing failed:', error);
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto space-y-6">
      <div
        className={`relative border-2 border-dashed rounded-xl p-8 transition-all ${
          dragActive
            ? 'border-blue-500 bg-blue-50/10'
            : 'border-slate-600 hover:border-slate-500'
        } ${isProcessed ? 'opacity-50 pointer-events-none' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".pdf"
          onChange={handleFileSelect}
          className="hidden"
          disabled={isProcessed}
        />

        <div className="flex flex-col items-center justify-center space-y-4">
          <div className="p-4 bg-blue-500/20 rounded-full">
            <Upload className="w-10 h-10 text-blue-400" />
          </div>

          <div className="text-center">
            <p className="text-lg font-medium text-slate-200">
              Drop PDF files here or click to browse
            </p>
            <p className="text-sm text-slate-400 mt-1">
              Support for multiple PDF documents
            </p>
          </div>

          <button
            onClick={() => fileInputRef.current?.click()}
            className="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors font-medium"
            disabled={isProcessed}
          >
            Select Files
          </button>
        </div>
      </div>

      {selectedFiles.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-medium text-slate-300">Selected Files</h3>
          <div className="space-y-2">
            {selectedFiles.map((file, index) => (
              <div
                key={index}
                className="flex items-center justify-between p-3 bg-slate-800/50 backdrop-blur rounded-lg border border-slate-700"
              >
                <div className="flex items-center space-x-3">
                  <File className="w-5 h-5 text-blue-400" />
                  <div>
                    <p className="text-sm font-medium text-slate-200">{file.name}</p>
                    <p className="text-xs text-slate-400">
                      {(file.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                </div>
                <button
                  onClick={() => removeFile(index)}
                  className="p-1 hover:bg-slate-700 rounded transition-colors"
                >
                  <X className="w-4 h-4 text-slate-400" />
                </button>
              </div>
            ))}
          </div>

          <button
            onClick={handleUpload}
            disabled={isLoading}
            className="w-full px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-700 disabled:cursor-not-allowed text-white rounded-lg transition-colors font-medium"
          >
            {isLoading ? 'Uploading...' : `Upload ${selectedFiles.length} File(s)`}
          </button>
        </div>
      )}

      {uploadComplete && uploadedFiles.length > 0 && !isProcessed && (
        <div className="space-y-3">
          <div className="flex items-center space-x-2 text-green-400">
            <CheckCircle2 className="w-5 h-5" />
            <p className="text-sm font-medium">Files uploaded successfully!</p>
          </div>

          <button
            onClick={handleProcess}
            disabled={isLoading}
            className="w-full px-6 py-3 bg-green-600 hover:bg-green-700 disabled:bg-slate-700 disabled:cursor-not-allowed text-white rounded-lg transition-colors font-medium"
          >
            {isLoading ? 'Processing Documents...' : 'Process Documents'}
          </button>
        </div>
      )}

      {isProcessed && (
        <div className="flex items-center justify-center space-x-2 p-4 bg-green-500/20 border border-green-500/50 rounded-lg">
          <CheckCircle2 className="w-5 h-5 text-green-400" />
          <p className="text-sm font-medium text-green-400">
            Documents processed! Ready to chat.
          </p>
        </div>
      )}

      {uploadedFiles.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-medium text-slate-300">Uploaded Documents</h3>
          <div className="space-y-1">
            {uploadedFiles.map((file, index) => (
              <div
                key={index}
                className="flex items-center space-x-2 p-2 bg-slate-800/30 rounded"
              >
                <File className="w-4 h-4 text-blue-400" />
                <span className="text-sm text-slate-300">{file}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
