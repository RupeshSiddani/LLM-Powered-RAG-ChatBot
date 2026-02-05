import { useState, useRef, useEffect } from 'react'
import './App.css'

const API_URL = 'http://localhost:8000'

// Simple markdown-like formatting
const formatMessage = (text) => {
  if (!text) return text
  
  let formatted = text
    .replace(/^### (.+)$/gm, '<h4>$1</h4>')
    .replace(/^## (.+)$/gm, '<h3>$1</h3>')
    .replace(/^# (.+)$/gm, '<h2>$1</h2>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/^[\-\*] (.+)$/gm, '<li>$1</li>')
    .replace(/^\d+\. (.+)$/gm, '<li>$1</li>')
    .replace(/\n\n/g, '</p><p>')
    .replace(/\n/g, '<br/>')
  
  if (formatted.includes('<li>')) {
    formatted = formatted.replace(/(<li>.*?<\/li>)+/g, '<ul>$&</ul>')
  }
  
  return `<p>${formatted}</p>`
}

function App() {
  const [files, setFiles] = useState([])
  const [uploadedFileNames, setUploadedFileNames] = useState([])
  const [isUploading, setIsUploading] = useState(false)
  const [isInitialized, setIsInitialized] = useState(false)
  const [messages, setMessages] = useState([])
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')
  const messagesEndRef = useRef(null)
  const fileInputRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  useEffect(() => {
    checkHealth()
  }, [])

  const checkHealth = async () => {
    try {
      const res = await fetch(`${API_URL}/api/health`)
      const data = await res.json()
      if (data.initialized && data.document_count > 0) {
        setIsInitialized(true)
        setUploadedFileNames(['Previously loaded documents'])
      }
    } catch (e) {
      console.log('API not available')
    }
  }

  const handleFileChange = (e) => {
    const selectedFiles = Array.from(e.target.files)
    setFiles(selectedFiles)
    if (selectedFiles.length > 0) {
      uploadFiles(selectedFiles)
    }
  }

  const uploadFiles = async (filesToUpload) => {
    setIsUploading(true)
    setError('')
    
    const formData = new FormData()
    filesToUpload.forEach(file => formData.append('files', file))
    
    try {
      const res = await fetch(`${API_URL}/api/upload`, {
        method: 'POST',
        body: formData
      })
      
      if (!res.ok) {
        const error = await res.json()
        throw new Error(error.detail || 'Upload failed')
      }
      
      const data = await res.json()
      setIsInitialized(true)
      setUploadedFileNames(data.files_processed || filesToUpload.map(f => f.name))
      setFiles([])
    } catch (e) {
      setError(e.message)
    } finally {
      setIsUploading(false)
    }
  }

  const sendMessage = async () => {
    if (!inputValue.trim() || isLoading) return
    
    const userMessage = inputValue.trim()
    setInputValue('')
    setMessages(prev => [...prev, { role: 'user', content: userMessage }])
    setIsLoading(true)
    
    setMessages(prev => [...prev, { role: 'assistant', content: '', isStreaming: true }])
    
    try {
      const res = await fetch(`${API_URL}/api/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userMessage, top_k: 3 })
      })
      
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let fullResponse = ''
      
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        
        const chunk = decoder.decode(value)
        const lines = chunk.split('\n')
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6)
            if (data === '[DONE]') break
            fullResponse += data
            setMessages(prev => {
              const updated = [...prev]
              updated[updated.length - 1] = { role: 'assistant', content: fullResponse, isStreaming: true }
              return updated
            })
          }
        }
      }
      
      setMessages(prev => {
        const updated = [...prev]
        updated[updated.length - 1] = { role: 'assistant', content: fullResponse || 'I couldn\'t find relevant information.' }
        return updated
      })
      
    } catch (e) {
      setMessages(prev => {
        const updated = [...prev]
        updated[updated.length - 1] = { role: 'assistant', content: 'Something went wrong. Please try again.' }
        return updated
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const resetDocuments = () => {
    setIsInitialized(false)
    setMessages([])
    setFiles([])
    setUploadedFileNames([])
    setError('')
  }

  return (
    <div className="app">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <h2>LLM Powered RAG Chat Bot</h2>
        </div>
        
        <div className="sidebar-content">
          <button 
            className="new-chat-btn"
            onClick={resetDocuments}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="12" y1="5" x2="12" y2="19"></line>
              <line x1="5" y1="12" x2="19" y2="12"></line>
            </svg>
            New Chat
          </button>
          
          {isInitialized && uploadedFileNames.length > 0 && (
            <div className="loaded-files">
              <div className="loaded-files-header">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                  <polyline points="14 2 14 8 20 8"></polyline>
                </svg>
                Active Documents
              </div>
              <ul className="file-list">
                {uploadedFileNames.map((name, i) => (
                  <li key={i}>{name}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="main-chat">
        {!isInitialized ? (
          <div className="welcome-screen">
            <h1>LLM Powered RAG Chat Bot</h1>
            <p>Upload your documents to start chatting</p>
            
            <div className="upload-area">
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".pdf,.txt,.csv,.docx,.xlsx,.json"
                onChange={handleFileChange}
                hidden
              />
              
              <button 
                className="upload-btn"
                onClick={() => fileInputRef.current?.click()}
                disabled={isUploading}
              >
                {isUploading ? (
                  <span className="loading-spinner"></span>
                ) : (
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                    <polyline points="17 8 12 3 7 8"></polyline>
                    <line x1="12" y1="3" x2="12" y2="15"></line>
                  </svg>
                )}
                {isUploading ? 'Processing...' : 'Upload Documents'}
              </button>
              
              <p className="upload-hint">Supports PDF, TXT, DOCX, CSV, XLSX, JSON</p>
              
              {error && <p className="error-text">{error}</p>}
            </div>
          </div>
        ) : (
          <>
            <div className="chat-messages">
              {messages.length === 0 && (
                <div className="empty-chat">
                  <h2>How can I help you?</h2>
                  <p>Ask me anything about your uploaded documents.</p>
                </div>
              )}
              
              {messages.map((msg, i) => (
                <div key={i} className={`message-row ${msg.role}`}>
                  <div className="message-container">
                    <div className="avatar">
                      {msg.role === 'user' ? 'U' : 'AI'}
                    </div>
                    <div className="message-content">
                      {msg.role === 'assistant' ? (
                        <div 
                          className="formatted-text"
                          dangerouslySetInnerHTML={{ __html: formatMessage(msg.content) }}
                        />
                      ) : (
                        <p>{msg.content}</p>
                      )}
                      {msg.isStreaming && <span className="cursor-blink"></span>}
                    </div>
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>

            <div className="input-area">
              <div className="input-container">
                <input
                  type="text"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Message LLM Powered RAG Chat Bot..."
                  disabled={isLoading}
                />
                <button 
                  className="send-btn"
                  onClick={sendMessage}
                  disabled={isLoading || !inputValue.trim()}
                >
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"></path>
                  </svg>
                </button>
              </div>
            </div>
          </>
        )}
      </main>
    </div>
  )
}

export default App
