import { useState, useRef, useEffect } from 'react'
import ChatMessage from './components/ChatMessage'
import FileUpload from './components/FileUpload'
import ThemeToggle from './components/ThemeToggle'
import './App.css'

const API_URL = 'http://localhost:8000'

function App() {
  const [uploadedFileNames, setUploadedFileNames] = useState([])
  const [isUploading, setIsUploading] = useState(false)
  const [isInitialized, setIsInitialized] = useState(false)
  const [messages, setMessages] = useState(() => {
    try {
      const saved = localStorage.getItem('rag-chatbot-messages')
      return saved ? JSON.parse(saved) : []
    } catch {
      return []
    }
  })
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isThinking, setIsThinking] = useState(false)
  const [error, setError] = useState('')
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Persist messages to localStorage
  useEffect(() => {
    try {
      const toSave = messages.filter(m => !m.isStreaming)
      localStorage.setItem('rag-chatbot-messages', JSON.stringify(toSave))
    } catch {
      // localStorage full or unavailable
    }
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
    setIsThinking(true)

    try {
      const res = await fetch(`${API_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userMessage, top_k: 3 })
      })

      if (!res.ok) {
        throw new Error('Chat request failed')
      }

      const data = await res.json()
      setIsThinking(false)
      
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: data.response || "I couldn't find relevant information.",
        isStreaming: false,
        sources: data.sources || []
      }])

    } catch (e) {
      setIsThinking(false)
      setMessages(prev => [...prev, { role: 'assistant', content: 'Something went wrong. Please try again.' }])
    } finally {
      setIsLoading(false)
      setIsThinking(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const clearChat = () => {
    setMessages([])
    localStorage.removeItem('rag-chatbot-messages')
  }

  const resetAll = async () => {
    try {
      await fetch(`${API_URL}/api/clear`, { method: 'POST' })
    } catch (e) {
      console.error('Failed to clear backend documents')
    }
    clearChat()
    setIsInitialized(false)
    setUploadedFileNames([])
    setError('')
  }

  return (
    <div className="app">
      {/* Mobile overlay */}
      {sidebarOpen && <div className="sidebar-overlay" onClick={() => setSidebarOpen(false)} />}

      {/* Sidebar */}
      <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <h2>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 2L2 7l10 5 10-5-10-5z" />
              <path d="M2 17l10 5 10-5" />
              <path d="M2 12l10 5 10-5" />
            </svg>
            RAG ChatBot
          </h2>
          <ThemeToggle />
        </div>

        <div className="sidebar-content">
          <button className="new-chat-btn" onClick={resetAll}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="12" y1="5" x2="12" y2="19" />
              <line x1="5" y1="12" x2="19" y2="12" />
            </svg>
            New Chat
          </button>

          {messages.length > 0 && (
            <button className="clear-chat-btn" onClick={clearChat}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="3 6 5 6 21 6" />
                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
              </svg>
              Clear Chat
            </button>
          )}

          {isInitialized && uploadedFileNames.length > 0 && (
            <div className="loaded-files">
              <div className="loaded-files-header">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                  <polyline points="14 2 14 8 20 8" />
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

          {/* Sidebar upload for adding more documents */}
          {isInitialized && (
            <div className="sidebar-upload">
              <div className="loaded-files-header">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="17 8 12 3 7 8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
                Add More Documents
              </div>
              <FileUpload
                onUpload={uploadFiles}
                isUploading={isUploading}
                error={error}
              />
            </div>
          )}
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="main-chat">
        {/* Mobile header */}
        <div className="mobile-header">
          <button className="menu-btn" onClick={() => setSidebarOpen(true)}>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="3" y1="12" x2="21" y2="12" />
              <line x1="3" y1="6" x2="21" y2="6" />
              <line x1="3" y1="18" x2="21" y2="18" />
            </svg>
          </button>
          <span>RAG ChatBot</span>
          <ThemeToggle />
        </div>

        {!isInitialized ? (
          <div className="welcome-screen">
            <div className="welcome-icon">
              <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1">
                <path d="M12 2L2 7l10 5 10-5-10-5z" />
                <path d="M2 17l10 5 10-5" />
                <path d="M2 12l10 5 10-5" />
              </svg>
            </div>
            <h1>LLM-Powered RAG ChatBot</h1>
            <p>Upload your documents and start asking intelligent questions</p>

            <FileUpload
              onUpload={uploadFiles}
              isUploading={isUploading}
              error={error}
            />
          </div>
        ) : (
          <>
            <div className="chat-messages">
              {messages.length === 0 && (
                <div className="empty-chat">
                  <div className="empty-icon">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                    </svg>
                  </div>
                  <h2>How can I help you?</h2>
                  <p>Ask me anything about your uploaded documents.</p>
                </div>
              )}

              {messages.map((msg, i) => (
                <ChatMessage key={i} message={msg} />
              ))}

              {/* Thinking indicator */}
              {isThinking && (
                <div className="message-row assistant">
                  <div className="message-container">
                    <div className="avatar">
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <path d="M12 2L2 7l10 5 10-5-10-5z" />
                        <path d="M2 17l10 5 10-5" />
                        <path d="M2 12l10 5 10-5" />
                      </svg>
                    </div>
                    <div className="message-content">
                      <div className="thinking-indicator">
                        <span className="dot" />
                        <span className="dot" />
                        <span className="dot" />
                      </div>
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>

            <div className="input-area">
              <div className="input-container">
                <input
                  type="text"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyDown={handleKeyPress}
                  placeholder="Ask about your documents..."
                  disabled={isLoading}
                />
                <button
                  className="send-btn"
                  onClick={sendMessage}
                  disabled={isLoading || !inputValue.trim()}
                >
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
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
