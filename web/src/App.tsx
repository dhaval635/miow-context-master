import { useState, useEffect } from 'react'
/* eslint-disable @typescript-eslint/no-unused-vars */

interface GenerateResponse {
  success: boolean
  result?: string
  error?: string
}

interface HealthResponse {
  status: string
  version: string
  qdrant_connected: boolean
  gemini_configured: boolean
}

interface ProjectSignature {
  language: string
  framework: string
  package_manager: string
  ui_library: string
  validation_library: string
  auth_library: string
  description: string
}

interface DebugContext {
  total_symbols: number
  total_files: number
  db_path: string
  collection_name: string
}

interface FileInfo {
  file_path: string
  symbol_name: string
  symbol_kind: string
  relevance_score: number
  preview: string
}

interface AgentEvent {
  type: 'Step' | 'Thought' | 'ToolCall' | 'ToolOutput' | 'Error' | 'Done'
  data?: {
    step?: number
    max_steps?: number
    content?: string
    tool?: string
    args?: any
    output?: string
    error?: string
  }
}

function App() {
  const [codebasePath, setCodebasePath] = useState('')
  const [userPrompt, setUserPrompt] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  
  
  // Agent streaming state
  const [isStreaming, setIsStreaming] = useState(false)
  const [agentEvents, setAgentEvents] = useState<AgentEvent[]>([])
  const [streamStatus, setStreamStatus] = useState<string>('')
  const [isPaused, setIsPaused] = useState(false)
  const [eventFilter, setEventFilter] = useState<Set<string>>(new Set(['Step', 'Thought', 'ToolCall', 'ToolOutput', 'Error', 'Done']))
  const [streamReader, setStreamReader] = useState<ReadableStreamDefaultReader<Uint8Array> | null>(null)
  
  // File upload
// uploadedFiles state removed (now using codebase file search)
  
  // File search from codebase
  const [fileSearchQuery, setFileSearchQuery] = useState('')
  const [fileSearchResults, setFileSearchResults] = useState<string[]>([])
  const [selectedCodebaseFiles, setSelectedCodebaseFiles] = useState<Set<string>>(new Set())
  const [isSearchingFiles, setIsSearchingFiles] = useState(false)
  
  // Debug info
  const [signature, setSignature] = useState<ProjectSignature | null>(null)
  const [context, setContext] = useState<DebugContext | null>(null)
  const [loadingDebug, setLoadingDebug] = useState(false)
  
  // File selection
  const [relevantFiles, setRelevantFiles] = useState<FileInfo[]>([])
  const [selectedFiles, setSelectedFiles] = useState<Set<string>>(new Set())
  const [loadingFiles, setLoadingFiles] = useState(false)
  const [showFileSelection, setShowFileSelection] = useState(false)

  // Check backend health on mount
  useEffect(() => {
    checkHealth()
  }, [])

  const checkHealth = async () => {
    try {
      const response = await fetch('/api/health', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      })
      const healthData: HealthResponse = await response.json()
      setHealth(healthData)
    } catch (err) {
      console.error('Health check failed:', err)
    }
  }

  const loadDebugInfo = async () => {
    if (!codebasePath.trim()) return
    
    setLoadingDebug(true)
    try {
      // Load signature
      const sigResponse = await fetch('/api/debug/signature', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ codebase_path: codebasePath })
      })
      const sigData = await sigResponse.json()
      if (sigData.success && sigData.signature) {
        setSignature(sigData.signature)
      }
      
      // Load context
      const ctxResponse = await fetch('/api/debug/context', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ codebase_path: codebasePath })
      })
      const ctxData = await ctxResponse.json()
      if (ctxData.success && ctxData.context) {
        setContext(ctxData.context)
      }
    } catch (err) {
      console.error('Failed to load debug info:', err)
    } finally {
      setLoadingDebug(false)
    }
  }

  const loadRelevantFiles = async () => {
    if (!codebasePath.trim() || !userPrompt.trim()) {
      setError('Please provide both codebase path and prompt')
      return
    }

    setLoadingFiles(true)
    setError(null)
    try {
      const response = await fetch('/api/files', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          codebase_path: codebasePath,
          user_prompt: userPrompt
        })
      })

      const data = await response.json()
      if (data.success && data.files) {
        setRelevantFiles(data.files)
        setShowFileSelection(true)
        setSelectedFiles(new Set()) // Reset selection
      } else {
        setError(data.error || 'Failed to load relevant files')
      }
    } catch (err) {
      setError('Failed to connect to backend. Make sure the server is running.')
      console.error('Request failed:', err)
    } finally {
      setLoadingFiles(false)
    }
  }

  const toggleFileSelection = (filePath: string) => {
    const newSelection = new Set(selectedFiles)
    if (newSelection.has(filePath)) {
      newSelection.delete(filePath)
    } else {
      newSelection.add(filePath)
    }
    setSelectedFiles(newSelection)
  }

  const handleGenerate = async () => {
    if (!codebasePath.trim() || !userPrompt.trim()) {
      setError('Please provide both codebase path and prompt')
      return
    }

    setIsStreaming(true)
    setIsLoading(true)
    setError(null)
    setResult(null)
    setAgentEvents([])
    setStreamStatus('Connecting to agent...')

    try {
      const response = await fetch('/api/generate-stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          codebase_path: codebasePath,
          user_prompt: userPrompt
        })
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const reader = response.body?.getReader()
      const decoder = new TextDecoder()

      if (!reader) {
        throw new Error('No response body')
      }

      // Store reader for pause/resume
      setStreamReader(reader)

      while (true) {
        // Check if paused
        if (isPaused) {
          await new Promise(resolve => setTimeout(resolve, 100))
          continue
        }

        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value)
        const lines = chunk.split('\n')

        for (const line of lines) {
          if (line.startsWith('event:')) {
            // Event type line, skip it
            continue
          }

          if (line.startsWith('data:')) {
            const data = line.substring(5).trim()
            
            if (!data) continue

            try {
              if (data === 'Starting autonomous agent...') {
                setStreamStatus(data)
              } else if (data.startsWith('No index found') || data.startsWith('Indexing')) {
                setStreamStatus(data)
              } else if (data.startsWith('{')) {
                // Agent event JSON
                const event: AgentEvent = JSON.parse(data)
                
                // Apply event filter
                if (eventFilter.has(event.type)) {
                  setAgentEvents(prev => [...prev, event])
                }
                
                // Update status based on event type
                if (event.type === 'Step' && event.data) {
                  setStreamStatus(`Step ${event.data.step}/${event.data.max_steps}`)
                } else if (event.type === 'Thought' && event.data?.content) {
                  setStreamStatus(event.data.content)
                } else if (event.type === 'ToolCall' && event.data?.tool) {
                  setStreamStatus(`Executing: ${event.data.tool}`)
                } else if (event.type === 'Done') {
                  setStreamStatus('Agent completed!')
                }
              } else {
                // Final result
                setResult(data)
                setStreamStatus('Complete!')
              }
            } catch (e) {
              console.error('Failed to parse SSE data:', e, data)
            }
          }
        }
      }
    } catch (err) {
      setError('Failed to connect to backend. Make sure the server is running.')
      console.error('Request failed:', err)
    } finally {
      setIsLoading(false)
      setIsStreaming(false)
      setStreamReader(null)
    }
  }

  const togglePause = () => {
    setIsPaused(prev => !prev)
  }

  const stopStreaming = () => {
    if (streamReader) {
      streamReader.cancel();
      setStreamReader(null);
    }
    setIsStreaming(false);
    setIsLoading(false);
    setStreamStatus('Stopped by user');
  };

  const toggleEventFilter = (eventType: string) => {
    setEventFilter(prev => {
      const newFilter = new Set(prev);
      if (newFilter.has(eventType)) {
        newFilter.delete(eventType);
      } else {
        newFilter.add(eventType);
      }
      return newFilter;
    });
  };

  // handleFileUpload removed – not needed with codebase search

  const searchCodebaseFiles = async (query: string) => {
    if (!query.trim() || !codebasePath.trim()) {
      setFileSearchResults([])
      return
    }

    setIsSearchingFiles(true)
    try {
      const response = await fetch('/api/search-files', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          codebase_path: codebasePath,
          query: query
        })
      })

      const data = await response.json()
      if (data.success && data.files) {
        setFileSearchResults(data.files)
      } else {
        setFileSearchResults([])
      }
    } catch (err) {
      console.error('File search failed:', err)
      setFileSearchResults([])
    } finally {
      setIsSearchingFiles(false)
    }
  }

  const toggleCodebaseFile = (filePath: string) => {
    setSelectedCodebaseFiles(prev => {
      const newSet = new Set(prev)
      if (newSet.has(filePath)) {
        newSet.delete(filePath)
      } else {
        newSet.add(filePath)
      }
      return newSet
    })
  }

  const handleGenerateWithFiles = async () => {
    if (!codebasePath.trim() || !userPrompt.trim()) {
      setError('Please provide both codebase path and prompt')
      return
    }

    if (selectedCodebaseFiles.size === 0) {
      setError('Please select at least one file')
      return
    }

    setIsLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await fetch('/api/generate-with-files', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          codebase_path: codebasePath,
          user_prompt: userPrompt,
          selected_files: Array.from(selectedCodebaseFiles)
        })
      })

      const data: GenerateResponse = await response.json()

      if (data.success && data.result) {
        setResult(data.result)
      } else {
        setError(data.error || 'Unknown error occurred')
      }
    } catch (err) {
      setError('Failed to connect to backend. Make sure the server is running.')
      console.error('Request failed:', err)
    } finally {
      setIsLoading(false)
    }
  }

  const copyToClipboard = async () => {
    if (result) {
      try {
        await navigator.clipboard.writeText(result)
        // Could add a toast notification here
      } catch (err) {
        console.error('Failed to copy:', err)
      }
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="logo">
          <h1>🤖 MIOW-CONTEXT</h1>
          <p>Autonomous Code Context Generation</p>
        </div>
        {health && (
          <div className="health-status">
            <div className={`status-indicator ${health.qdrant_connected ? 'connected' : 'disconnected'}`}>
              Qdrant: {health.qdrant_connected ? '🟢' : '🔴'}
            </div>
            <div className={`status-indicator ${health.gemini_configured ? 'connected' : 'disconnected'}`}>
              Gemini: {health.gemini_configured ? '🟢' : '🔴'}
            </div>
          </div>
        )}
      </header>

      <main className="main">
        <div className="input-section">
          <div className="input-group">
            <label htmlFor="codebase-path">📁 Codebase Path</label>
            <div style={{ display: 'flex', gap: '8px' }}>
              <input
                id="codebase-path"
                type="text"
                placeholder="e.g., /path/to/your/project or ./current/folder"
                value={codebasePath}
                onChange={(e) => {
                  setCodebasePath(e.target.value)
                  setSignature(null)
                  setContext(null)
                }}
                className="input-field"
                style={{ flex: 1 }}
              />
              <button
                onClick={loadDebugInfo}
                disabled={!codebasePath.trim() || loadingDebug}
                className="debug-btn"
                title="Load project signature and context info"
              >
                {loadingDebug ? '⏳' : '🔍'}
              </button>
            </div>
            <small className="input-hint">
              Absolute path or relative path to your codebase
            </small>
          </div>

          <div className="input-group">
            <label htmlFor="user-prompt">💭 Your Task/Prompt</label>
            <textarea
              id="user-prompt"
              placeholder="e.g., Add user authentication with JWT tokens to my React app"
              value={userPrompt}
              onChange={(e) => setUserPrompt(e.target.value)}
              className="textarea-field"
              rows={4}
            />
            <small className="input-hint">
              Describe what you want to implement or ask about your codebase
            </small>
          </div>

          <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
            <button
              onClick={loadRelevantFiles}
              disabled={loadingFiles || !health?.qdrant_connected || !health?.gemini_configured}
              className="secondary-btn"
            >
              {loadingFiles ? (
                <>
                  <div className="spinner"></div>
                  Loading Files...
                </>
              ) : (
                <>
                  📋 Load Relevant Files
                </>
              )}
            </button>
            
            <button
              onClick={handleGenerate}
              disabled={isLoading || !health?.qdrant_connected || !health?.gemini_configured}
              className="generate-btn"
            >
              {isLoading ? (
                <>
                  <div className="spinner"></div>
                  Generating Context...
                </>
              ) : (
                <>
                  🚀 Generate Context
                </>
              )}
            </button>
            
            {/* Streaming Controls */}
            {isStreaming && (
              <>
                <button
                  onClick={togglePause}
                  className="secondary-btn"
                  style={{ flex: '0 0 auto' }}
                >
                  {isPaused ? '▶️ Resume' : '⏸️ Pause'}
                </button>
                
                <button
                  onClick={stopStreaming}
                  className="close-btn"
                  style={{ flex: '0 0 auto' }}
                >
                  ⏹️ Stop
                </button>
              </>
            )}
          </div>

          {/* File Search Section */}
          <div className="input-group" style={{ marginTop: '1rem' }}>
            <label htmlFor="file-search">🔍 Search & Add Files from Codebase (Optional)</label>
            <input
              id="file-search"
              type="text"
              placeholder="Search for files... (e.g., 'agent', 'utils.ts', 'components')"
              value={fileSearchQuery}
              onChange={(e) => {
                setFileSearchQuery(e.target.value)
                searchCodebaseFiles(e.target.value)
              }}
              className="input-field"
              disabled={!codebasePath.trim()}
            />
            
            {isSearchingFiles && (
              <div style={{ marginTop: '0.5rem', fontSize: '0.9rem', color: '#666' }}>
                🔄 Searching...
              </div>
            )}
            
            {fileSearchResults.length > 0 && (
              <div style={{ 
                marginTop: '0.75rem', 
                maxHeight: '200px', 
                overflowY: 'auto',
                border: '1px solid #e1e5e9',
                borderRadius: '8px',
                background: '#f8f9fa'
              }}>
                {fileSearchResults.map((filePath, idx) => (
                  <div
                    key={idx}
                    onClick={() => toggleCodebaseFile(filePath)}
                    style={{
                      padding: '0.75rem',
                      cursor: 'pointer',
                      borderBottom: idx < fileSearchResults.length - 1 ? '1px solid #e1e5e9' : 'none',
                      background: selectedCodebaseFiles.has(filePath) ? '#e7f3ff' : 'white',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.5rem'
                    }}
                  >
                    <input
                      type="checkbox"
                      checked={selectedCodebaseFiles.has(filePath)}
                      onChange={() => {}}
                      style={{ cursor: 'pointer' }}
                    />
                    <span style={{ 
                      fontSize: '0.9rem',
                      fontFamily: 'monospace',
                      color: selectedCodebaseFiles.has(filePath) ? '#0066cc' : '#333'
                    }}>
                      {filePath}
                    </span>
                  </div>
                ))}
              </div>
            )}
            
            {selectedCodebaseFiles.size > 0 && (
              <div style={{ marginTop: '0.75rem', padding: '0.75rem', background: '#e7f3ff', borderRadius: '8px' }}>
                <strong style={{ fontSize: '0.9rem', color: '#0066cc' }}>
                  ✓ {selectedCodebaseFiles.size} file(s) selected
                </strong>
                <div style={{ marginTop: '0.5rem', fontSize: '0.85rem', color: '#666' }}>
                  {Array.from(selectedCodebaseFiles).map((file, idx) => (
                    <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.25rem' }}>
                      <span>📄 {file.split('/').pop()}</span>
                      <button
                        onClick={() => toggleCodebaseFile(file)}
                        style={{
                          background: 'none',
                          border: 'none',
                          color: '#dc3545',
                          cursor: 'pointer',
                          fontSize: '0.85rem'
                        }}
                      >
                        ✕
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {!health?.qdrant_connected && (
            <div className="warning">
              ⚠️ Qdrant database not connected. Make sure it's running: docker-compose up -d
            </div>
          )}

          {!health?.gemini_configured && (
            <div className="warning">
              ⚠️ Gemini API key not configured. Set GEMINI_API_KEY environment variable.
            </div>
          )}

          {/* Debug Info Section */}
          {(signature || context) && (
            <div className="debug-section">
              <h3>🔍 Project Debug Info</h3>
              {signature && (
                <div className="debug-card">
                  <h4>📋 Project Signature</h4>
                  <div className="debug-grid">
                    <div><strong>Language:</strong> {signature.language || 'Unknown'}</div>
                    <div><strong>Framework:</strong> {signature.framework || 'Unknown'}</div>
                    <div><strong>Package Manager:</strong> {signature.package_manager || 'Unknown'}</div>
                    <div><strong>UI Library:</strong> {signature.ui_library || 'None'}</div>
                    <div><strong>Validation:</strong> {signature.validation_library || 'None'}</div>
                    <div><strong>Auth:</strong> {signature.auth_library || 'None'}</div>
                  </div>
                  {signature.description && (
                    <div style={{ marginTop: '8px', fontSize: '0.9em', color: '#666' }}>
                      {signature.description}
                    </div>
                  )}
                </div>
              )}
              {context && (
                <div className="debug-card">
                  <h4>📊 Context Statistics</h4>
                  <div className="debug-grid">
                    <div><strong>Total Symbols:</strong> {context.total_symbols.toLocaleString()}</div>
                    <div><strong>Total Files:</strong> {context.total_files.toLocaleString()}</div>
                    <div><strong>DB Path:</strong> <code style={{ fontSize: '0.85em' }}>{context.db_path}</code></div>
                    <div><strong>Collection:</strong> <code style={{ fontSize: '0.85em' }}>{context.collection_name}</code></div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* File Selection Section */}
          {showFileSelection && relevantFiles.length > 0 && (
            <div className="file-selection-section">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                <h3>📁 Relevant Files ({selectedFiles.size} selected)</h3>
                <button
                  onClick={() => {
                    setShowFileSelection(false)
                    setSelectedFiles(new Set())
                  }}
                  className="close-btn"
                >
                  ✕
                </button>
              </div>
              <div className="file-list">
                {relevantFiles.map((file, idx) => (
                  <div
                    key={idx}
                    className={`file-item ${selectedFiles.has(file.file_path) ? 'selected' : ''}`}
                    onClick={() => toggleFileSelection(file.file_path)}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <input
                        type="checkbox"
                        checked={selectedFiles.has(file.file_path)}
                        onChange={() => toggleFileSelection(file.file_path)}
                        onClick={(e) => e.stopPropagation()}
                      />
                      <div style={{ flex: 1 }}>
                        <div style={{ fontWeight: 'bold', color: '#333' }}>
                          {file.file_path}
                        </div>
                        <div style={{ fontSize: '0.85em', color: '#666', marginTop: '4px' }}>
                          <span style={{ backgroundColor: '#e3f2fd', padding: '2px 6px', borderRadius: '4px', marginRight: '8px' }}>
                            {file.symbol_kind}
                          </span>
                          <span>Relevance: {(file.relevance_score * 100).toFixed(1)}%</span>
                        </div>
                        <div style={{ fontSize: '0.8em', color: '#999', marginTop: '4px', fontFamily: 'monospace' }}>
                          {file.preview}...
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              <button
                onClick={handleGenerateWithFiles}
                disabled={isLoading || selectedFiles.size === 0 || !health?.qdrant_connected || !health?.gemini_configured}
                className="generate-btn"
                style={{ marginTop: '16px', width: '100%' }}
              >
                {isLoading ? (
                  <>
                    <div className="spinner"></div>
                    Generating with Selected Files...
                  </>
                ) : (
                  <>
                    🚀 Generate with Selected Files ({selectedFiles.size})
                  </>
                )}
              </button>
            </div>
          )}
        </div>

        <div className="output-section">
          {/* Agent State Section */}
          {(isStreaming || agentEvents.length > 0) && (
            <div className="agent-state-container">
              <h3>🤖 Agent State</h3>
              
              {streamStatus && (
                <div className="stream-status">
                  <div className="status-badge">
                    {isStreaming ? '🔄' : '✅'} {streamStatus}
                  </div>
                </div>
              )}

              {/* Event Filter Controls */}
              <div style={{ marginTop: '1rem', marginBottom: '1rem' }}>
                <h4 style={{ fontSize: '0.95rem', marginBottom: '0.5rem', color: '#666' }}>Event Filters:</h4>
                <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
                  {['Step', 'Thought', 'ToolCall', 'ToolOutput', 'Error', 'Done'].map(eventType => (
                    <label key={eventType} style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', cursor: 'pointer' }}>
                      <input
                        type="checkbox"
                        checked={eventFilter.has(eventType)}
                        onChange={() => toggleEventFilter(eventType)}
                        style={{ cursor: 'pointer' }}
                      />
                      <span style={{ fontSize: '0.9rem' }}>{eventType}</span>
                    </label>
                  ))}
                </div>
              </div>

              {agentEvents.length > 0 && (
                <div className="agent-events">
                  <h4>Event History</h4>
                  <div className="events-list">
                    {agentEvents.map((event, idx) => (
                      <div key={idx} className={`event event-${event.type.toLowerCase()}`}>
                        <div className="event-header">
                          <span className="event-type">
                            {event.type === 'Step' && '📍'}
                            {event.type === 'Thought' && '💭'}
                            {event.type === 'ToolCall' && '🔨'}
                            {event.type === 'ToolOutput' && '📤'}
                            {event.type === 'Error' && '❌'}
                            {event.type === 'Done' && '✅'}
                            {' '}{event.type}
                          </span>
                          {event.type === 'Step' && event.data && (
                            <span className="event-meta">
                              Step {event.data.step}/{event.data.max_steps}
                            </span>
                          )}
                        </div>
                        <div className="event-content">
                          {event.type === 'Thought' && event.data?.content}
                          {event.type === 'ToolCall' && event.data?.tool && (
                            <div>
                              <strong>{event.data.tool}</strong>
                              {event.data.args && (
                                <pre style={{ fontSize: '0.85em', marginTop: '4px' }}>
                                  {JSON.stringify(event.data.args, null, 2)}
                                </pre>
                              )}
                            </div>
                          )}
                          {event.type === 'ToolOutput' && event.data?.output && (
                            <pre style={{ fontSize: '0.85em', maxHeight: '200px', overflow: 'auto' }}>
                              {event.data.output}
                            </pre>
                          )}
                          {event.type === 'Error' && event.data?.error}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {error && (
            <div className="error-message">
              <h3>❌ Error</h3>
              <pre>{error}</pre>
            </div>
          )}

          {result && (
            <div className="result-container">
              <div className="result-header">
                <h3>✅ Generated Context</h3>
                <button onClick={copyToClipboard} className="copy-btn">
                  📋 Copy to Clipboard
                </button>
              </div>
              <pre className="result-content">{result}</pre>
            </div>
          )}
        </div>
      </main>

      <footer className="footer">
        <div className="features">
          <div className="feature">
            <h4>🧠 Autonomous</h4>
            <p>LLM-driven analysis with no hardcoded biases</p>
          </div>
          <div className="feature">
            <h4>🔍 Smart Search</h4>
            <p>Multi-agent workers with dependency resolution</p>
          </div>
          <div className="feature">
            <h4>🎯 Context-Aware</h4>
            <p>Token-optimized prompts for any LLM</p>
          </div>
          <div className="feature">
            <h4>⚡ Fast</h4>
            <p>Gemini 2.5 Flash + Qdrant vector search</p>
          </div>
        </div>
        <div className="copyright">
          <p>Made with ❤️ for developers who want smarter code generation</p>
        </div>
      </footer>
    </div>
  )
}

export default App
