import React, { useEffect, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import toast from 'react-hot-toast'
import { apiClient } from '../utils/api'

interface TradingMessage {
  role: 'user' | 'assistant'
  content: string
  timestamp?: string
}

const TradingPlanner: React.FC = () => {
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [sessions, setSessions] = useState<Array<{ id: string; title: string; updated_at?: string }>>([])
  const [messages, setMessages] = useState<TradingMessage[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false)
  const [textareaRef, setTextareaRef] = useState<HTMLTextAreaElement | null>(null)

  const loadSessions = async () => {
    try {
      const res = await apiClient.tradingListSessions()
      const list = res.sessions || []
      setSessions(list)
      if (!sessionId && list.length > 0) {
        await switchSession(list[0].id)
      }
    } catch (e) {
      console.error('Failed to load trading sessions', e)
    }
  }

  const switchSession = async (id: string) => {
    setSessionId(id)
    try {
      const res = await apiClient.tradingGetSession(id)
      if (res.status === 'success') {
        const loaded = (res.messages || []).map((m: any) => ({ role: m.role, content: m.content, timestamp: m.timestamp }))
        setMessages(loaded)
      } else {
        setMessages([])
      }
    } catch (e) {
      console.error('Failed to get trading session', e)
      setMessages([])
    }
  }

  const createSession = async () => {
    try {
      const res = await apiClient.tradingCreateSession()
      setSessionId(res.session_id)
      setMessages([])
      await loadSessions()
      toast.success('New trading session created')
    } catch (e) {
      console.error('Failed to create trading session', e)
      toast.error('Failed to create session')
    }
  }

  const deleteSession = async (sessionIdToDelete: string) => {
    try {
      await apiClient.tradingDeleteSession(sessionIdToDelete)
      
      // If we're deleting the current session, clear it
      if (sessionId === sessionIdToDelete) {
        setSessionId(null)
        setMessages([])
      }
      
      // Reload sessions list
      await loadSessions()
      toast.success('Trading session deleted')
    } catch (e) {
      console.error('Failed to delete trading session', e)
      toast.error('Failed to delete session')
    }
  }

  // Auto-expand textarea function
  const autoExpandTextarea = (textarea: HTMLTextAreaElement) => {
    textarea.style.height = 'auto'
    textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px' // Max height of 200px
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value)
    autoExpandTextarea(e.target)
  }

  const send = async () => {
    if (!input.trim()) return
    if (!sessionId) {
      await createSession()
    }
    const sid = sessionId
    const userMsg: TradingMessage = { role: 'user', content: input }
    setMessages(prev => [...prev, userMsg, { role: 'assistant', content: 'Processing...' }])
    setInput('')
    // Reset textarea height after sending
    if (textareaRef) {
      textareaRef.style.height = 'auto'
    }
    setIsLoading(true)
    try {
      const res = await apiClient.tradingChat(userMsg.content, sid || undefined)
      const assistantText = res.response || 'No response'
      setSessionId(res.session_id || sid)
      setMessages(prev => {
        const copy = [...prev]
        // replace the last placeholder assistant message
        for (let i = copy.length - 1; i >= 0; i--) {
          if (copy[i].role === 'assistant' && copy[i].content === 'Processing...') {
            copy[i] = { role: 'assistant', content: assistantText }
            break
          }
        }
        return copy
      })
      await loadSessions()
    } catch (e) {
      console.error('Trading chat failed', e)
      toast.error('Trading chat failed')
      setMessages(prev => prev.map(m => (m.role === 'assistant' && m.content === 'Processing...') ? { role: 'assistant', content: 'Error generating response' } : m))
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    loadSessions()
  }, [])

  return (
    <div className="flex h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      {/* Sidebar with collapse functionality */}
      <aside className={`${isSidebarCollapsed ? 'w-16' : 'w-80'} transition-all duration-300 ease-in-out border-r border-indigo-200/50 bg-gradient-to-b from-indigo-50/90 to-purple-50/90 backdrop-blur-xl shadow-2xl`}>
        <div className="p-4 space-y-4">
          {/* Sidebar Toggle Button */}
          <div className="flex items-center justify-between">
            {!isSidebarCollapsed && (
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-gradient-to-r from-indigo-600 to-purple-600 rounded-lg flex items-center justify-center shadow-lg">
                  <span className="text-white text-sm font-bold">📈</span>
                </div>
                <h3 className="text-lg font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
                  Trading Sessions
                </h3>
              </div>
            )}
            <button
              onClick={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
              className="p-2 rounded-lg bg-white/80 hover:bg-white shadow-md transition-all duration-200 hover:scale-105"
            >
              <span className="text-gray-600 text-lg">
                {isSidebarCollapsed ? '→' : '←'}
              </span>
            </button>
          </div>

          {/* New Session Button */}
          <button 
            onClick={createSession} 
            className={`w-full bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white py-3 rounded-xl shadow-lg transition-all duration-200 hover:scale-105 hover:shadow-xl flex items-center justify-center space-x-2 ${isSidebarCollapsed ? 'px-2' : 'px-4'}`}
          >
            <span className="text-lg">📈</span>
            {!isSidebarCollapsed && <span className="font-medium">New Trading Session</span>}
          </button>

          {/* Sessions List */}
          {!isSidebarCollapsed && (
            <>
              <div className="text-sm text-indigo-600 font-semibold flex items-center space-x-2">
                <span>📊</span>
                <span>Active Sessions</span>
              </div>
              <div className="space-y-2 overflow-y-auto max-h-[70vh] pr-2 scrollbar-thin scrollbar-thumb-blue-300 scrollbar-track-transparent">
          {sessions.length === 0 && (
                  <div className="text-gray-400 text-sm text-center py-4 bg-white/50 rounded-lg">
                    No sessions yet
                  </div>
          )}
                {sessions.map(s => (
                  <div key={s.id} className="group relative">
                    <button 
                      onClick={() => switchSession(s.id)} 
                      className={`w-full text-left px-4 py-3 rounded-xl transition-all duration-200 hover:scale-105 ${
                        sessionId === s.id 
                          ? 'bg-gradient-to-r from-indigo-100 to-purple-100 text-indigo-800 border-2 border-indigo-300 shadow-lg' 
                          : 'hover:bg-white/80 text-gray-700 hover:shadow-md border border-transparent hover:border-indigo-200'
                      }`}
                    >
                      <div className="truncate font-medium flex items-center space-x-2">
                        <span className="text-sm">📈</span>
                        <span>{s.title || 'Trading Session'}</span>
                      </div>
                      <div className="text-xs text-indigo-400 mt-1">
                        {s.updated_at ? new Date(s.updated_at).toLocaleString() : ''}
                      </div>
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        if (confirm('Delete this trading session?')) {
                          deleteSession(s.id)
                        }
                      }}
                      className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 w-6 h-6 rounded-full text-xs bg-red-100 text-red-600 hover:bg-red-200 transition-all duration-200 flex items-center justify-center shadow-md hover:shadow-lg"
                      title="Delete session"
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col">
        {/* Header */}
        <header className="p-6 border-b border-white/20 bg-gradient-to-r from-indigo-50 to-purple-50 backdrop-blur-xl shadow-lg">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent flex items-center gap-3">
                <span className="text-2xl">📈</span>
                Broadaxis Trading Planner
              </h2>
              <p className="text-sm text-gray-600 mt-1 flex items-center space-x-2">
                <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></span>
                <span>AI-Powered Trading Analysis • Options Strategies • Earnings Plays</span>
              </p>
            </div>
            
            {/* Trading Quick Actions */}
            <div className="flex items-center space-x-3">
              <div className="flex items-center space-x-2 bg-white/80 px-4 py-2 rounded-lg border border-indigo-200">
                <span className="text-sm font-medium text-indigo-700">Market Status:</span>
                <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></span>
                <span className="text-sm text-green-600 font-medium">Open</span>
              </div>
              <button className="px-4 py-2 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-lg hover:from-indigo-700 hover:to-purple-700 transition-all duration-200 shadow-lg hover:shadow-xl text-sm font-medium">
                📊 Market Data
              </button>
            </div>
          </div>
        </header>

        {/* Chat Messages Area */}
        <section className="flex-1 overflow-y-auto p-6 space-y-6 bg-gradient-to-b from-transparent to-blue-50/30">
          {messages.length === 0 ? (
              <div className="text-center mt-20">
                <div className="w-24 h-24 bg-gradient-to-r from-indigo-100 to-purple-100 rounded-full flex items-center justify-center mx-auto mb-6 shadow-lg">
                  <span className="text-4xl">📈</span>
                </div>
                <h3 className="text-xl font-semibold text-gray-700 mb-2">Ready to Start Trading Analysis?</h3>
                <p className="text-gray-500 mb-6">Create a new session and begin your AI-powered trading journey</p>
                
                {/* Trading Quick Start Cards */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 max-w-4xl mx-auto">
                  <div className="bg-gradient-to-r from-green-50 to-emerald-50 p-4 rounded-lg border border-green-200 hover:shadow-md transition-shadow">
                    <div className="text-2xl mb-2">📊</div>
                    <h4 className="font-semibold text-green-800 mb-1">Earnings Analysis</h4>
                    <p className="text-sm text-green-600">Analyze upcoming earnings and options strategies</p>
                  </div>
                  <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-4 rounded-lg border border-blue-200 hover:shadow-md transition-shadow">
                    <div className="text-2xl mb-2">⚡</div>
                    <h4 className="font-semibold text-blue-800 mb-1">Options Strategies</h4>
                    <p className="text-sm text-blue-600">Get AI-powered options recommendations</p>
                  </div>
                  <div className="bg-gradient-to-r from-purple-50 to-pink-50 p-4 rounded-lg border border-purple-200 hover:shadow-md transition-shadow">
                    <div className="text-2xl mb-2">🎯</div>
                    <h4 className="font-semibold text-purple-800 mb-1">Market Insights</h4>
                    <p className="text-sm text-purple-600">Real-time market analysis and trends</p>
                  </div>
                </div>
              </div>
            ) : (
              <div className="space-y-6">
                {messages.map((m, idx) => (
                  <div key={idx} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'} animate-fadeIn`}>
                    <div className={`max-w-4xl w-full px-6 py-4 rounded-2xl shadow-lg transition-all duration-200 hover:shadow-xl ${
                      m.role === 'user' 
                        ? 'bg-gradient-to-r from-blue-500 to-indigo-500 text-white border-2 border-blue-400' 
                        : 'bg-white/95 border border-gray-200/50 backdrop-blur-sm'
                    }`}>
                  {m.role === 'assistant' ? (
                        <ReactMarkdown 
                          remarkPlugins={[remarkGfm]}
                      components={{
                            a: ({href, children}) => (
                              <a 
                                href={href as string} 
                                target="_blank" 
                                rel="noreferrer" 
                                className="underline text-blue-600 hover:text-blue-800 transition-colors duration-200"
                              >
                                {children}
                              </a>
                            ),
                            code: ({children}) => (
                              <code className="bg-gray-100 text-gray-800 px-2 py-1 rounded text-sm font-mono">
                                {children}
                              </code>
                            ),
                            h1: ({children}) => (
                              <h1 className="text-3xl font-bold text-gray-900 mb-6 mt-8 first:mt-0 border-b-2 border-blue-200 pb-3 flex items-center gap-3">
                                <span className="text-2xl">📊</span>
                                {children}
                              </h1>
                            ),
                            h2: ({children}) => (
                              <h2 className="text-2xl font-bold text-blue-800 mb-4 mt-6 first:mt-0 flex items-center gap-3 bg-blue-50 p-4 rounded-lg border border-blue-200">
                                <span className="text-xl">📈</span>
                                {children}
                              </h2>
                            ),
                            h3: ({children}) => (
                              <h3 className="text-xl font-semibold text-gray-800 mb-3 mt-5 first:mt-0 flex items-center gap-2">
                                <span className="text-lg">🎯</span>
                                {children}
                              </h3>
                            ),
                            h4: ({children}) => (
                              <h4 className="text-lg font-semibold text-gray-700 mb-2 mt-4 first:mt-0 flex items-center gap-2">
                                <span className="text-base">💡</span>
                                {children}
                              </h4>
                            ),
                            table: ({children}) => (
                              <div className="overflow-x-auto my-6 rounded-lg border border-gray-200 shadow-lg">
                                <table className="min-w-full divide-y divide-gray-200 bg-white">
                                  {children}
                                </table>
                              </div>
                            ),
                            thead: ({children}) => (
                              <thead className="bg-gradient-to-r from-blue-600 to-indigo-600">
                                {children}
                              </thead>
                            ),
                            tbody: ({children}) => (
                              <tbody className="bg-white divide-y divide-gray-200">
                                {children}
                              </tbody>
                            ),
                            tr: ({children}) => (
                              <tr className="hover:bg-blue-50 transition-colors duration-200">
                                {children}
                              </tr>
                            ),
                            th: ({children}) => (
                              <th className="px-6 py-4 text-left text-xs font-bold text-white uppercase tracking-wider">
                                {children}
                              </th>
                            ),
                            td: ({children}) => (
                              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                                {children}
                              </td>
                            ),
                            p: ({children}) => (
                              <p className="text-gray-700 leading-relaxed mb-4 text-base">
                                {children}
                              </p>
                            ),
                            ul: ({children}) => (
                              <ul className="list-disc list-inside space-y-2 mb-4 text-gray-700 bg-gray-50 p-4 rounded-lg border border-gray-200">
                                {children}
                              </ul>
                            ),
                            ol: ({children}) => (
                              <ol className="list-decimal list-inside space-y-2 mb-4 text-gray-700 bg-gray-50 p-4 rounded-lg border border-gray-200">
                                {children}
                              </ol>
                            ),
                            li: ({children}) => (
                              <li className="leading-relaxed">
                                {children}
                              </li>
                            ),
                            strong: ({children}) => (
                              <strong className="font-bold text-gray-900 bg-yellow-100 px-2 py-1 rounded">
                                {children}
                              </strong>
                            ),
                            blockquote: ({children}) => (
                              <blockquote className="border-l-4 border-blue-500 pl-4 py-2 bg-blue-50 rounded-r-lg mb-4 italic text-gray-700">
                                {children}
                              </blockquote>
                            )
                          }}
                          className="prose prose-lg max-w-none trading-analysis-content"
                        >
                          {m.content}
                        </ReactMarkdown>
                      ) : (
                        <div className="text-white whitespace-pre-wrap font-medium">{m.content}</div>
                  )}
                </div>
                  </div>
                ))}
              </div>
          )}
        </section>

        {/* Input Footer */}
          <footer className="p-6 border-t border-white/20 bg-gradient-to-r from-indigo-50/80 to-purple-50/80 backdrop-blur-xl shadow-lg">
            <div className="flex items-center space-x-4">
              <div className="flex-1 relative">
                <textarea
                  ref={setTextareaRef}
                  value={input} 
                  onChange={handleInputChange} 
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault()
                      send()
                    }
                  }} 
                  placeholder="Enter trading symbols, analysis requests, or market questions..." 
                  className="w-full px-6 py-4 border-2 border-indigo-200 rounded-2xl bg-white/90 focus:border-indigo-500 focus:ring-4 focus:ring-indigo-100 transition-all duration-200 text-gray-700 placeholder-gray-400 resize-none min-h-[3rem] max-h-[12rem] overflow-y-auto" 
                  disabled={isLoading}
                  rows={1}
                  style={{ height: 'auto' }}
                />
                {isLoading && (
                  <div className="absolute right-4 top-1/2 transform -translate-y-1/2">
                    <div className="w-6 h-6 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin"></div>
                  </div>
                )}
              </div>
              <button 
                onClick={send} 
                disabled={isLoading || !input.trim()} 
                className="px-8 py-4 rounded-2xl bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white font-medium shadow-lg transition-all duration-200 hover:scale-105 hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 flex items-center space-x-2"
              >
                <span className="text-lg">📈</span>
                <span>Analyze</span>
              </button>
            </div>
        </footer>
      </main>
    </div>
  )
}

export default TradingPlanner


