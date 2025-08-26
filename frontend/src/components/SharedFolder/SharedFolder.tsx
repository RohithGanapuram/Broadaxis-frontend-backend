import React, { useState, useEffect } from 'react'
import { useAuth } from '../../context/AuthContext'
import { apiClient } from '../../utils/api'

interface SharedFile {
  filename: string
  file_size: number
  modified_at: string
  type: string
  web_url?: string
  download_url?: string
  path: string
  id?: string
}

const SharedFolder: React.FC = () => {
  const [files, setFiles] = useState<SharedFile[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [currentPath, setCurrentPath] = useState<string>('')
  const [pathHistory, setPathHistory] = useState<string[]>([''])
  const [fileCache, setFileCache] = useState<Record<string, SharedFile[]>>({})
  const [lastFetchTime, setLastFetchTime] = useState<Record<string, number>>({})
  const { currentUser } = useAuth()
  
  const CACHE_DURATION = 5 * 60 * 1000 // 5 minutes

  useEffect(() => {
    loadFiles(currentPath)
  }, [currentPath])

  const loadFiles = async (folderPath: string = '', forceRefresh: boolean = false) => {
    const cacheKey = folderPath || 'root'
    const now = Date.now()
    
    // Check if we have cached data that's still fresh
    if (!forceRefresh && fileCache[cacheKey] && lastFetchTime[cacheKey]) {
      const timeSinceLastFetch = now - lastFetchTime[cacheKey]
      if (timeSinceLastFetch < CACHE_DURATION) {
        setFiles(fileCache[cacheKey])
        return
      }
    }
    
    setIsLoading(true)
    try {
      const data = await apiClient.listSharePointFiles(folderPath)
      
      if (data.status === 'success') {
        const filesData = data.files || []
        setFiles(filesData)
        
        // Update cache
        setFileCache(prev => ({ ...prev, [cacheKey]: filesData }))
        setLastFetchTime(prev => ({ ...prev, [cacheKey]: now }))
      } else {
        console.error('Error loading files:', data.message)
        setFiles([])
      }
    } catch (error) {
      console.error('Error loading files:', error)
      setFiles([])
    } finally {
      setIsLoading(false)
    }
  }

  const handleFolderClick = (folder: SharedFile) => {
    if (folder.type === 'folder') {
      const newPath = currentPath ? `${currentPath}/${folder.filename}` : folder.filename
      setCurrentPath(newPath)
      setPathHistory([...pathHistory, newPath])
    }
  }

  const handleBackClick = () => {
    if (pathHistory.length > 1) {
      const newHistory = pathHistory.slice(0, -1)
      const previousPath = newHistory[newHistory.length - 1]
      setPathHistory(newHistory)
      setCurrentPath(previousPath)
    }
  }

  const handleBreadcrumbClick = (index: number) => {
    const newHistory = pathHistory.slice(0, index + 1)
    const targetPath = newHistory[newHistory.length - 1]
    setPathHistory(newHistory)
    setCurrentPath(targetPath)
  }

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const getFileIcon = (file: SharedFile): string => {
    if (file.type === 'folder') return '📁'

    const ext = file.filename.split('.').pop()?.toLowerCase()
    switch (ext) {
      case 'pdf': return '📄'
      case 'docx':
      case 'doc': return '📝'
      case 'txt': return '📃'
      case 'md': return '📋'
      case 'xlsx':
      case 'xls': return '📊'
      case 'pptx':
      case 'ppt': return '📈'
      case 'jpg':
      case 'jpeg':
      case 'png':
      case 'gif': return '🖼️'
      default: return '📄'
    }
  }

  const getBreadcrumbs = () => {
    if (!currentPath) return ['SharePoint']

    const parts = currentPath.split('/')
    const breadcrumbs = ['SharePoint']

    parts.forEach((part, index) => {
      if (part) {
        breadcrumbs.push(part)
      }
    })

    return breadcrumbs
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">SharePoint Files</h1>
          <p className="text-gray-600 mt-1">Team shared documents and folders - Welcome back, {currentUser?.name}</p>
        </div>
        <div className="flex gap-2">
          {currentPath && (
            <button
              onClick={handleBackClick}
              className="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg transition-colors"
            >
              ← Back
            </button>
          )}
          <button
            onClick={() => loadFiles(currentPath, true)}
            disabled={isLoading}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors disabled:opacity-50"
          >
            {isLoading ? '🔄 Loading...' : '🔄 Refresh'}
          </button>
        </div>
      </div>

      {/* Breadcrumb Navigation */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
        <div className="flex items-center space-x-2 text-sm overflow-x-auto">
          <span className="text-gray-500 flex-shrink-0">📍</span>
          <div className="flex items-center space-x-2 min-w-0 flex-1">
            {getBreadcrumbs().map((crumb, index) => (
              <React.Fragment key={index}>
                {index > 0 && <span className="text-gray-400 flex-shrink-0">/</span>}
                <button
                  onClick={() => handleBreadcrumbClick(index)}
                  className={`hover:text-blue-600 transition-colors truncate max-w-xs ${
                    index === getBreadcrumbs().length - 1
                      ? 'text-blue-600 font-medium'
                      : 'text-gray-700 hover:underline'
                  }`}
                  title={crumb}
                >
                  {crumb.length > 20 ? `${crumb.substring(0, 17)}...` : crumb}
                </button>
              </React.Fragment>
            ))}
          </div>
        </div>
      </div>

      {files.length > 0 ? (
        <div className="bg-white rounded-xl shadow-lg border border-gray-100 p-6">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-6 gap-4">
            <div className="min-w-0 flex-1">
              <h2 className="text-xl font-bold text-gray-900 flex items-center">
                <span className="mr-2 flex-shrink-0">📁</span>
                <span className="truncate">
                  {currentPath
                    ? `${currentPath.split('/').pop()} (${files.length} items)`
                    : `SharePoint Files (${files.length} items)`
                  }
                </span>
              </h2>
              {currentPath && (
                <p className="text-sm text-gray-500 mt-1 truncate">
                  📂 {currentPath}
                </p>
              )}
            </div>
            <div className="text-sm text-gray-500 flex-shrink-0">
              <div className="flex gap-4">
                <span>📄 {files.filter(f => f.type !== 'folder').length} files</span>
                <span>📁 {files.filter(f => f.type === 'folder').length} folders</span>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {files.map((file, index) => (
              <div
                key={index}
                className={`border border-gray-200 rounded-lg p-4 transition-all duration-200 hover:border-blue-300 ${
                  file.type === 'folder'
                    ? 'hover:shadow-md cursor-pointer hover:bg-blue-50'
                    : 'hover:shadow-md'
                }`}
                onClick={() => file.type === 'folder' && handleFolderClick(file)}
              >
                <div className="flex flex-col space-y-3">
                  {/* Icon and Type */}
                  <div className="flex items-center justify-between">
                    <div className="text-3xl">{getFileIcon(file)}</div>
                    {file.type === 'folder' && (
                      <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded-full">
                        Folder
                      </span>
                    )}
                  </div>

                  {/* Filename */}
                  <div className="min-w-0">
                    <div
                      className="font-medium text-gray-900 text-sm leading-tight break-words"
                      title={file.filename}
                      style={{
                        wordBreak: 'break-word',
                        overflowWrap: 'break-word',
                        hyphens: 'auto'
                      }}
                    >
                      {file.filename}
                    </div>
                  </div>

                  {/* File Info */}
                  <div className="text-xs text-gray-500 space-y-1">
                    <div className="flex items-center justify-between">
                      <span>{file.type === 'folder' ? 'Folder' : formatFileSize(file.file_size)}</span>
                      {file.type !== 'folder' && (
                        <span className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded">
                          {file.type.toUpperCase()}
                        </span>
                      )}
                    </div>
                    <div className="text-xs text-gray-400">
                      {new Date(file.modified_at).toLocaleDateString()}
                    </div>
                  </div>

                  {/* Action Buttons */}
                  {file.type !== 'folder' && (
                    <div className="flex gap-2 pt-2 border-t border-gray-100">
                      {file.web_url && (
                        <a
                          href={file.web_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="flex-1 text-center bg-green-50 text-green-600 hover:bg-green-100 px-3 py-2 rounded text-xs font-medium transition-colors"
                          title="Open in SharePoint"
                        >
                          🌐 SharePoint
                        </a>
                      )}
                      {file.download_url && (
                        <a
                          href={file.download_url}
                          download
                          className="flex-1 text-center bg-blue-50 text-blue-600 hover:bg-blue-100 px-3 py-2 rounded text-xs font-medium transition-colors"
                          title="Download file"
                        >
                          ⬇️ Download
                        </a>
                      )}
                    </div>
                  )}

                  {/* Folder Click Hint */}
                  {file.type === 'folder' && (
                    <div className="text-xs text-blue-600 text-center pt-2 border-t border-blue-100">
                      📂 Click to open
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="bg-white rounded-xl shadow-lg border border-gray-100 p-8 text-center">
          <div className="text-6xl mb-4">📁</div>
          <h2 className="text-2xl font-bold text-gray-900 mb-2">
            {currentPath ? 'Empty Folder' : 'No Files Found'}
          </h2>
          <p className="text-gray-600 mb-4">
            {currentPath
              ? 'This folder is currently empty'
              : 'No files or folders found in SharePoint'
            }
          </p>
          <p className="text-sm text-gray-500">
            {currentPath
              ? 'Team members can add files to this folder in SharePoint'
              : 'Check your SharePoint connection or try refreshing'
            }
          </p>
        </div>
      )}

      <div className="bg-gradient-to-r from-blue-50 to-blue-100 rounded-xl shadow-lg border border-blue-200 p-6">
        <h3 className="text-lg font-bold text-gray-900 mb-4">SharePoint Integration</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-2xl mb-2">📁</div>
            <div className="text-sm font-medium text-gray-700">Folders</div>
            <div className="text-xs text-gray-500">Click to navigate</div>
          </div>
          <div className="text-center">
            <div className="text-2xl mb-2">📄</div>
            <div className="text-sm font-medium text-gray-700">Documents</div>
            <div className="text-xs text-gray-500">PDF, Word, Excel</div>
          </div>
          <div className="text-center">
            <div className="text-2xl mb-2">🌐</div>
            <div className="text-sm font-medium text-gray-700">SharePoint</div>
            <div className="text-xs text-gray-500">Open in browser</div>
          </div>
          <div className="text-center">
            <div className="text-2xl mb-2">⬇️</div>
            <div className="text-sm font-medium text-gray-700">Download</div>
            <div className="text-xs text-gray-500">Save locally</div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default SharedFolder