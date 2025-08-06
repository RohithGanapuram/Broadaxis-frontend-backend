import React, { useState, useEffect } from 'react'
import { useAuth } from '../../context/AuthContext'

interface SharedFile {
  name: string
  size: number
  modified: string
  path: string
}

const SharedFolder: React.FC = () => {
  const [files, setFiles] = useState<SharedFile[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const { currentUser } = useAuth()

  useEffect(() => {
    loadFiles()
  }, [])

  const loadFiles = async () => {
    setIsLoading(true)
    try {
      const response = await fetch('http://localhost:8000/api/files')
      const data = await response.json()
      if (data.status === 'success') {
        setFiles(data.files.map((file: any) => ({
          name: file.filename,
          size: file.file_size,
          modified: file.modified_at,
          path: `/api/files/${file.filename}`
        })))
      }
    } catch (error) {
      console.error('Error loading files:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const getFileIcon = (filename: string): string => {
    const ext = filename.split('.').pop()?.toLowerCase()
    switch (ext) {
      case 'pdf': return '📄'
      case 'docx':
      case 'doc': return '📝'
      case 'txt': return '📃'
      case 'md': return '📋'
      default: return '📁'
    }
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Shared Folder</h1>
          <p className="text-gray-600 mt-1">Generated files and documents - Welcome back, {currentUser?.name}</p>
        </div>
        <button
          onClick={loadFiles}
          disabled={isLoading}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors disabled:opacity-50"
        >
          {isLoading ? '🔄 Loading...' : '🔄 Refresh'}
        </button>
      </div>

      {files.length > 0 ? (
        <div className="bg-white rounded-xl shadow-lg border border-gray-100 p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-bold text-gray-900 flex items-center">
              <span className="mr-2">📁</span>
              Shared Files ({files.length})
            </h2>
            <div className="text-sm text-gray-500">
              Total: {formatFileSize(files.reduce((sum, file) => sum + file.size, 0))}
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {files.map((file, index) => (
              <div key={index} className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-all duration-200 hover:border-blue-300">
                <div className="flex items-start justify-between">
                  <div className="flex items-start space-x-3 flex-1">
                    <div className="text-2xl">{getFileIcon(file.name)}</div>
                    <div className="flex-1 min-w-0">
                      <div className="font-medium text-gray-900 truncate" title={file.name}>
                        {file.name}
                      </div>
                      <div className="text-sm text-gray-500 space-x-2">
                        <span>{formatFileSize(file.size)}</span>
                        <span>•</span>
                        <span>{new Date(file.modified).toLocaleDateString()}</span>
                      </div>
                      <div className="text-xs text-gray-400 mt-1">
                        {new Date(file.modified).toLocaleTimeString()}
                      </div>
                    </div>
                  </div>
                  <a
                    href={`http://localhost:8000${file.path}`}
                    download
                    className="text-blue-600 hover:text-blue-700 text-xl hover:scale-110 transition-transform"
                    title="Download file"
                  >
                    ⬇️
                  </a>
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="bg-white rounded-xl shadow-lg border border-gray-100 p-8 text-center">
          <div className="text-6xl mb-4">📁</div>
          <h2 className="text-2xl font-bold text-gray-900 mb-2">No Files Yet</h2>
          <p className="text-gray-600 mb-4">
            Generated files from AI tools will appear here
          </p>
          <p className="text-sm text-gray-500">
            Use the Chat interface to generate documents, PDFs, and other files
          </p>
        </div>
      )}

      <div className="bg-gradient-to-r from-blue-50 to-blue-100 rounded-xl shadow-lg border border-blue-200 p-6">
        <h3 className="text-lg font-bold text-gray-900 mb-4">Supported File Types</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-2xl mb-2">📄</div>
            <div className="text-sm font-medium text-gray-700">PDF Documents</div>
          </div>
          <div className="text-center">
            <div className="text-2xl mb-2">📝</div>
            <div className="text-sm font-medium text-gray-700">Word Documents</div>
          </div>
          <div className="text-center">
            <div className="text-2xl mb-2">📃</div>
            <div className="text-sm font-medium text-gray-700">Text Files</div>
          </div>
          <div className="text-center">
            <div className="text-2xl mb-2">📋</div>
            <div className="text-sm font-medium text-gray-700">Markdown Files</div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default SharedFolder