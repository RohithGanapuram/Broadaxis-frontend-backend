import React, { useState, useEffect } from 'react'
import toast from 'react-hot-toast'
import { apiClient } from '../utils/api'

interface GeneratedFile {
  filename: string
  file_size: number
  modified_at: string
  type: string
}

const FileManager: React.FC = () => {
  const [generatedFiles, setGeneratedFiles] = useState<GeneratedFile[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [selectedFile, setSelectedFile] = useState<GeneratedFile | null>(null)

  useEffect(() => {
    loadGeneratedFiles()
  }, [])

  const loadGeneratedFiles = async () => {
    setIsLoading(true)
    
    try {
      const response = await apiClient.listFiles()
      if (response.status === 'success' && response.files) {
        setGeneratedFiles(response.files)
      } else {
        console.log('No files found or API error')
        setGeneratedFiles([])
      }
    } catch (error) {
      console.error('Error loading files:', error)
      toast.error('Failed to load files')
      setGeneratedFiles([])
    } finally {
      setIsLoading(false)
    }
  }

  const getFileIcon = (fileType: string) => {
    switch (fileType.toLowerCase()) {
      case 'pdf': return '📄'
      case 'docx':
      case 'doc': return '📝'
      case 'txt':
      case 'md': return '📋'
      case 'json': return '📊'
      case 'csv': return '📈'
      default: return '📁'
    }
  }

  const formatFileSize = (bytes: number) => {
    if (bytes > 1024 * 1024) {
      return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
    } else if (bytes > 1024) {
      return `${(bytes / 1024).toFixed(1)} KB`
    }
    return `${bytes} bytes`
  }

  const handleDownload = (filename: string) => {
    toast.loading(`Downloading ${filename}...`, { id: filename })
    
    // Create download link to backend API
    const element = document.createElement('a')
    element.href = `http://localhost:8000/api/files/${filename}`
    element.download = filename
    element.target = '_blank'
    document.body.appendChild(element)
    element.click()
    document.body.removeChild(element)
    
    toast.success(`${filename} downloaded`, { id: filename })
  }

  const handleDelete = (filename: string) => {
    if (!confirm(`Are you sure you want to delete ${filename}?`)) {
      return
    }

    // Mock delete operation
    setGeneratedFiles(prev => prev.filter(f => f.filename !== filename))
    if (selectedFile?.filename === filename) {
      setSelectedFile(null)
    }
    toast.success(`${filename} deleted`)
  }

  return (
    <div className="w-full p-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">📁 Generated Files</h1>
        <p className="text-gray-600">Manage files created by BroadAxis-AI</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Files List */}
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-white rounded-lg shadow">
            <div className="px-6 py-4 border-b border-gray-200 flex justify-between items-center">
              <h2 className="text-lg font-semibold text-gray-900">
                Generated Files ({generatedFiles.length})
              </h2>
              <button
                onClick={loadGeneratedFiles}
                disabled={isLoading}
                className="px-3 py-1 text-sm bg-blue-100 text-blue-700 rounded hover:bg-blue-200 transition-colors disabled:opacity-50"
              >
                🔄 Refresh
              </button>
            </div>
            
            <div className="divide-y divide-gray-200">
              {isLoading ? (
                <div className="px-6 py-8 text-center text-gray-500">
                  <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600 mx-auto mb-2"></div>
                  <p>Loading files...</p>
                </div>
              ) : generatedFiles.length === 0 ? (
                <div className="px-6 py-8 text-center text-gray-500">
                  <div className="text-4xl mb-2">📄</div>
                  <p>No generated files yet</p>
                  <p className="text-sm">Files created by AI tools will appear here</p>
                </div>
              ) : (
                generatedFiles.map((file, index) => (
                  <div
                    key={index}
                    className={`px-6 py-4 hover:bg-gray-50 cursor-pointer transition-colors ${
                      selectedFile?.filename === file.filename ? 'bg-blue-50' : ''
                    }`}
                    onClick={() => setSelectedFile(file)}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <span className="text-2xl">{getFileIcon(file.type)}</span>
                        <div>
                          <p className="text-sm font-medium text-gray-900">
                            {file.filename}
                          </p>
                          <p className="text-sm text-gray-500">
                            {formatFileSize(file.file_size)} • {new Date(file.modified_at).toLocaleDateString()}
                          </p>
                        </div>
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            handleDownload(file.filename)
                          }}
                          className="px-3 py-1 text-sm bg-green-100 text-green-700 rounded hover:bg-green-200 transition-colors"
                        >
                          ⬇️ Download
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            handleDelete(file.filename)
                          }}
                          className="px-3 py-1 text-sm bg-red-100 text-red-700 rounded hover:bg-red-200 transition-colors"
                        >
                          🗑️ Delete
                        </button>
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* File Actions */}
          {generatedFiles.length > 0 && (
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">🛠️ Bulk Actions</h2>
              <div className="flex space-x-3">
                <button
                  onClick={() => {
                    generatedFiles.forEach(file => handleDownload(file.filename))
                  }}
                  className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 transition-colors"
                >
                  📦 Download All
                </button>
                <button
                  onClick={() => {
                    if (confirm('Are you sure you want to delete all files?')) {
                      setGeneratedFiles([])
                      setSelectedFile(null)
                      toast.success('All files deleted')
                    }
                  }}
                  className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 transition-colors"
                >
                  🗑️ Delete All
                </button>
              </div>
            </div>
          )}
        </div>

        {/* File Preview */}
        <div className="lg:col-span-1">
          <div className="bg-white rounded-lg shadow">
            <div className="px-6 py-4 border-b border-gray-200">
              <h2 className="text-lg font-semibold text-gray-900">File Details</h2>
            </div>
            
            <div className="p-6">
              {selectedFile ? (
                <div className="space-y-4">
                  <div className="flex items-center space-x-3">
                    <span className="text-3xl">{getFileIcon(selectedFile.type)}</span>
                    <div>
                      <p className="font-medium text-gray-900">{selectedFile.filename}</p>
                      <p className="text-sm text-gray-500">
                        {formatFileSize(selectedFile.file_size)}
                      </p>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-1 gap-3 text-sm">
                    <div>
                      <p className="text-gray-500">File Type</p>
                      <p className="font-medium">{selectedFile.type.toUpperCase()}</p>
                    </div>
                    <div>
                      <p className="text-gray-500">Size</p>
                      <p className="font-medium">{formatFileSize(selectedFile.file_size)}</p>
                    </div>
                    <div>
                      <p className="text-gray-500">Modified</p>
                      <p className="font-medium">{new Date(selectedFile.modified_at).toLocaleString()}</p>
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <button
                      onClick={() => handleDownload(selectedFile.filename)}
                      className="w-full px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
                    >
                      ⬇️ Download File
                    </button>
                    <button
                      onClick={() => handleDelete(selectedFile.filename)}
                      className="w-full px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 transition-colors"
                    >
                      🗑️ Delete File
                    </button>
                  </div>
                </div>
              ) : (
                <div className="text-center text-gray-500">
                  <div className="text-4xl mb-2">👁️</div>
                  <p>Select a file to view details</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default FileManager