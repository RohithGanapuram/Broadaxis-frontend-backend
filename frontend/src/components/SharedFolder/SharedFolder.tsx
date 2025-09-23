import React, { useState, useEffect, useRef } from 'react'
import { useAuth } from '../../context/AuthContext'
import { useAppContext } from '../../context/AppContext'
import { apiClient } from '../../utils/api'
import toast from 'react-hot-toast'

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
  const [isUploading, setIsUploading] = useState(false)
  const [showUploadModal, setShowUploadModal] = useState(false)
  const [selectedFolder, setSelectedFolder] = useState<'RFP' | 'RFI' | 'RFQ' | ''>('')
  const fileInputRef = useRef<HTMLInputElement>(null)
  const { currentUser } = useAuth()
  const { sharePointCache, setSharePointCache } = useAppContext()
  
  const CACHE_DURATION = 5 * 60 * 1000 // 5 minutes

  useEffect(() => {
    loadFiles(currentPath)
  }, [currentPath])

  const loadFiles = async (folderPath: string = '', forceRefresh: boolean = false) => {
    const cacheKey = folderPath || 'root'
    const now = Date.now()
    
    // Check if we have global cached data that's still fresh and matches current path
    if (!forceRefresh && sharePointCache && sharePointCache.currentPath === cacheKey) {
      const timeSinceLastFetch = now - sharePointCache.lastFetchTime
      if (timeSinceLastFetch < CACHE_DURATION) {
        setFiles(sharePointCache.files)
        return
      }
    }
    
    setIsLoading(true)
    try {
      const data = await apiClient.listSharePointFiles(folderPath)

      if (data.status === 'success') {
        const filesData = data.files || []

        // Are we at Emails/<account> level?
        const isAccountLevel =
          !!folderPath &&
          folderPath.split('/').filter(Boolean).length === 2 &&
          folderPath.startsWith('Emails/')

        const toTime = (s?: string) => {
          if (!s) return 0
          const t = Date.parse(s)
          return Number.isNaN(t) ? 0 : t
        }

        const datePrefix = (name: string) => {
          const m = name.match(/^(\d{4}-\d{2}-\d{2})\b/)
          return m ? m[1] : null
        }

        // Sort newest → oldest. Folders first. At account level, use YYYY-MM-DD prefix.
        const sorted = [...filesData].sort((a, b) => {
          const af = a.type === 'folder'
          const bf = b.type === 'folder'
          if (af && !bf) return -1
          if (!af && bf) return 1

          if (isAccountLevel && af && bf) {
            const ad = datePrefix(a.filename)
            const bd = datePrefix(b.filename)
            if (ad && bd) return bd.localeCompare(ad) // newer (b) first
            if (ad) return -1
            if (bd) return 1
          }

          // fallback to modified_at (newer first)
          return toTime(b.modified_at) - toTime(a.modified_at)
        })

        setFiles(sorted)

        // Update global cache
        setSharePointCache({
          files: sorted,
          lastFetchTime: now,
          currentPath: cacheKey
        })


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

  const handleFolderUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = event.target.files
    if (!selectedFiles || selectedFiles.length === 0 || !selectedFolder) return

    setIsUploading(true)
    const toastId = toast.loading(`Uploading ${selectedFiles.length} files to ${selectedFolder} folder...`)

    try {
      const result = await apiClient.uploadFolder(selectedFiles, selectedFolder, 'default')
      
      if (result.status === 'success') {
        toast.success(`Successfully uploaded ${result.successful_uploads} files to ${selectedFolder} folder!`, { id: toastId })
      } else if (result.status === 'partial_success') {
        toast.success(`Uploaded ${result.successful_uploads}/${result.total_files} files to ${selectedFolder} folder`, { id: toastId })
        if (result.failed_uploads > 0) {
          toast.error(`${result.failed_uploads} files failed to upload`, { duration: 5000 })
        }
      } else {
        toast.error('Failed to upload folder', { id: toastId })
      }

      // Refresh the current folder to show uploaded files
      loadFiles(currentPath, true)
      
    } catch (error) {
      console.error('Folder upload error:', error)
      toast.error('Failed to upload folder. Please try again.', { id: toastId })
    } finally {
      setIsUploading(false)
      setShowUploadModal(false)
      setSelectedFolder('')
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
    }
  }

  const openUploadModal = (folder: 'RFP' | 'RFI' | 'RFQ') => {
    setSelectedFolder(folder)
    setShowUploadModal(true)
    if (fileInputRef.current) {
      fileInputRef.current.click()
    }
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">SharePoint Files</h1>
          <p className="text-gray-600 mt-1">Team shared documents and folders - Welcome back, {currentUser?.name}</p>
        </div>
        <div className="flex gap-2">
          {/* Folder Upload Buttons - Only show at root level */}
          {!currentPath && (
            <>
              <button
                onClick={() => openUploadModal('RFP')}
                disabled={isUploading}
                className="px-4 py-2 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white rounded-lg transition-all duration-200 shadow-md hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed font-medium"
                title="Upload folder to RFP directory"
              >
                📁 Upload to RFP
              </button>
              <button
                onClick={() => openUploadModal('RFI')}
                disabled={isUploading}
                className="px-4 py-2 bg-gradient-to-r from-indigo-600 to-indigo-700 hover:from-indigo-700 hover:to-indigo-800 text-white rounded-lg transition-all duration-200 shadow-md hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed font-medium"
                title="Upload folder to RFI directory"
              >
                📁 Upload to RFI
              </button>
              <button
                onClick={() => openUploadModal('RFQ')}
                disabled={isUploading}
                className="px-4 py-2 bg-gradient-to-r from-slate-600 to-slate-700 hover:from-slate-700 hover:to-slate-800 text-white rounded-lg transition-all duration-200 shadow-md hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed font-medium"
                title="Upload folder to RFQ directory"
              >
                📁 Upload to RFQ
              </button>
            </>
          )}
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

      {/* Hidden file input for folder uploads */}
      <input
        ref={fileInputRef}
        type="file"
        multiple
        webkitdirectory=""
        onChange={handleFolderUpload}
        style={{ display: 'none' }}
        accept=".pdf,.txt,.md,.docx,.doc,.xlsx,.xls,.pptx,.ppt"
      />

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
            <div className="flex items-center gap-4 flex-shrink-0">
              <div className="text-sm text-gray-500">
                <div className="flex gap-4">
                  <span>📄 {files.filter(f => f.type !== 'folder').length} files</span>
                  <span>📁 {files.filter(f => f.type === 'folder').length} folders</span>
                </div>
              </div>
              <button
                onClick={() => loadFiles(currentPath, true)}
                disabled={isLoading}
                className="px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors duration-200 shadow-sm hover:shadow-md disabled:opacity-50 disabled:cursor-not-allowed text-sm font-medium"
                title="Refresh folder contents"
              >
                🔄 Refresh
              </button>
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

      {/* Folder Upload Information */}
      {!currentPath && (
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl shadow-lg border border-blue-200 p-6">
          <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center">
            <span className="mr-2">📤</span>
            Folder Upload Feature
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center p-4 bg-white rounded-lg border border-blue-200 shadow-sm">
              <div className="text-2xl mb-2">📋</div>
              <div className="text-sm font-medium text-gray-700">RFP Folder</div>
              <div className="text-xs text-gray-500">Upload RFP documents and folders</div>
            </div>
            <div className="text-center p-4 bg-white rounded-lg border border-indigo-200 shadow-sm">
              <div className="text-2xl mb-2">📝</div>
              <div className="text-sm font-medium text-gray-700">RFI Folder</div>
              <div className="text-xs text-gray-500">Upload RFI documents and folders</div>
            </div>
            <div className="text-center p-4 bg-white rounded-lg border border-slate-200 shadow-sm">
              <div className="text-2xl mb-2">📊</div>
              <div className="text-sm font-medium text-gray-700">RFQ Folder</div>
              <div className="text-xs text-gray-500">Upload RFQ documents and folders</div>
            </div>
          </div>
                 <div className="mt-4 p-3 bg-white rounded-lg border border-blue-200 shadow-sm">
                   <p className="text-sm text-gray-600">
                     <strong>How to use:</strong> Click on any of the upload buttons above to select a folder from your local machine. 
                     The entire folder structure will be uploaded to the corresponding SharePoint directory (RFP, RFI, or RFQ) and will be available for processing in the chat interface.
                   </p>
                 </div>
        </div>
      )}
    </div>
  )
}

export default SharedFolder