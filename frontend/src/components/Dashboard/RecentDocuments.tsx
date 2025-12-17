import React, { useState, useEffect } from 'react';
import { useAuth } from '../../context/AuthContext';
import { useAppContext } from '../../context/AppContext';
import { apiClient } from '../../utils/api';
type UserLite = { id?: string; name: string; email?: string };

interface Task {
  id: string;
  category: string;
  type: string;
  title: string;
  document: string;
  status: string;
  assignedTo: string;
  assignedBy: string;
  priority: string;
  dueDate: string;
  decision: string;
  createdAt: string;
  updatedAt: string;
  parentTaskId?: string;  // For document tasks linked to parent RFP
  parentRfpPath?: string;  // e.g., "RFP/Dallas City"
  documentDetails?: string;  // Details about document requirement
  originalStatus?: string;  // ✅/🟡/❌ from AI
  documentCount?: number;  // For parent tasks - how many documents needed
}
 
const RecentDocuments: React.FC = () => {
  const { currentUser } = useAuth();
  const { tasksCache, updateTasksCache, clearTasksCache, usersCache, updateUsersCache } = useAppContext();
  const [showAddForm, setShowAddForm] = useState(false);
  const [newTask, setNewTask] = useState({
    category: 'Project',
    type: 'RFP',
    title: '',
    document: '',
    assignedTo: currentUser?.name || '',
    status: 'Assigned',
    priority: 'Medium',
    dueDate: '',
    decision: 'Decision Pending'
  });
  type UserLite = { id?: string; name: string; email?: string };

  // Fallback names if the API returns nothing or errors
  const FIXED_ASSIGNEES = ["Sakshi", "Rohith", "Masood", "Uzi", "Abhir", "Divya"];


  // Real users from API - initialize from cache if available
  const [availableUsers, setAvailableUsers] = useState<string[]>(() => {
    if (usersCache) {
      const cacheAge = Date.now() - usersCache.lastFetchTime;
      const CACHE_MAX_AGE = 10 * 60 * 1000; // 10 minutes for users
      if (cacheAge < CACHE_MAX_AGE) {
        console.log('✅ Initializing users from cache');
        return usersCache.users;
      }
    }
    return FIXED_ASSIGNEES; // Fallback to fixed list
  });
  const [loadingUsers, setLoadingUsers] = useState(false);

  // SharePoint folder browser state
  const [sharePointFolders, setSharePointFolders] = useState<any[]>([]);
  const [loadingFolders, setLoadingFolders] = useState(false);
  const [currentPath, setCurrentPath] = useState<string>('');
  const [pathHistory, setPathHistory] = useState<string[]>(['']);
  const [showFolderBrowser, setShowFolderBrowser] = useState(false);

  // Initialize tasks from cache immediately if available
  const [tasks, setTasks] = useState<Task[]>(() => {
    if (tasksCache) {
      const cacheAge = Date.now() - tasksCache.lastFetchTime;
      const CACHE_MAX_AGE = 5 * 60 * 1000; // 5 minutes
      if (cacheAge < CACHE_MAX_AGE) {
        console.log('✅ Initializing tasks from cache');
        return tasksCache.tasks;
      }
    }
    return [];
  });
  const [showCompletedTasks, setShowCompletedTasks] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [taskToDelete, setTaskToDelete] = useState<Task | null>(null);

  // 👇 Dropdown users list

// Load user names once when component mounts


  // Load SharePoint folders, users, and tasks when component mounts
  useEffect(() => {
    // Load tasks first (will use cache if available)
    loadTasks(false); // Don't force refresh on mount if cache exists
    
    // Load users and SharePoint folders in background (non-blocking)
    loadUsers();
    // Only load SharePoint folders if folder browser is shown
    // loadSharePointFolders(); // Load lazily when needed
    
    // Cleanup old completed tasks on component mount (only once per session)
    if (!tasksCache) {
      cleanupOldTasks();
    }
  }, []);

  const loadUsers = async (forceRefresh: boolean = false) => {
    // Check cache first (unless forcing refresh)
    if (!forceRefresh && usersCache) {
      const cacheAge = Date.now() - usersCache.lastFetchTime;
      const CACHE_MAX_AGE = 10 * 60 * 1000; // 10 minutes
      
      if (cacheAge < CACHE_MAX_AGE) {
        console.log('✅ Using cached users');
        setAvailableUsers(usersCache.users);
        return;
      }
    }
    
    setLoadingUsers(true);
    try {
      console.log('🔍 Fetching users from API...');
      const users = await apiClient.getUsers();
      console.log('🔍 API returned users:', users);
      console.log('🔍 Number of users:', users.length);
      
      // Extract user names from the API response
      const userNames = users.map((user: any) => user.name);
      const finalUsers = userNames.length ? userNames : FIXED_ASSIGNEES;
      setAvailableUsers(finalUsers);
      // Update cache
      updateUsersCache(finalUsers);
      console.log('✅ Loaded and cached users:', finalUsers.length);
    } catch (error) {
      console.error('❌ Failed to load users:', error);
      // Use cached users if available, otherwise fallback
      if (usersCache && usersCache.users.length > 0) {
        setAvailableUsers(usersCache.users);
      } else {
        setAvailableUsers(FIXED_ASSIGNEES);
      }
    } finally {
      setLoadingUsers(false);
    }
  };

  const onAssigneeChange = async (taskId: string, newName: string) => {
    try {
      await apiClient.updateTaskAssignee(taskId, newName)
      // Optimistic update:
      setTasks(prev => prev.map(t => t.id === taskId ? { ...t, assignedTo: newName } : t))
      // (Optional) await loadTasks() to re-sync from server
    } catch (e) {
      console.error('❌ Failed to update assignee:', e)
    }
};


  const loadTasks = async (forceRefresh: boolean = false) => {
    // Check cache first (unless forcing refresh)
    if (!forceRefresh && tasksCache) {
      const cacheAge = Date.now() - tasksCache.lastFetchTime;
      const CACHE_MAX_AGE = 5 * 60 * 1000; // 5 minutes
      
      // Use cache if it's fresh (less than 5 minutes old)
      if (cacheAge < CACHE_MAX_AGE) {
        console.log('✅ Using cached tasks (age:', Math.round(cacheAge / 1000), 'seconds)');
        setTasks(tasksCache.tasks);
        return;
      } else {
        console.log('⚠️ Cache expired, fetching fresh tasks...');
      }
    }
    
    try {
      console.log('🔍 Fetching tasks from API...');
      const tasks = await apiClient.getTasks();
      console.log('🔍 API returned tasks:', tasks);
      
      // Convert API response to Task format
      const taskList = tasks.map((task: any) => ({
        id: task.id,
        category: task.category,
        type: task.type,
        title: task.title,
        document: task.document,
        status: task.status,
        assignedTo: task.assigned_to,
        assignedBy: task.assigned_by,
        priority: task.priority,
        dueDate: task.due_date,
        decision: task.decision || 'Decision Pending',
        createdAt: task.created_at,
        updatedAt: task.updated_at,
        parentTaskId: task.parent_task_id,
        parentRfpPath: task.parent_rfp_path,
        documentDetails: task.document_details,
        originalStatus: task.original_status,
        documentCount: task.document_count
      }));
      
      setTasks(taskList);
      // Update cache
      updateTasksCache(taskList);
      console.log('✅ Loaded and cached tasks:', taskList.length);
    } catch (error) {
      console.error('❌ Failed to load tasks:', error);
      // If we have cached tasks, use them even if fetch failed
      if (tasksCache && tasksCache.tasks.length > 0) {
        console.log('⚠️ Using cached tasks as fallback');
        setTasks(tasksCache.tasks);
      } else {
        setTasks([]);
      }
    }
  };

  const cleanupOldTasks = async () => {
    try {
      console.log('🧹 Starting cleanup of old completed tasks...');
      const response = await apiClient.cleanupOldTasks();
      console.log('🧹 Cleanup result:', response);
      
      // Reload tasks after cleanup (force refresh)
      await loadTasks(true);
      
      // Show success message
      if (response.deleted_count > 0) {
        console.log(`✅ Cleaned up ${response.deleted_count} old completed tasks`);
      }
    } catch (error) {
      console.error('❌ Failed to cleanup old tasks:', error);
    }
  };

  const handleDeleteTask = (task: Task) => {
    setTaskToDelete(task);
    setShowDeleteModal(true);
  };

  const confirmDeleteTask = async () => {
    if (!taskToDelete) return;
    
    try {
      console.log('🗑️ Deleting task:', taskToDelete.title);
      await apiClient.deleteTask(taskToDelete.id);
      
      // Reload tasks after deletion (force refresh)
      await loadTasks(true);
      
      console.log(`✅ Successfully deleted task: ${taskToDelete.title}`);
    } catch (error) {
      console.error('❌ Failed to delete task:', error);
    } finally {
      setShowDeleteModal(false);
      setTaskToDelete(null);
    }
  };

  const cancelDeleteTask = () => {
    setShowDeleteModal(false);
    setTaskToDelete(null);
  };

  // Load folders when currentPath changes (for navigation) - lazy load only when needed
  useEffect(() => {
    if (showFolderBrowser) {
      loadSharePointFolders(currentPath);
    }
  }, [currentPath, showFolderBrowser]);

  const loadSharePointFolders = async (path: string = '') => {
    setLoadingFolders(true);
    try {
      console.log('Loading SharePoint folders for path:', path);
      const response = await apiClient.listSharePointFiles(path);
      console.log('SharePoint API response:', response);
      
      if (response.status === 'success' && response.files) {
        // Get all folders with their full information
        const folders = response.files.filter((file: any) => file.type === 'folder');
        console.log('Filtered folders:', folders);
        
        // If no folders found, show all items for debugging
        if (folders.length === 0) {
          console.log('No folders found, showing all items:', response.files);
          // For now, let's show all items to see what we're getting
          setSharePointFolders(response.files);
        } else {
          setSharePointFolders(folders);
        }
        setCurrentPath(path);
      } else {
        console.log('No files found or API error:', response);
        setSharePointFolders([]);
      }
    } catch (error) {
      console.error('Failed to load SharePoint folders:', error);
      setSharePointFolders([]);
    } finally {
      setLoadingFolders(false);
    }
  };

  const handleFolderClick = (folder: any) => {
    if (folder.type === 'folder') {
      const newPath = currentPath ? `${currentPath}/${folder.filename}` : folder.filename;
      
      // Optimistic update - immediately show loading state
      setLoadingFolders(true);
      setCurrentPath(newPath);
      setPathHistory([...pathHistory, newPath]);
      
      // Load folders will be triggered by useEffect
    }
  };

  const handleBackClick = () => {
    if (pathHistory.length > 1) {
      const newHistory = pathHistory.slice(0, -1);
      const previousPath = newHistory[newHistory.length - 1];
      
      // Optimistic update - immediately show loading state
      setLoadingFolders(true);
      setPathHistory(newHistory);
      setCurrentPath(previousPath);
      
      // Load folders will be triggered by useEffect
    }
  };

  const handleBreadcrumbClick = (index: number) => {
    const newHistory = pathHistory.slice(0, index + 1);
    const targetPath = newHistory[newHistory.length - 1];
    
    // Optimistic update - immediately show loading state
    setLoadingFolders(true);
    setPathHistory(newHistory);
    setCurrentPath(targetPath);
    
    // Load folders will be triggered by useEffect
  };

  const getBreadcrumbs = () => {
    return pathHistory.filter(path => path !== '').map(path => path.split('/').pop() || 'Root');
  };

  const handleAddTask = async () => {
    if (newTask.title && newTask.assignedTo) {
      try {
        console.log('🔍 Creating task assignment:', newTask);
        
        // Create task via API
        const taskData = {
          category: newTask.category,
          type: newTask.type,
          title: newTask.title,
          document: newTask.document,
          assigned_to: newTask.assignedTo,
          status: newTask.status,
          priority: newTask.priority,
          due_date: newTask.dueDate,
          decision: newTask.decision
        };
        
        const createdTask = await apiClient.createTask(taskData);
        console.log('✅ Task created:', createdTask);
        
        // Reload tasks to get the latest data (force refresh)
        await loadTasks(true);
        
        // Reset form
        setNewTask({
          category: 'Project',
          type: 'RFP',
          title: '',
          document: '',
          assignedTo: currentUser?.name || '',
          status: 'Assigned',
          priority: 'Medium',
          dueDate: '',
          decision: 'Decision Pending'
        });
        setShowAddForm(false);
        
      } catch (error) {
        console.error('❌ Failed to create task:', error);
        // You could add a toast notification here
      }
    }
  };

  const handleStatusChange = async (index: number, newStatus: string) => {
    const task = tasks[index];
    if (!task) return;
    
    try {
      console.log('🔍 Updating task status:', task.id, 'to', newStatus);
      
      // Update status via API
      await apiClient.updateTaskStatus(task.id, newStatus);
      console.log('✅ Task status updated');
      
      // Optimistically update local state and cache
      const updatedTasks = [...tasks];
      updatedTasks[index].status = newStatus;
      setTasks(updatedTasks);
      updateTasksCache(updatedTasks);
      
      // Reload tasks to get the latest data (force refresh in background)
      await loadTasks(true);
      
    } catch (error) {
      console.error('❌ Failed to update task status:', error);
      // Revert the local change on error
      const updatedTasks = [...tasks];
      updatedTasks[index].status = task.status; // Revert to original status
      setTasks(updatedTasks);
    }
  };

  const handleDecisionChange = async (index: number, newDecision: string) => {
    const task = tasks[index];
    if (!task) return;
    
    try {
      console.log('🔍 Updating task decision:', task.id, 'to', newDecision);
      
      // Update decision via API
      await apiClient.updateTaskDecision(task.id, newDecision);
      console.log('✅ Task decision updated');
      
      // Optimistically update local state and cache
      const updatedTasks = [...tasks];
      updatedTasks[index].decision = newDecision;
      setTasks(updatedTasks);
      updateTasksCache(updatedTasks);
      
      // Reload tasks to get the latest data (force refresh in background)
      await loadTasks(true);
      
    } catch (error) {
      console.error('❌ Failed to update task decision:', error);
      // Revert the local change on error
      const updatedTasks = [...tasks];
      updatedTasks[index].decision = task.decision; // Revert to original decision
      setTasks(updatedTasks);
    }
  };

  const getDecisionBadge = (decision: string) => {
    const baseClasses = "px-2 py-1 rounded-full text-xs font-medium";
    let decisionClasses = '';
    
    switch (decision) {
      case 'Go':
        decisionClasses = 'bg-green-100 text-green-700';
        break;
      case 'No-Go':
        decisionClasses = 'bg-red-100 text-red-700';
        break;
      case 'Decision Pending':
        decisionClasses = 'bg-yellow-100 text-yellow-700';
        break;
      default:
        decisionClasses = 'bg-gray-100 text-gray-700';
    }
    
    return `${baseClasses} ${decisionClasses}`;
  };
 
  const getStatusBadge = (status: string) => {
    const baseClasses = "px-3 py-1 rounded-full text-xs font-medium";
    let statusClasses = '';
    
    switch (status) {
      case 'Completed':
        statusClasses = 'bg-green-100 text-green-800';
        break;
      case 'In Progress':
        statusClasses = 'bg-blue-100 text-blue-800';
        break;
      case 'Review':
        statusClasses = 'bg-yellow-100 text-yellow-800';
        break;
      case 'Assigned':
        statusClasses = 'bg-purple-100 text-purple-800';
        break;
      case 'Pending':
        statusClasses = 'bg-gray-100 text-gray-800';
        break;
      default:
        statusClasses = 'bg-yellow-100 text-yellow-800';
    }
   
    const getStatusIcon = (status: string) => {
      switch (status) {
        case 'Completed': return '✅';
        case 'In Progress': return '🔄';
        case 'Review': return '👀';
        case 'Assigned': return '📋';
        case 'Pending': return '⏳';
        default: return '📋';
      }
    };
   
    return (
      <span className={`${baseClasses} ${statusClasses} flex items-center space-x-1`}>
        <span>{getStatusIcon(status)}</span>
        <span>{status}</span>
      </span>
    );
  };

  const getProgressBar = (status: string) => {
    let progress = 0;
    let color = '';
    
    switch (status) {
      case 'Completed':
        progress = 100;
        color = 'bg-green-500';
        break;
      case 'In Progress':
        progress = 75;
        color = 'bg-blue-500';
        break;
      case 'Review':
        progress = 50;
        color = 'bg-yellow-500';
        break;
      case 'Assigned':
        progress = 10;
        color = 'bg-purple-500';
        break;
      case 'Pending':
        progress = 25;
        color = 'bg-gray-500';
        break;
      default:
        progress = 0;
        color = 'bg-gray-500';
    }
    
    return (
      <div className="w-full bg-gray-200 rounded-full h-2">
        <div 
          className={`h-2 rounded-full ${color} transition-all duration-300`}
          style={{ width: `${progress}%` }}
        ></div>
      </div>
    );
  };

  // Calculate decision statistics
  const getDecisionStats = () => {
    const projectTasks = tasks.filter(task => 
      task.category === 'Project' && 
      (task.type === 'RFP' || task.type === 'RFI' || task.type === 'RFQ')
    );
    
    const goCount = projectTasks.filter(task => task.decision === 'Go').length;
    const noGoCount = projectTasks.filter(task => task.decision === 'No-Go').length;
    const pendingCount = projectTasks.filter(task => task.decision === 'Decision Pending').length;
    const totalProjectTasks = projectTasks.length;
    
    return { goCount, noGoCount, pendingCount, totalProjectTasks };
  };

  // Separate active and completed tasks
  const activeTasks = tasks.filter(task => task.status !== 'Completed');
  const completedTasks = tasks.filter(task => task.status === 'Completed');
 
  return (
    <div className="relative overflow-hidden">
      {/* Background with animated gradient */}
      <div className="absolute inset-0 bg-gradient-to-br from-slate-900 via-blue-900 to-indigo-900 opacity-95"></div>
      <div className="absolute inset-0 bg-gradient-to-tr from-amber-500/10 via-transparent to-orange-500/10"></div>
      
      {/* Animated background elements */}
      <div className="absolute top-0 right-0 w-96 h-96 bg-gradient-to-br from-blue-400/20 to-purple-600/20 rounded-full blur-3xl transform translate-x-32 -translate-y-32"></div>
      <div className="absolute bottom-0 left-0 w-80 h-80 bg-gradient-to-tr from-amber-400/20 to-orange-500/20 rounded-full blur-3xl transform -translate-x-24 translate-y-24"></div>
      
      <div className="relative bg-white/95 backdrop-blur-sm rounded-2xl shadow-2xl border border-white/20 p-8">
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-4">
              <div className="p-3 bg-gradient-to-br from-blue-600 to-indigo-700 rounded-xl shadow-lg">
                <span className="text-2xl">🎯</span>
              </div>
              <div>
                <h2 className="text-2xl font-bold bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent">
                  Broadaxis Task Management
                </h2>
              </div>
            </div>
            <div className="flex items-center space-x-3">
              <button
                onClick={loadTasks}
                className="group px-4 py-2.5 bg-white/80 hover:bg-white border border-gray-200 hover:border-gray-300 text-gray-700 rounded-xl text-sm font-semibold transition-all duration-200 shadow-sm hover:shadow-md"
                title="Refresh tasks"
              >
                <span className="flex items-center space-x-2">
                  <span className="group-hover:rotate-180 transition-transform duration-300">🔄</span>
                  <span>Refresh</span>
                </span>
              </button>
              {completedTasks.length > 0 && (
                <button
                  onClick={cleanupOldTasks}
                  className="group px-4 py-2.5 bg-white/80 hover:bg-white border border-gray-200 hover:border-gray-300 text-gray-700 rounded-xl text-sm font-semibold transition-all duration-200 shadow-sm hover:shadow-md"
                  title="Delete completed tasks older than 7 days"
                >
                  <span className="flex items-center space-x-2">
                    <span>Cleanup</span>
                  </span>
                </button>
              )}
              {completedTasks.length > 0 && (
                <button
                  onClick={() => setShowCompletedTasks(!showCompletedTasks)}
                  className={`group px-4 py-2.5 rounded-xl text-sm font-semibold transition-all duration-200 shadow-sm hover:shadow-md ${
                    showCompletedTasks 
                      ? 'bg-gradient-to-r from-green-500 to-emerald-500 text-white' 
                      : 'bg-white/80 hover:bg-white border border-gray-200 hover:border-gray-300 text-gray-700'
                  }`}
                  title={showCompletedTasks ? "Hide completed tasks" : "Show completed tasks"}
                >
                  <span className="flex items-center space-x-2">
                    <span className="group-hover:scale-110 transition-transform duration-200">
                      {showCompletedTasks ? '👁️' : '📋'}
                    </span>
                    <span>{showCompletedTasks ? 'Hide' : 'Show'} Completed ({completedTasks.length})</span>
                  </span>
                </button>
              )}
              <button
                onClick={() => setShowAddForm(!showAddForm)}
                className="group px-6 py-2.5 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white rounded-xl text-sm font-semibold transition-all duration-200 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5"
              >
                <span className="flex items-center space-x-2">
                  <span className="group-hover:scale-110 transition-transform duration-200">➕</span>
                  <span>Assign New Task</span>
                </span>
              </button>
            </div>
          </div>
          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl p-4 border border-blue-100">
            <p className="text-blue-800 font-semibold text-lg">Streamlined Task Assignment & Progress Tracking</p>
            <p className="text-blue-600 mt-1">Empower your team with intelligent project management and real-time collaboration</p>
          </div>
        </div>

      {/* Decision Statistics Tiles */}
      {(() => {
        const stats = getDecisionStats();
        if (stats.totalProjectTasks > 0) {
          return (
            <div className="mb-8 grid grid-cols-1 md:grid-cols-4 gap-6">
              <div className="group relative overflow-hidden bg-gradient-to-br from-green-50 to-emerald-100 rounded-2xl border border-green-200/50 p-6 shadow-lg hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1">
                <div className="absolute top-0 right-0 w-20 h-20 bg-gradient-to-br from-green-400/20 to-emerald-500/20 rounded-full blur-xl"></div>
                <div className="relative">
                  <div className="flex items-center justify-between mb-4">
                    <div className="p-2 bg-gradient-to-br from-green-500 to-emerald-600 rounded-lg shadow-md">
                      <span className="text-xl">✅</span>
                    </div>
                    <div className="text-right">
                      <p className="text-sm font-semibold text-green-700">Go Decisions</p>
                      <p className="text-xs text-green-600">Approved projects</p>
                    </div>
                  </div>
                  <div className="text-4xl font-bold text-green-800 mb-2">{stats.goCount}</div>
                  <div className="flex items-center space-x-2">
                    <div className="flex-1 bg-green-200 rounded-full h-2">
                      <div 
                        className="bg-gradient-to-r from-green-500 to-emerald-500 h-2 rounded-full transition-all duration-500"
                        style={{ width: `${stats.totalProjectTasks > 0 ? (stats.goCount / stats.totalProjectTasks) * 100 : 0}%` }}
                      ></div>
                    </div>
                    <span className="text-sm font-semibold text-green-700">
                      {stats.totalProjectTasks > 0 ? Math.round((stats.goCount / stats.totalProjectTasks) * 100) : 0}%
                    </span>
                  </div>
                </div>
              </div>
              
              <div className="group relative overflow-hidden bg-gradient-to-br from-red-50 to-rose-100 rounded-2xl border border-red-200/50 p-6 shadow-lg hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1">
                <div className="absolute top-0 right-0 w-20 h-20 bg-gradient-to-br from-red-400/20 to-rose-500/20 rounded-full blur-xl"></div>
                <div className="relative">
                  <div className="flex items-center justify-between mb-4">
                    <div className="p-2 bg-gradient-to-br from-red-500 to-rose-600 rounded-lg shadow-md">
                      <span className="text-xl">❌</span>
                    </div>
                    <div className="text-right">
                      <p className="text-sm font-semibold text-red-700">No-Go Decisions</p>
                      <p className="text-xs text-red-600">Declined projects</p>
                    </div>
                  </div>
                  <div className="text-4xl font-bold text-red-800 mb-2">{stats.noGoCount}</div>
                  <div className="flex items-center space-x-2">
                    <div className="flex-1 bg-red-200 rounded-full h-2">
                      <div 
                        className="bg-gradient-to-r from-red-500 to-rose-500 h-2 rounded-full transition-all duration-500"
                        style={{ width: `${stats.totalProjectTasks > 0 ? (stats.noGoCount / stats.totalProjectTasks) * 100 : 0}%` }}
                      ></div>
                    </div>
                    <span className="text-sm font-semibold text-red-700">
                      {stats.totalProjectTasks > 0 ? Math.round((stats.noGoCount / stats.totalProjectTasks) * 100) : 0}%
                    </span>
                  </div>
                </div>
              </div>
              
              <div className="group relative overflow-hidden bg-gradient-to-br from-yellow-50 to-amber-100 rounded-2xl border border-yellow-200/50 p-6 shadow-lg hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1">
                <div className="absolute top-0 right-0 w-20 h-20 bg-gradient-to-br from-yellow-400/20 to-amber-500/20 rounded-full blur-xl"></div>
                <div className="relative">
                  <div className="flex items-center justify-between mb-4">
                    <div className="p-2 bg-gradient-to-br from-yellow-500 to-amber-600 rounded-lg shadow-md">
                      <span className="text-xl">⏳</span>
                    </div>
                    <div className="text-right">
                      <p className="text-sm font-semibold text-yellow-700">Pending Decisions</p>
                      <p className="text-xs text-yellow-600">Under review</p>
                    </div>
                  </div>
                  <div className="text-4xl font-bold text-yellow-800 mb-2">{stats.pendingCount}</div>
                  <div className="flex items-center space-x-2">
                    <div className="flex-1 bg-yellow-200 rounded-full h-2">
                      <div 
                        className="bg-gradient-to-r from-yellow-500 to-amber-500 h-2 rounded-full transition-all duration-500"
                        style={{ width: `${stats.totalProjectTasks > 0 ? (stats.pendingCount / stats.totalProjectTasks) * 100 : 0}%` }}
                      ></div>
                    </div>
                    <span className="text-sm font-semibold text-yellow-700">
                      {stats.totalProjectTasks > 0 ? Math.round((stats.pendingCount / stats.totalProjectTasks) * 100) : 0}%
                    </span>
                  </div>
                </div>
              </div>
              
              <div className="group relative overflow-hidden bg-gradient-to-br from-blue-50 to-indigo-100 rounded-2xl border border-blue-200/50 p-6 shadow-lg hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1">
                <div className="absolute top-0 right-0 w-20 h-20 bg-gradient-to-br from-blue-400/20 to-indigo-500/20 rounded-full blur-xl"></div>
                <div className="relative">
                  <div className="flex items-center justify-between mb-4">
                    <div className="p-2 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-lg shadow-md">
                      <span className="text-xl">📄</span>
                    </div>
                    <div className="text-right">
                      <p className="text-sm font-semibold text-blue-700">Total Projects</p>
                      <p className="text-xs text-blue-600">RFP/RFI/RFQ</p>
                    </div>
                  </div>
                  <div className="text-4xl font-bold text-blue-800 mb-2">{stats.totalProjectTasks}</div>
                  <div className="flex items-center space-x-2">
                    <div className="flex-1 bg-blue-200 rounded-full h-2">
                      <div className="bg-gradient-to-r from-blue-500 to-indigo-500 h-2 rounded-full w-full"></div>
                    </div>
                    <span className="text-sm font-semibold text-blue-700">100%</span>
                  </div>
                </div>
              </div>
            </div>
          );
        }
        return null;
      })()}

      {/* Add New Task Form */}
      {showAddForm && (
        <div className="mb-8 relative overflow-hidden bg-gradient-to-br from-white to-gray-50 rounded-2xl border border-gray-200/50 p-8 shadow-xl">
          <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-br from-blue-400/10 to-indigo-500/10 rounded-full blur-2xl"></div>
          <div className="relative">
            <div className="flex items-center space-x-3 mb-6">
              <div className="p-2 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-lg shadow-md">
                <span className="text-xl">📝</span>
              </div>
              <h3 className="text-2xl font-bold bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent">Assign New Task</h3>
            </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Category</label>
              <select
                value={newTask.category}
                onChange={(e) => {
                  const category = e.target.value;
                  setNewTask({
                    ...newTask, 
                    category,
                    type: category === 'Project' ? 'RFP' : 
                          category === 'Meeting' ? 'Scheduling' :
                          category === 'Internal' ? 'Document Review' : 'Other'
                  });
                }}
                className="w-full px-4 py-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 h-12 bg-white/80 backdrop-blur-sm transition-all duration-200 hover:bg-white shadow-sm hover:shadow-md"
              >
                <option value="Project">📄 Project</option>
                <option value="Meeting">📅 Meeting</option>
                <option value="Internal">🔧 Internal</option>
                <option value="Review">👀 Review</option>
                <option value="Other">📋 Other</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Task Type</label>
              <select
                value={newTask.type}
                onChange={(e) => setNewTask({...newTask, type: e.target.value})}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 h-10"
              >
                {newTask.category === 'Project' && (
                  <>
                    <option value="RFP">RFP</option>
                    <option value="RFI">RFI</option>
                    <option value="RFQ">RFQ</option>
                    <option value="Proposal">Proposal</option>
                  </>
                )}
                {newTask.category === 'Meeting' && (
                  <>
                    <option value="Scheduling">Scheduling</option>
                    <option value="Preparation">Preparation</option>
                    <option value="Follow-up">Follow-up</option>
                  </>
                )}
                {newTask.category === 'Internal' && (
                  <>
                    <option value="Document Review">Document Review</option>
                    <option value="Research">Research</option>
                    <option value="Analysis">Analysis</option>
                    <option value="Reporting">Reporting</option>
                  </>
                )}
                {newTask.category === 'Review' && (
                  <>
                    <option value="Technical Review">Technical Review</option>
                    <option value="Compliance Check">Compliance Check</option>
                    <option value="Quality Assurance">Quality Assurance</option>
                  </>
                )}
                {newTask.category === 'Other' && (
                  <>
                    <option value="General Task">General Task</option>
                    <option value="Administrative">Administrative</option>
                    <option value="Training">Training</option>
                  </>
                )}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Task Title</label>
              <input
                type="text"
                value={newTask.title}
                onChange={(e) => setNewTask({...newTask, title: e.target.value})}
                placeholder="Enter task title..."
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 h-10"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Priority</label>
              <select
                value={newTask.priority}
                onChange={(e) => setNewTask({...newTask, priority: e.target.value})}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 h-10"
              >
                <option value="High">🔴 High</option>
                <option value="Medium">🟡 Medium</option>
                <option value="Low">🟢 Low</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Due Date (Optional)</label>
              <input
                type="date"
                value={newTask.dueDate}
                onChange={(e) => setNewTask({...newTask, dueDate: e.target.value})}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 h-10"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Document/Folder (Optional)</label>
              <button
                onClick={() => setShowFolderBrowser(true)}
                className="w-full px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition-colors h-10"
              >
                📁 Browse SharePoint Folders
              </button>
            </div>
            <div>
        <div className="flex items-center justify-between mb-2">
                <label className="block text-sm font-medium text-gray-700">Assign To</label>
                <button
                  onClick={loadUsers}
                  disabled={loadingUsers}
                  className="text-xs text-blue-600 hover:text-blue-800 disabled:opacity-50"
                  title="Refresh user list"
                >
                  {loadingUsers ? '🔄 Loading...' : '🔄 Refresh'}
                </button>
              </div>
              <select
                value={newTask.assignedTo}
                onChange={(e) => setNewTask({...newTask, assignedTo: e.target.value})}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 h-10"
                disabled={loadingUsers}
              >
                <option value="">
                  {loadingUsers ? 'Loading users...' : 'Select user...'}
                </option>
                {availableUsers.map((user, index) => (
                  <option key={index} value={user}>{user}</option>
                ))}
              </select>
              {availableUsers.length === 0 && !loadingUsers && (
                <p className="text-xs text-gray-500 mt-1">No users found. Click refresh to reload.</p>
              )}
              {availableUsers.length > 0 && (
                <p className="text-xs text-gray-500 mt-1">Found {availableUsers.length} user(s)</p>
              )}
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">&nbsp;</label>
              <button
                onClick={handleAddTask}
                className="group w-full px-6 py-3 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white rounded-xl font-semibold transition-all duration-200 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 h-12"
              >
                <span className="flex items-center justify-center space-x-2">
                  <span className="group-hover:scale-110 transition-transform duration-200">🚀</span>
                  <span>Assign Task</span>
                </span>
              </button>
            </div>
          </div>
          </div>
        </div>
      )}

      {/* SharePoint Folder Browser Modal */}
      {showFolderBrowser && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-4xl w-full mx-4 max-h-[80vh] overflow-hidden">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900">Browse SharePoint Folders</h3>
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => loadSharePointFolders(currentPath)}
                  disabled={loadingFolders}
                  className="px-3 py-1 bg-blue-100 hover:bg-blue-200 text-blue-700 rounded text-sm transition-colors disabled:opacity-50"
                >
                  {loadingFolders ? '🔄 Loading...' : '🔄 Refresh'}
                </button>
                <button
                  onClick={() => setShowFolderBrowser(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  ✕
                </button>
              </div>
            </div>

            {/* Breadcrumb Navigation */}
            <div className="mb-4 p-3 bg-gray-50 rounded-lg">
              <div className="flex items-center space-x-2 text-sm">
                <span className="text-gray-500">📍</span>
                <div className="flex items-center space-x-2 min-w-0 flex-1">
                  {getBreadcrumbs().map((crumb, index) => (
                    <React.Fragment key={index}>
                      {index > 0 && <span className="text-gray-400">/</span>}
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

            {/* Back Button */}
            {pathHistory.length > 1 && (
              <button
                onClick={handleBackClick}
                className="mb-4 px-3 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg text-sm transition-colors"
              >
                ← Back
              </button>
            )}

            {/* Folder Grid */}
            <div className="max-h-96 overflow-y-auto">
              {loadingFolders ? (
                <div className="flex items-center justify-center py-12">
                  <div className="text-center">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
                    <p className="text-gray-600">Loading folders...</p>
                  </div>
                </div>
              ) : sharePointFolders.length === 0 ? (
                <div className="text-center py-12">
                  <div className="text-4xl mb-2">📁</div>
                  <p className="text-gray-600">No folders found in this location</p>
                  <p className="text-sm text-gray-500 mt-2">Current path: {currentPath || 'Root'}</p>
                  <p className="text-xs text-gray-400 mt-1">Check browser console for debug info</p>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                  {sharePointFolders.map((item, index) => (
                    <div
                      key={index}
                      className={`p-3 border border-gray-200 rounded-lg transition-all ${
                        item.type === 'folder' 
                          ? 'hover:border-blue-300 hover:bg-blue-50 cursor-pointer' 
                          : 'hover:border-green-300 hover:bg-green-50 cursor-pointer'
                      }`}
                      onClick={() => {
                        if (item.type === 'folder') {
                          handleFolderClick(item);
                               } else {
                                 // Select file for assignment
                                 const filePath = currentPath ? `${currentPath}/${item.filename}` : item.filename;
                                 setNewTask({...newTask, document: filePath});
                                 setShowFolderBrowser(false);
                               }
                      }}
                    >
                      <div className="flex items-center space-x-3">
                        <div className="text-2xl">{item.type === 'folder' ? '📁' : '📄'}</div>
                        <div className="flex-1 min-w-0">
                          <div className="font-medium text-gray-900 truncate" title={item.filename}>
                            {item.filename}
                          </div>
                          <div className="text-xs text-gray-500">
                            {item.type === 'folder' ? 'Folder' : 'File'} • {new Date(item.modified_at).toLocaleDateString()}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
      </div>
 
            {/* Select Current Path Button */}
            {currentPath && (
              <div className="mt-4 pt-4 border-t border-gray-200">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600">Selected Path:</p>
                    <p className="font-medium text-gray-900">{currentPath}</p>
                  </div>
                         <button
                           onClick={() => {
                             setNewTask({...newTask, document: currentPath});
                             setShowFolderBrowser(false);
                           }}
                           className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg font-medium transition-colors"
                         >
                           Select This Path
                         </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {activeTasks.length === 0 && !showCompletedTasks ? (
        <div className="text-center py-16 relative">
          <div className="absolute inset-0 bg-gradient-to-br from-blue-50/50 to-indigo-50/50 rounded-2xl"></div>
          <div className="relative">
            <div className="text-8xl mb-6 opacity-80">📋</div>
            <h3 className="text-2xl font-bold text-gray-900 mb-3">No Tasks Assigned Yet</h3>
            <p className="text-gray-600 mb-8 text-lg">Click "Assign New Task" to assign work to team members</p>
            <button
              onClick={() => setShowAddForm(true)}
              className="group px-8 py-4 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white rounded-xl font-semibold transition-all duration-200 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5"
            >
              <span className="flex items-center space-x-2">
                <span className="group-hover:scale-110 transition-transform duration-200">➕</span>
                <span>Assign Your First Task</span>
              </span>
            </button>
          </div>
        </div>
      ) : (
        <div className="relative overflow-hidden bg-white/80 backdrop-blur-sm rounded-2xl border border-gray-200/50 shadow-xl">
          <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-br from-blue-400/5 to-indigo-500/5 rounded-full blur-2xl"></div>
          <div className="relative overflow-x-auto">
        <table className="w-full">
          <thead>
                <tr className="border-b border-gray-200 bg-gradient-to-r from-gray-50 to-gray-100">
                  <th className="text-left py-4 px-6 font-semibold text-gray-800">Category</th>
                  <th className="text-left py-4 px-6 font-semibold text-gray-800">Task</th>
                  <th className="text-left py-4 px-6 font-semibold text-gray-800">Priority</th>
                  <th className="text-left py-4 px-6 font-semibold text-gray-800">Status</th>
                  <th className="text-left py-4 px-6 font-semibold text-gray-800">Decision</th>
                  <th className="text-left py-4 px-6 font-semibold text-gray-800">Assigned To</th>
                  <th className="text-left py-4 px-6 font-semibold text-gray-800">Assigned By</th>
                  <th className="text-left py-4 px-6 font-semibold text-gray-800">Due Date</th>
                  <th className="text-left py-4 px-6 font-semibold text-gray-800">Actions</th>
            </tr>
          </thead>
          <tbody>
              {activeTasks.map((task, index) => (
              <tr key={task.id} className="border-b border-gray-100 hover:bg-gradient-to-r hover:from-blue-50/50 hover:to-indigo-50/50 transition-all duration-200 group">
                <td className="py-5 px-6">
                  <span className="flex items-center space-x-2">
                    <span>{task.category === 'Project' ? '📄' : 
                           task.category === 'Meeting' ? '📅' :
                           task.category === 'Internal' ? '🔧' :
                           task.category === 'Review' ? '👀' :
                           task.category === 'Document Creation' ? '📝' : '📋'}</span>
                    <span className="font-medium text-gray-900">{task.category}</span>
                  </span>
                </td>
                <td className="py-4 px-4 text-gray-700">
                  <div className="max-w-xs">
                    {/* Show parent/child indicator */}
                    {task.parentTaskId && (
                      <div className="text-xs text-blue-600 mb-1 flex items-center space-x-1">
                        <span>↳</span>
                        <span>Sub-task</span>
                        {task.originalStatus && <span>{task.originalStatus}</span>}
                      </div>
                    )}
                    <div className="font-medium text-gray-900 truncate" title={task.title}>
                      {task.title}
                    </div>
                    <div className="text-xs text-gray-500 truncate" title={task.document}>
                      {task.document && `📁 ${task.document}`}
                    </div>
                    <div className="text-xs text-gray-400 flex items-center space-x-2">
                      <span>{task.type}</span>
                      {/* Show document count for parent RFP tasks */}
                      {task.category === 'Project' && task.documentCount && task.documentCount > 0 && (
                        <span className="bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full font-medium">
                          📋 {task.documentCount} docs
                        </span>
                      )}
                    </div>
                    {/* Show document details if available */}
                    {task.documentDetails && (
                      <div className="text-xs text-gray-500 mt-1 italic">
                        {task.documentDetails}
                      </div>
                    )}
                  </div>
                </td>
                <td className="py-5 px-6">
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                    task.priority === 'High' ? 'bg-red-100 text-red-700' :
                    task.priority === 'Medium' ? 'bg-yellow-100 text-yellow-700' :
                    'bg-green-100 text-green-700'
                  }`}>
                    {task.priority === 'High' ? '🔴' : task.priority === 'Medium' ? '🟡' : '🟢'} {task.priority}
                  </span>
                </td>
                <td className="py-4 px-4 min-w-[200px]">
                  <div className="space-y-3">
                        <select
                          value={task.status}
                          onChange={(e) => handleStatusChange(index, e.target.value)}
                          className="w-full px-4 py-3 text-sm border border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 bg-white/90 backdrop-blur-sm shadow-sm hover:shadow-md transition-all duration-200"
                        >
                      <option value="Assigned">📋 Assigned</option>
                      <option value="Review">👀 Review</option>
                      <option value="In Progress">🔄 In Progress</option>
                      <option value="Completed">✅ Completed</option>
                    </select>
                    {getProgressBar(task.status)}
                    <div className="text-sm text-gray-600 font-medium">
                      {task.status === 'Completed' ? '100% Complete' :
                       task.status === 'In Progress' ? '75% Complete' :
                       task.status === 'Review' ? '50% Complete' :
                       task.status === 'Assigned' ? '10% Complete' :
                       task.status === 'Pending' ? '25% Complete' : '0% Complete'}
                    </div>
                  </div>
                </td>
                <td className="py-4 px-4 min-w-[180px]">
                  {(task.category === 'Project' && (task.type === 'RFP' || task.type === 'RFI' || task.type === 'RFQ')) ? (
                    <div className="space-y-3">
                              <select
                                value={task.decision}
                                onChange={(e) => handleDecisionChange(index, e.target.value)}
                                className="w-full px-4 py-3 text-sm border border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 bg-white/90 backdrop-blur-sm shadow-sm hover:shadow-md transition-all duration-200"
                              >
                        <option value="Decision Pending">⏳ Decision Pending</option>
                        <option value="Go">✅ Go</option>
                        <option value="No-Go">❌ No-Go</option>
                      </select>
                      <div className={`text-sm font-medium px-3 py-1 rounded-lg ${getDecisionBadge(task.decision)}`}>
                        {task.decision === 'Go' ? '✅ Go' :
                         task.decision === 'No-Go' ? '❌ No-Go' :
                         '⏳ Pending'}
                      </div>
                    </div>
                  ) : (
                    <div className="text-center py-4">
                      <span className="text-sm text-gray-400 bg-gray-100 px-3 py-1 rounded-lg">N/A</span>
                    </div>
                  )}
                </td>
                <td className="py-4 px-4">
                  <div className="flex items-center gap-2">
                    <select
                      className="border rounded-lg px-2 py-1 text-sm"
                      value={task.assignedTo || ''}
                      onChange={(e) => onAssigneeChange(task.id, e.target.value)}
                    >
                      <option value="" disabled>Select user…</option>
                      {(availableUsers.length ? availableUsers : FIXED_ASSIGNEES).map((name) => (
                        <option key={name} value={name}>{name}</option>
                      ))}
                    </select>

                    {task.assignedTo === currentUser?.name && (
                      <span className="text-xs bg-blue-100 text-blue-600 px-2 py-0.5 rounded-full">
                        You
                      </span>
                    )}
                  </div>
                </td>

                <td className="py-4 px-4 text-gray-700 min-w-[140px]">
                  <span className="text-sm">{task.assignedBy || '—'}</span>
                </td>
                <td className="py-4 px-4 text-gray-700 min-w-[140px]">
                  <span className="text-xs text-gray-500">
                    {task.dueDate ? new Date(task.dueDate).toLocaleDateString() : 'No due date'}
                  </span>
                </td>
                <td className="py-5 px-6">
                  <button
                    onClick={() => handleDeleteTask(task)}
                    className="group px-3 py-1.5 bg-red-50 hover:bg-red-100 border border-red-200 hover:border-red-300 text-red-600 hover:text-red-700 rounded-lg text-xs font-medium transition-all duration-200 shadow-sm hover:shadow-md"
                    title="Delete this task"
                  >
                    <span className="flex items-center space-x-1">
                      <span className="group-hover:scale-110 transition-transform duration-200">🗑️</span>
                      <span>Delete</span>
                    </span>
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        </div>
        </div>
      )}

      {/* Completed Tasks Section */}
      {showCompletedTasks && completedTasks.length > 0 && (
        <div className="mt-8 relative overflow-hidden bg-gradient-to-br from-green-50/80 to-emerald-50/80 backdrop-blur-sm rounded-2xl border border-green-200/50 shadow-xl">
          <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-br from-green-400/10 to-emerald-500/10 rounded-full blur-2xl"></div>
          <div className="relative p-6">
            <div className="flex items-center space-x-3 mb-6">
              <div className="p-2 bg-gradient-to-br from-green-500 to-emerald-600 rounded-lg shadow-md">
                <span className="text-xl">✅</span>
              </div>
              <div>
                <h3 className="text-xl font-bold text-green-800">Completed Tasks</h3>
                <p className="text-green-600 text-sm">Tasks that have been finished ({completedTasks.length} total)</p>
              </div>
            </div>
            
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-green-200 bg-gradient-to-r from-green-50 to-emerald-50">
                    <th className="text-left py-3 px-4 font-semibold text-green-800">Category</th>
                    <th className="text-left py-3 px-4 font-semibold text-green-800">Task</th>
                    <th className="text-left py-3 px-4 font-semibold text-green-800">Priority</th>
                    <th className="text-left py-3 px-4 font-semibold text-green-800">Decision</th>
                    <th className="text-left py-3 px-4 font-semibold text-green-800">Assigned To</th>
                    <th className="text-left py-3 px-4 font-semibold text-green-800">Completed Date</th>
                  </tr>
                </thead>
                <tbody>
                  {completedTasks.map((task, index) => (
                    <tr key={task.id} className="border-b border-green-100 hover:bg-gradient-to-r hover:from-green-50/50 hover:to-emerald-50/50 transition-all duration-200 group">
                      <td className="py-4 px-4">
                        <span className="flex items-center space-x-2">
                          <span>{task.category === 'Project' ? '📄' : 
                                 task.category === 'Meeting' ? '📅' :
                                 task.category === 'Internal' ? '🔧' :
                                 task.category === 'Review' ? '👀' : '📋'}</span>
                          <span className="font-medium text-gray-900">{task.category}</span>
                        </span>
                      </td>
                      <td className="py-4 px-4 text-gray-700">
                        <div className="max-w-xs">
                          <div className="font-medium text-gray-900 truncate" title={task.title}>
                            {task.title}
                          </div>
                          <div className="text-xs text-gray-500 truncate" title={task.document}>
                            {task.document && `📁 ${task.document}`}
                          </div>
                          <div className="text-xs text-gray-400">
                            {task.type}
                          </div>
                        </div>
                      </td>
                      <td className="py-4 px-4">
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${task.priority === 'High' ? 'bg-red-100 text-red-700' : task.priority === 'Medium' ? 'bg-yellow-100 text-yellow-700' : 'bg-green-100 text-green-700'}`}>
                          {task.priority === 'High' ? '🔴' : task.priority === 'Medium' ? '🟡' : '🟢'} {task.priority}
                        </span>
                      </td>
                      <td className="py-4 px-4">
                        {(task.category === 'Project' && (task.type === 'RFP' || task.type === 'RFI' || task.type === 'RFQ')) ? (
                          <div className={`text-sm font-medium px-3 py-1 rounded-lg ${getDecisionBadge(task.decision)}`}>
                            {task.decision === 'Go' ? '✅ Go' :
                             task.decision === 'No-Go' ? '❌ No-Go' :
                             '⏳ Pending'}
                          </div>
                        ) : (
                          <div className="text-center py-2">
                            <span className="text-sm text-gray-400 bg-gray-100 px-3 py-1 rounded-lg">N/A</span>
                          </div>
                        )}
                      </td>
                      <td className="py-4 px-4">
                        <div className="flex items-center gap-2">
                          <select
                            className="border rounded-lg px-2 py-1 text-sm"
                            value={task.assignedTo || ''}
                            onChange={(e) => onAssigneeChange(task.id, e.target.value)}
                          >
                            <option value="" disabled>Select user…</option>
                            {(availableUsers.length ? availableUsers : FIXED_ASSIGNEES).map((name) => (
                              <option key={name} value={name}>{name}</option>
                            ))}
                          </select>

                          {task.assignedTo === currentUser?.name && (
                            <span className="text-xs bg-blue-100 text-blue-600 px-2 py-0.5 rounded-full">
                              You
                            </span>
                          )}
                        </div>
                      </td>

                      <td className="py-4 px-4 text-gray-700">
                        <span className="text-xs text-gray-500">
                          {new Date(task.updatedAt).toLocaleDateString()}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {showDeleteModal && taskToDelete && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-2xl p-8 max-w-md w-full mx-4 shadow-2xl">
            <div className="flex items-center space-x-3 mb-6">
              <div className="p-3 bg-red-100 rounded-xl">
                <span className="text-2xl">⚠️</span>
              </div>
              <div>
                <h3 className="text-xl font-bold text-gray-900">Delete Task</h3>
                <p className="text-gray-600">This action cannot be undone</p>
              </div>
            </div>
            
            <div className="mb-6 p-4 bg-gray-50 rounded-xl">
              <p className="text-gray-800 font-medium mb-2">Task Details:</p>
              <p className="text-gray-700"><strong>Title:</strong> {taskToDelete.title}</p>
              <p className="text-gray-700"><strong>Category:</strong> {taskToDelete.category}</p>
              <p className="text-gray-700"><strong>Assigned To:</strong> {taskToDelete.assignedTo}</p>
              <p className="text-gray-700"><strong>Status:</strong> {taskToDelete.status}</p>
            </div>
            
            <div className="flex items-center space-x-3">
              <button
                onClick={confirmDeleteTask}
                className="flex-1 px-6 py-3 bg-red-600 hover:bg-red-700 text-white rounded-xl font-semibold transition-all duration-200 shadow-lg hover:shadow-xl"
              >
                Yes, Delete Task
              </button>
              <button
                onClick={cancelDeleteTask}
                className="flex-1 px-6 py-3 bg-gray-200 hover:bg-gray-300 text-gray-800 rounded-xl font-semibold transition-all duration-200"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
      </div>
    </div>
  );
};
 
export default RecentDocuments;