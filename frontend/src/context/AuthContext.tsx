import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import { apiClient } from '../utils/api'

export interface User {
  id: string
  name: string
  email: string
  created_at: string
  last_login?: string
}

export interface AuthContextType {
  isLoggedIn: boolean
  currentUser: User | null
  login: (userData: { email: string; password: string }) => Promise<boolean>
  register: (userData: { name: string; email: string; password: string }) => Promise<boolean>
  logout: () => Promise<void>
  loading: boolean
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

interface AuthProviderProps {
  children: ReactNode
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [isLoggedIn, setIsLoggedIn] = useState(false)
  const [currentUser, setCurrentUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(true)

  // Check if user is already logged in on app start
  useEffect(() => {
    const checkAuthStatus = async () => {
      try {
        const token = localStorage.getItem('access_token')
        if (token) {
          const userData = await apiClient.getCurrentUser()
          if (userData) {
            setIsLoggedIn(true)
            setCurrentUser(userData)
          } else {
            // Token is invalid, remove it
            localStorage.removeItem('access_token')
          }
        }
      } catch (error) {
        console.error('Auth check failed:', error)
        localStorage.removeItem('access_token')
      } finally {
        setLoading(false)
      }
    }

    checkAuthStatus()
  }, [])

  const login = async (userData: { email: string; password: string }): Promise<boolean> => {
    try {
      setLoading(true)
      const response = await apiClient.login(userData)
      
      // Store token and user data
      localStorage.setItem('access_token', response.access_token)
      setIsLoggedIn(true)
      setCurrentUser(response.user)
      
      return true
    } catch (error) {
      console.error('Login failed:', error)
      return false
    } finally {
      setLoading(false)
    }
  }

  const register = async (userData: { name: string; email: string; password: string }): Promise<boolean> => {
    try {
      setLoading(true)
      const response = await apiClient.register(userData)
      
      // Don't auto-login after registration
      // Just return success, user needs to login manually
      console.log('Registration successful, user needs to login')
      
      return true
    } catch (error: any) {
      console.error('Registration failed:', error)
      // Extract error message from response if available
      const errorMessage = error?.response?.data?.error || error?.message || 'Registration failed'
      console.error('Registration error details:', errorMessage)
      // Throw error with message so Register component can display it
      throw new Error(errorMessage)
    } finally {
      setLoading(false)
    }
  }

  const logout = async (): Promise<void> => {
    try {
      await apiClient.logout()
    } catch (error) {
      console.error('Logout failed:', error)
    } finally {
      // Always clear local state
      localStorage.removeItem('access_token')
      setIsLoggedIn(false)
      setCurrentUser(null)
    }
  }

  return (
    <AuthContext.Provider value={{ isLoggedIn, currentUser, login, register, logout, loading }}>
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

export default AuthContext