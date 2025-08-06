import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react'

export interface User {
  id: number
  name: string
  email: string
  isLoggedIn: boolean
}

export interface AuthContextType {
  isLoggedIn: boolean
  currentUser: User | null
  login: () => void
  logout: () => void
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

interface AuthProviderProps {
  children: ReactNode
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [isLoggedIn, setIsLoggedIn] = useState(false)
  const [currentUser, setCurrentUser] = useState<User | null>(null)

  // Check if user is already logged in on app start
  useEffect(() => {
    const user = localStorage.getItem('user')
    if (user) {
      const userData = JSON.parse(user)
      if (userData.isLoggedIn) {
        setIsLoggedIn(true)
        setCurrentUser(userData)
      }
    }
  }, [])

  const login = () => {
    setIsLoggedIn(true)
    // Get updated user data
    const user = localStorage.getItem('user')
    if (user) {
      setCurrentUser(JSON.parse(user))
    }
  }

  const logout = () => {
    localStorage.removeItem('user')
    setIsLoggedIn(false)
    setCurrentUser(null)
  }

  return (
    <AuthContext.Provider value={{ isLoggedIn, currentUser, login, logout }}>
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