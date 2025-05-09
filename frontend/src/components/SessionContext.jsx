import React, { createContext, useState, useContext, useEffect } from 'react';
import axios from 'axios';

// Create context
const SessionContext = createContext();

/**
 * Provider component for session management
 * 
 * @param {Object} props
 * @param {React.ReactNode} props.children - Child components
 * @returns {JSX.Element} - Context provider
 */
export const SessionProvider = ({ children }) => {
  const [sessionId, setSessionId] = useState(null);
  const [sessionStatus, setSessionStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // API client
  const api = axios.create({
    baseURL: process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000',
    timeout: 15000,
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
  });

  // Check for existing session on load
  useEffect(() => {
    // Get session from URL parameter
    const params = new URLSearchParams(window.location.search);
    const sessionParam = params.get('session');
    
    if (sessionParam) {
      setSessionId(sessionParam);
      fetchSessionStatus(sessionParam);
    } else {
      setLoading(false);
    }
  }, []);

  // Fetch session status
  const fetchSessionStatus = async (session) => {
    if (!session) return;
    
    try {
      setLoading(true);
      const response = await api.get(`/api/session/${session}/status`);
      
      if (response.status === 200) {
        setSessionStatus(response.data);
      } else {
        setError('Failed to fetch session status');
      }
    } catch (error) {
      setError(`Error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Create a new session
  const createSession = async () => {
    try {
      setLoading(true);
      const response = await api.post('/api/session/create');
      
      if (response.status === 200) {
        const newSessionId = response.data.session_id;
        setSessionId(newSessionId);
        return newSessionId;
      } else {
        setError('Failed to create session');
        return null;
      }
    } catch (error) {
      setError(`Error: ${error.message}`);
      return null;
    } finally {
      setLoading(false);
    }
  };

  // Delete current session
  const deleteSession = async () => {
    if (!sessionId) return false;
    
    try {
      setLoading(true);
      const response = await api.delete(`/api/session/${sessionId}`);
      
      if (response.status === 200) {
        setSessionId(null);
        setSessionStatus(null);
        return true;
      } else {
        setError('Failed to delete session');
        return false;
      }
    } catch (error) {
      setError(`Error: ${error.message}`);
      return false;
    } finally {
      setLoading(false);
    }
  };

  return (
    <SessionContext.Provider
      value={{
        sessionId,
        sessionStatus,
        loading,
        error,
        createSession,
        deleteSession,
        fetchSessionStatus,
      }}
    >
      {children}
    </SessionContext.Provider>
  );
};

/**
 * Custom hook to use the session context
 * 
 * @returns {Object} - Session context
 */
export const useSession = () => {
  const context = useContext(SessionContext);
  if (!context) {
    throw new Error('useSession must be used within a SessionProvider');
  }
  return context;
};

export default SessionContext;