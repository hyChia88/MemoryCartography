import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import FileUploadZone from '../components/FileUploadZone';
import { useSession } from '../contexts/SessionContext';

/**
 * Home page component with file upload zone
 * 
 * @returns {JSX.Element} - Home page component
 */
const HomePage = () => {
  const { sessionId, loading } = useSession();
  const navigate = useNavigate();
  
  // Redirect to memories page if there's an active session
  useEffect(() => {
    if (sessionId && !loading) {
      navigate(`/memories?session=${sessionId}`);
    }
  }, [sessionId, loading, navigate]);
  
  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-gray-500"></div>
      </div>
    );
  }
  
  return (
    <div className="min-h-screen bg-white">
      <FileUploadZone />
    </div>
  );
};

export default HomePage;