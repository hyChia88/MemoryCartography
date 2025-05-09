import React, { useState, useEffect } from 'react';
import FileUploadZone from './components/FileUploadZone';
import MemoryApp from './MemoryApp.tsx'; // Correct path with file extension

/**
 * Main App component with simplified routing
 * No dependencies on react-router-dom or SessionContext
 */
const App = () => {
  const [currentView, setCurrentView] = useState('upload'); // 'upload' or 'memories'
  const [sessionId, setSessionId] = useState(null);
  
  // Check URL for session parameter on load
  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search);
    const sessionParam = urlParams.get('session');
    
    if (sessionParam) {
      setSessionId(sessionParam);
      setCurrentView('memories');
    }
  }, []);
  
  // Change view to memories and update URL
  const navigateToMemories = (sessionId) => {
    setSessionId(sessionId);
    setCurrentView('memories');
    
    // Update URL without react-router
    window.history.pushState(
      {}, 
      '', 
      `${window.location.pathname}?session=${sessionId}`
    );
  };
  
  // Return to upload view
  const navigateToUpload = () => {
    setCurrentView('upload');
    window.history.pushState({}, '', window.location.pathname);
  };
  
  return (
    <div>
      {currentView === 'upload' ? (
        <FileUploadZone onSessionCreated={navigateToMemories} />
      ) : (
        <MemoryApp onExit={navigateToUpload} />
      )}
    </div>
  );
};

export default App;