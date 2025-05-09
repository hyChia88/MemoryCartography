import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import SearchBar from '../components/SearchBar';
import MemoryList from '../components/MemoryList';
import NarrativeDisplay from '../components/NarrativeDisplay';
import ResetWeightsButton from '../components/ResetWeightsButton';
import { useSession } from '../contexts/SessionContext';
import { searchMemories, generateNarrative, adjustMemoryWeight, getThumbnailUrl } from '../utils/api';

/**
 * Memory exploration page component
 * 
 * @returns {JSX.Element} - Memories page component
 */
const MemoriesPage = () => {
  const { sessionId, sessionStatus, loading: sessionLoading, error: sessionError, deleteSession } = useSession();
  const navigate = useNavigate();
  
  const [searchQuery, setSearchQuery] = useState('');
  const [memories, setMemories] = useState([]);
  const [narrative, setNarrative] = useState(null);
  const [memoryType, setMemoryType] = useState('user');
  const [sortBy, setSortBy] = useState('weight');
  const [loading, setLoading] = useState(false);
  const [isGeneratingNarrative, setIsGeneratingNarrative] = useState(false);
  const [error, setError] = useState(null);
  const [resetMessage, setResetMessage] = useState(null);
  
  // Redirect to home if no session
  useEffect(() => {
    if (!sessionId && !sessionLoading) {
      navigate('/');
    }
  }, [sessionId, sessionLoading, navigate]);
  
  // Handle search submission
  const handleSearch = async (query) => {
    if (!query.trim()) {
      setError('Please enter a search term');
      return;
    }
    
    setSearchQuery(query);
    setLoading(true);
    setError(null);
    
    try {
      const results = await searchMemories({
        sessionId,
        query,
        memoryType,
        sortBy
      });
      
      setMemories(results);
    } catch (err) {
      console.error('Search error', err);
      setError(err.message || 'An error occurred while searching');
    } finally {
      setLoading(false);
    }
  };
  
  // Handle narrative generation
  const handleGenerateNarrative = async (query) => {
    if (!query.trim()) {
      setError('Please enter a search term');
      return;
    }
    
    setIsGeneratingNarrative(true);
    setError(null);
    
    try {
      const result = await generateNarrative({
        sessionId,
        query,
        memoryType
      });
      
      setNarrative(result);
    } catch (err) {
      console.error('Narrative generation error', err);
      setError(err.message || 'An error occurred while generating the narrative');
    } finally {
      setIsGeneratingNarrative(false);
    }
  };
  
  // Handle memory type change
  const handleMemoryTypeChange = (type) => {
    setMemoryType(type);
    
    // If there's a current search query, re-run the search
    if (searchQuery.trim()) {
      handleSearch(searchQuery);
    }
  };
  
  // Handle increasing memory weight (left click)
  const handleIncreaseWeight = async (memoryId, event) => {
    event.preventDefault();
    
    try {
      const response = await adjustMemoryWeight({
        sessionId,
        memoryId,
        adjustment: 0.1
      });
      
      // Update memory in the list
      setMemories(prevMemories => {
        // Map through memories and update the one that changed
        return prevMemories.map(memory => 
          memory.id === memoryId 
            ? { ...memory, weight: response.new_weight } 
            : memory
        );
      });
    } catch (error) {
      console.error('Error increasing memory weight:', error);
    }
  };
  
  // Handle decreasing memory weight (right click)
  const handleDecreaseWeight = async (memoryId, event) => {
    event.preventDefault();
    
    try {
      const response = await adjustMemoryWeight({
        sessionId,
        memoryId,
        adjustment: -0.1
      });
      
      // Update memory in the list
      setMemories(prevMemories => {
        // Map through memories and update the one that changed
        return prevMemories.map(memory => 
          memory.id === memoryId 
            ? { ...memory, weight: response.new_weight } 
            : memory
        );
      });
    } catch (error) {
      console.error('Error decreasing memory weight:', error);
    }
  };
  
  // Handle reset weight completion
  const handleResetComplete = (data) => {
    if (data.status === 'success') {
      setResetMessage(`Reset ${data.updated_count} memories successfully!`);
      
      // Clear the message after 3 seconds
      // Clear the message after 3 seconds
      setTimeout(() => {
        setResetMessage(null);
      }, 3000);
      
      // If we have a current search, refresh the results to see updated weights
      if (searchQuery.trim()) {
        handleSearch(searchQuery);
      }
    } else {
      setError(data.message || 'Failed to reset weights');
    }
  };
  
  // Handle session deletion
  const handleDeleteSession = async () => {
    if (window.confirm('Are you sure you want to delete this session and all associated data? This action cannot be undone.')) {
      try {
        const success = await deleteSession();
        
        if (success) {
          // Redirect to home page
          navigate('/');
        }
      } catch (error) {
        setError('Failed to delete session');
      }
    }
  };
  
  // Handle closing the narrative
  const handleCloseNarrative = () => {
    setNarrative(null);
  };

  // If session is loading, show loading state
  if (sessionLoading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-gray-500"></div>
      </div>
    );
  }

  return (
    <div className="bg-white min-h-screen p-4">
      <div className="container mx-auto max-w-4xl">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-3xl font-bold text-gray-800">Memory Cartography</h1>
          
          {sessionStatus && (
            <div className="flex items-center">
              <div className="text-sm text-gray-500 flex items-center mr-4">
                <span className="mr-1">⏱️</span>
                <span>Session expires in 24 hours</span>
              </div>
              <button
                onClick={handleDeleteSession}
                className="flex items-center text-red-500 hover:text-red-600"
              >
                <span className="mr-1">🗑️</span>
                <span>Delete Session</span>
              </button>
            </div>
          )}
        </div>
        
        {/* Reset Success Message */}
        {resetMessage && (
          <div className="bg-green-100 border border-green-400 text-green-700 px-4 py-2 rounded mb-4 text-sm">
            {resetMessage}
          </div>
        )}
        
        {/* Error Handling */}
        {(error || sessionError) && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
            <span className="block sm:inline">{error || sessionError}</span>
          </div>
        )}
        
        {/* Session Status */}
        {sessionStatus && (
          <div className="mb-6 bg-gray-50 p-4 rounded-lg">
            <div className="flex flex-wrap items-center justify-between">
              <div className="flex items-center mb-2 md:mb-0">
                <span className="text-gray-500 mr-2">ℹ️</span>
                <div>
                  <p className="font-medium">Session Status</p>
                  <p className="text-sm text-gray-600">
                    {sessionStatus.user_memories} personal photos • {sessionStatus.public_memories} public photos
                  </p>
                </div>
              </div>
              
              {sessionStatus.locations && sessionStatus.locations.length > 0 && (
                <div className="flex flex-wrap gap-2">
                  {sessionStatus.locations.map((location, index) => (
                    <span key={index} className="inline-block bg-gray-100 text-gray-800 px-2 py-1 rounded-full text-sm">
                      {location}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
        
        {/* Search Bar */}
        <SearchBar
          initialValue={searchQuery}
          onSearch={handleSearch}
          onGenerateNarrative={handleGenerateNarrative}
          isLoading={loading}
          isGeneratingNarrative={isGeneratingNarrative}
          memoryType={memoryType}
          onMemoryTypeChange={handleMemoryTypeChange}
          extraButtons={
            <ResetWeightsButton
              memoryType={memoryType}
              onResetComplete={handleResetComplete}
            />
          }
        />
        
        {/* Narrative Display */}
        {narrative && (
          <NarrativeDisplay
            narrative={narrative.text}
            keywords={narrative.keywords}
            highlightedTerms={narrative.highlighted_terms}
            sourceMemories={narrative.source_memories}
            onClose={handleCloseNarrative}
          />
        )}
        
        {/* Memory List */}
        {loading ? (
          <div className="flex justify-center items-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-gray-500"></div>
          </div>
        ) : (
          <div>
            <MemoryList
              memories={memories}
              onIncreaseWeight={handleIncreaseWeight}
              onDecreaseWeight={handleDecreaseWeight}
              getThumbnailUrl={getThumbnailUrl}
            />
            
            {/* No results message */}
            {memories.length === 0 && searchQuery && !loading && (
              <div className="text-center py-8 text-gray-500">
                No memories found for "{searchQuery}"
              </div>
            )}
            
            {/* Instructions */}
            {memories.length > 0 && (
              <div className="mt-4 text-xs text-gray-500 text-center">
                Left-click on a memory to increase its weight • Right-click to decrease weight
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default MemoriesPage;