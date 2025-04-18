// src/components/MemoryMapApp.jsx
import React, { useState, useEffect } from 'react';
import { ToggleLeft, ToggleRight } from 'lucide-react';
import MemoryList from './MemoryList';
import SearchBar from './SearchBar';
import GeneratedText from './GeneratedText';
import MemoryVisualizer from './MemoryVisualizer';
import MemoryInput from './MemoryInput';
import { fetchMemories, generateNarrative, addMemory as apiAddMemory } from '../services/api';

const MemoryMapApp = () => {
  const [activeDatabase, setActiveDatabase] = useState('user');
  const [searchLocation, setSearchLocation] = useState('');
  const [generatedContent, setGeneratedContent] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [userMemories, setUserMemories] = useState([]);
  const [publicMemories, setPublicMemories] = useState([]);
  const [newMemory, setNewMemory] = useState('');

  // Fetch memories when active database changes
  useEffect(() => {
    const loadMemories = async () => {
      try {
        const memories = await fetchMemories(activeDatabase);
        if (activeDatabase === 'user') {
          setUserMemories(memories);
        } else {
          setPublicMemories(memories);
        }
      } catch (error) {
        console.error('Failed to load memories:', error);
      }
    };

    loadMemories();
  }, [activeDatabase]);

  const handleSearch = async () => {
    if (!searchLocation.trim()) return;
    
    setIsLoading(true);
    try {
      const data = await generateNarrative(searchLocation, activeDatabase);
      setGeneratedContent(data);
    } catch (error) {
      console.error('Error generating narrative:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const toggleDatabase = () => {
    setActiveDatabase(activeDatabase === 'user' ? 'public' : 'user');
    // If we have a generated content, regenerate for the new database type
    if (generatedContent && searchLocation) {
      handleSearch();
    }
  };

  const handleMemoryClick = (location) => {
    setSearchLocation(location);
    handleSearch();
  };

  const addMemoryEntry = async () => {
    if (!newMemory.trim() || !searchLocation.trim()) return;
    
    try {
      // Extract a title from the memory content
      const title = newMemory.split('\n')[0] || `Memory about ${searchLocation}`;
      
      // Create memory object
      const memoryData = {
        title: title.length > 50 ? title.substring(0, 47) + '...' : title,
        location: searchLocation,
        date: new Date().toISOString().split('T')[0],
        type: activeDatabase,
        keywords: generatedContent ? 
          generatedContent.keywords
            .filter(k => k.type === 'primary')
            .map(k => k.text) : 
          [searchLocation],
        description: newMemory
      };
      
      // Submit to API
      await apiAddMemory(memoryData);
      
      // Refresh memories list
      const updatedMemories = await fetchMemories(activeDatabase);
      if (activeDatabase === 'user') {
        setUserMemories(updatedMemories);
      } else {
        setPublicMemories(updatedMemories);
      }
      
      // Clear input
      setNewMemory('');
      
      // Re-generate narrative to include new memory
      handleSearch();
    } catch (error) {
      console.error('Error adding memory:', error);
    }
  };

  return (
    <div className="min-h-screen bg-white flex flex-col p-4 max-w-7xl mx-auto">
      {/* Header */}
      <header className="flex justify-between items-center mb-4">
        <h1 className="text-2xl font-light text-gray-800">Memory Cartography</h1>
        <div className="flex items-center space-x-2">
          <button 
            onClick={toggleDatabase} 
            className="flex items-center space-x-2 text-gray-600 hover:text-gray-900"
          >
            {activeDatabase === 'user' ? (
              <>
                <ToggleLeft className="text-blue-500" />
                <span className="text-sm">User Memories</span>
              </>
            ) : (
              <>
                <ToggleRight className="text-green-500" />
                <span className="text-sm">Public Memories</span>
              </>
            )}
          </button>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex flex-1 space-x-4">
        {/* Memories List */}
        <div className="w-1/4 bg-gray-50 rounded-lg p-3 overflow-y-auto">
          <h2 className="text-sm font-semibold text-gray-600 mb-3 uppercase tracking-wider">
            {activeDatabase === 'user' ? 'Your Memories' : 'Collective Memories'}
          </h2>
          <MemoryList 
            memories={activeDatabase === 'user' ? userMemories : publicMemories}
            onMemoryClick={handleMemoryClick}
          />
        </div>

        {/* Central Content Area */}
        <div className="flex-1 flex flex-col space-y-4">
          {/* Search Bar */}
          <SearchBar 
            value={searchLocation}
            onChange={setSearchLocation}
            onSearch={handleSearch}
            isLoading={isLoading}
          />

          {/* Generated Text Area */}
          <GeneratedText 
            generatedContent={generatedContent}
            isLoading={isLoading}
          />

          {/* Memory Visualization */}
          {generatedContent && (
            <MemoryVisualizer 
              primaryMemories={generatedContent.primary_memories || []}
              connectedMemories={generatedContent.connected_memories || []}
              activeType={activeDatabase}
            />
          )}

          {/* Input Area */}
          <MemoryInput 
            value={newMemory}
            onChange={setNewMemory}
            onSubmit={addMemoryEntry}
            isDisabled={!newMemory.trim() || !searchLocation.trim()}
          />
        </div>
      </div>
    </div>
  );
};

export default MemoryMapApp;