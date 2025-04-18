import React, { useState, useEffect } from 'react';
import { ToggleLeft, ToggleRight } from 'lucide-react';
import memoryService from '../services/api';
import MemoryList from './MemoryList';
import SearchBar from './SearchBar';
import GeneratedText from './GeneratedText';
import MemoryInput from './MemoryInput';

const MemoryMapApp = () => {
  const [activeDatabase, setActiveDatabase] = useState('user');
  const [searchLocation, setSearchLocation] = useState('');
  const [generatedText, setGeneratedText] = useState('');
  const [keywords, setKeywords] = useState([]);
  const [userMemories, setUserMemories] = useState([]);
  const [publicMemories, setPublicMemories] = useState([]);
  const [newMemory, setNewMemory] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  // Fetch memories on component mount and database switch
  useEffect(() => {
    fetchMemories();
  }, [activeDatabase]);

  const fetchMemories = async () => {
    try {
      const memories = await memoryService.getMemories(activeDatabase);
      if (activeDatabase === 'user') {
        setUserMemories(memories);
      } else {
        setPublicMemories(memories);
      }
    } catch (error) {
      console.error('Error fetching memories:', error);
    }
  };

  const handleSearch = async () => {
    if (!searchLocation.trim()) return;
    
    setIsLoading(true);
    try {
      const response = await memoryService.generateNarrative(searchLocation, activeDatabase);
      setGeneratedText(response.text);
      setKeywords(response.keywords);
    } catch (error) {
      console.error('Error generating narrative:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const toggleDatabase = () => {
    setActiveDatabase(activeDatabase === 'user' ? 'public' : 'user');
  };

  const handleMemoryClick = (location) => {
    setSearchLocation(location);
    handleSearch();
  };

  const addMemory = async () => {
    if (!newMemory.trim() || !searchLocation.trim()) return;
    
    try {
      // Extract a title from the memory content
      const title = newMemory.split('\n')[0] || `Memory about ${searchLocation}`;
      
      await memoryService.addMemory({
        title: title.length > 50 ? title.substring(0, 47) + '...' : title,
        location: searchLocation,
        date: new Date().toISOString().split('T')[0],
        type: activeDatabase,
        keywords: keywords
          .filter(k => k.type === 'primary')
          .map(k => k.text),
        content: newMemory
      });
      
      fetchMemories();
      setNewMemory('');
    } catch (error) {
      console.error('Error adding memory:', error);
    }
  };

  const seedSampleData = async () => {
    try {
      await memoryService.seedData();
      fetchMemories();
    } catch (error) {
      console.error('Error seeding data:', error);
    }
  };

  return (
    <div className="min-h-screen bg-white flex flex-col p-4 max-w-7xl mx-auto">
      {/* Header */}
      <header className="flex justify-between items-center mb-4">
        <h1 className="text-2xl font-light text-gray-800">Memory Cartography</h1>
        <div className="flex items-center space-x-4">
          <button 
            onClick={seedSampleData} 
            className="text-xs text-gray-500 hover:text-gray-700"
          >
            Seed Demo Data
          </button>
          <button 
            onClick={toggleDatabase} 
            className="flex items-center space-x-2 text-gray-600 hover:text-gray-900"
          >
            {activeDatabase === 'user' ? (
              <>
                <ToggleLeft className="text-blue-500" />
                <span className="text-sm">User</span>
              </>
            ) : (
              <>
                <ToggleRight className="text-green-500" />
                <span className="text-sm">Public</span>
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
            {activeDatabase === 'user' ? 'User Memories' : 'Public Memories'}
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
            generatedText={generatedText}
            keywords={keywords}
          />

          {/* Input Area */}
          <MemoryInput 
            value={newMemory}
            onChange={setNewMemory}
            onSubmit={addMemory}
            isDisabled={!newMemory.trim() || !searchLocation.trim()}
          />
        </div>
      </div>
    </div>
  );
};

export default MemoryMapApp;