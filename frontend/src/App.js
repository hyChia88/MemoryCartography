import React, { useState, useEffect } from 'react';
import SearchBar from './components/SearchBar';
import MemoryList from './components/MemoryList';
import MemoryForm from './components/MemoryForm';
import NarrativeDisplay from './components/NarrativeDisplay';

// Services
import memoryProcessingService from './services/memoryProcessingService';
import recommenderService from './services/recommenderService';
import openaiService from './services/openaiService';
import highlightUtils from './utils/highlightUtils';
import weightCalculator from './utils/weightCalculator';
import memoryService from './services/api';

function App() {
  // State management
  const [memories, setMemories] = useState({});
  const [searchResults, setSearchResults] = useState([]);
  const [syntheticNarrative, setSyntheticNarrative] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeDatabase, setActiveDatabase] = useState('user');

  // Load initial memories
  useEffect(() => {
    const loadMemories = async () => {
      try {
        setIsLoading(true);
        const memories = await memoryService.getMemories();
        console.log('Fetched memories:', memories);
        setMemories(memories);
      } catch (error) {
        console.error('Error loading memories:', error);
        setError('Failed to load memories');
      } finally {
        setIsLoading(false);
      }
    };
  
    loadMemories();
  }, []);

  // Handle image upload and processing
  const handleImageUpload = async (imageFile) => {
    setIsLoading(true);
    try {
      // Process the image
      const memoryEvent = await memoryProcessingService.processImage(imageFile);
      
      // Save the memory event
      const updatedMemories = memoryProcessingService.saveMemoryEvents([memoryEvent]);
      setMemories(updatedMemories);
      
      setIsLoading(false);
    } catch (err) {
      setError('Failed to process image');
      setIsLoading(false);
    }
  };

  // Handle search and recommendation
  const handleSearch = async (searchTerms) => {
    setIsLoading(true);
    try {
      // Split search terms
      const terms = searchTerms.split(/\s+/);
      
      // Recommend memories based on search terms
      const recommendedMemories = recommenderService.recommendMemories(
        terms, 
        Object.values(memories)
      );
      
      setSearchResults(recommendedMemories);
      
      // Generate synthetic narrative if enough recommended memories
      if (recommendedMemories.length > 0) {
        const narrative = await openaiService.generateSyntheticMemory(recommendedMemories);
        setSyntheticNarrative(narrative);
      }
      
      setIsLoading(false);
    } catch (err) {
      setError('Search failed');
      setIsLoading(false);
    }
  };

  // Handle memory interaction to increase weight
  const handleMemoryInteraction = (memoryId) => {
    const updatedMemories = { ...memories };
    const memory = updatedMemories[memoryId];
    
    if (memory) {
      // Increment weight
      memory.weight = weightCalculator.incrementWeight(memory.weight);
      
      // Save updated memories
      memoryProcessingService.saveMemoryEvents([memory]);
      setMemories(updatedMemories);
    }
  };

  // Toggle between user and public databases
  const toggleDatabase = () => {
    setActiveDatabase(prev => prev === 'user' ? 'public' : 'user');
  };

  return (
    <div className="container mx-auto p-4">
      <header className="flex justify-between items-center mb-4">
        <h1 className="text-2xl font-bold">Memory Map</h1>
        <button 
          onClick={toggleDatabase}
          className="bg-blue-500 text-white px-4 py-2 rounded"
        >
          Switch to {activeDatabase === 'user' ? 'Public' : 'User'} Memories
        </button>
      </header>

      <div className="grid grid-cols-3 gap-4">
        {/* Memory List */}
        <div className="col-span-1">
          <MemoryList 
            memories={searchResults.length > 0 ? searchResults : Object.values(memories)}
            onMemoryInteract={handleMemoryInteraction}
          />
        </div>
        
        {/* Central Content */}
        <div className="col-span-2">
          {/* Search Bar */}
          <SearchBar onSearch={handleSearch} />
          
          {/* Loading and Error States */}
          {isLoading && <p>Loading...</p>}
          {error && <p className="text-red-500">{error}</p>}
          
          {/* Synthetic Narrative Display */}
          {syntheticNarrative && (
            <NarrativeDisplay 
              narrative={syntheticNarrative}
              searchTerms={searchResults.length > 0 
                ? searchResults.flatMap(m => m.keywords) 
                : []
              }
            />
          )}
          
          {/* Memory Upload Form */}
          <MemoryForm onImageUpload={handleImageUpload} />
        </div>
      </div>
    </div>
  );
}

export default App;