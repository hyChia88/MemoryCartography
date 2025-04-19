import React from 'react';
import { MemoryProvider } from './context/MemoryContext';
import Header from './components/Header';
import MemoryList from './components/MemoryList';
import SearchBar from './components/SearchBar';
import NarrativeDisplay from './components/NarrativeDisplay';
import MemoryForm from './components/MemoryForm';
import LoadingSpinner from './components/LoadingSpinner';
import { useMemory } from './context/MemoryContext';

// App content component that uses context
const AppContent = () => {
  const { isLoading, error } = useMemory();

  return (
    <div className="min-h-screen bg-white flex flex-col p-4 max-w-7xl mx-auto">
      {/* Header */}
      <Header />
      
      {/* Main Content */}
      <div className="flex flex-1 space-x-4">
        {/* Memories List */}
        <div className="w-1/4 bg-gray-50 rounded-lg p-3 overflow-y-auto">
          <MemoryList />
        </div>
        
        {/* Central Content Area */}
        <div className="flex-1 flex flex-col space-y-4">
          {/* Search Bar */}
          <SearchBar />
          
          {/* Generated Text Area */}
          {isLoading ? (
            <div className="flex-1 flex items-center justify-center">
              <LoadingSpinner />
            </div>
          ) : (
            <NarrativeDisplay />
          )}
          
          {/* Display error if any */}
          {error && (
            <div className="bg-red-50 text-red-600 p-3 rounded-md">
              {error}
            </div>
          )}
          
          {/* Input Area */}
          <MemoryForm />
        </div>
      </div>
    </div>
  );
};

// Main App component with Provider
const App = () => {
  return (
    <MemoryProvider>
      <AppContent />
    </MemoryProvider>
  );
};

export default App;