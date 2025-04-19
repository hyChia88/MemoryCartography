import React, { useState } from 'react';
import { Search } from 'lucide-react';
import { useMemory } from '../context/MemoryContext';

const SearchBar = () => {
  const { searchQuery, handleSearch, isLoading } = useMemory();
  const [inputValue, setInputValue] = useState(searchQuery);
  
  const onSubmit = (e) => {
    e.preventDefault();
    handleSearch(inputValue);
  };
  
  return (
    <form onSubmit={onSubmit} className="relative">
      <input 
        type="text" 
        value={inputValue}
        onChange={(e) => setInputValue(e.target.value)}
        placeholder="Enter a location or keyword to explore memories..."
        className="w-full px-3 py-2 border-b border-gray-300 focus:outline-none focus:border-blue-500 text-lg"
        disabled={isLoading}
      />
      <button 
        type="submit"
        className="absolute right-0 top-1/2 transform -translate-y-1/2 text-gray-500 hover:text-gray-700 disabled:opacity-50"
        disabled={!inputValue.trim() || isLoading}
      >
        <Search className={isLoading ? "animate-pulse" : ""} />
      </button>
    </form>
  );
};

export default SearchBar;