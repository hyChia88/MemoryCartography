import React from 'react';
import { useMemory } from '../context/MemoryContext';

const KeywordTag = ({ keyword }) => {
  const { handleSearch } = useMemory();
  
  // Determine tag style based on type
  const typeStyles = {
    primary: 'bg-red-200 text-red-800 font-bold',
    related: 'bg-yellow-200 text-yellow-800',
    connected: 'bg-gray-200 text-gray-800'
  };
  
  // Get style based on keyword type, default to connected style
  const style = typeStyles[keyword.type] || typeStyles.connected;
  
  // Handle click on tag to perform a new search
  const handleClick = () => {
    handleSearch(keyword.text);
  };
  
  return (
    <span 
      onClick={handleClick}
      className={`px-2 py-1 rounded-md mr-2 mb-2 inline-block cursor-pointer hover:brightness-95 transition-all ${style}`}
      title={`Search for ${keyword.text}`}
    >
      {keyword.text}
    </span>
  );
};

export default KeywordTag;