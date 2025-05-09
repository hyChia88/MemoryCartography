import React, { useState } from 'react';

/**
 * Search bar component with additional action buttons
 * 
 * @param {Object} props
 * @param {string} props.initialValue - Initial search term value
 * @param {Function} props.onSearch - Callback for search submission
 * @param {Function} props.onGenerateNarrative - Callback for narrative generation
 * @param {boolean} props.isLoading - Loading state for search button
 * @param {boolean} props.isGeneratingNarrative - Loading state for generate narrative button
 * @param {string} props.memoryType - Current memory type filter
 * @param {Function} props.onMemoryTypeChange - Handler for memory type change
 * @param {React.ReactNode} props.extraButtons - Additional buttons to display
 * @returns {JSX.Element} - Search bar component with memory type filters
 */
const SearchBar = ({
  initialValue = '',
  onSearch,
  onGenerateNarrative,
  isLoading = false,
  isGeneratingNarrative = false,
  memoryType = 'user',
  onMemoryTypeChange,
  extraButtons
}) => {
  const [searchTerm, setSearchTerm] = useState(initialValue);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (onSearch) {
      onSearch(searchTerm);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSubmit(e);
    }
  };

  return (
    <div className="mb-4">
      {/* Memory Type Toggle */}
      <div className="flex justify-center mb-4">
        <div className="bg-gray-100 rounded-full p-1 flex">
          <button
            onClick={() => onMemoryTypeChange && onMemoryTypeChange('user')}
            className={`px-4 py-2 rounded-full transition-colors ${
              memoryType === 'user'
                ? 'bg-gray-400 text-gray-800'
                : 'text-gray-600 hover:bg-gray-200'
            }`}
          >
            User Memories
          </button>
          <button
            onClick={() => onMemoryTypeChange && onMemoryTypeChange('public')}
            className={`px-4 py-2 rounded-full transition-colors ${
              memoryType === 'public'
                ? 'bg-gray-400 text-gray-800'
                : 'text-gray-600 hover:bg-gray-200'
            }`}
          >
            Public Memories
          </button>
        </div>
      </div>
      
      {/* Search Bar and Buttons */}
      <div className="flex flex-wrap items-center gap-2">
        <input
          type="text"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          placeholder={`Search ${memoryType} memories...`}
          className="flex-grow p-2 border rounded bg-white text-gray-800"
          onKeyPress={handleKeyPress}
        />
        
        <div className="flex flex-wrap gap-2">
          <button
            onClick={handleSubmit}
            disabled={isLoading}
            className="bg-gray-200 text-gray-800 p-2 rounded disabled:opacity-50"
          >
            {isLoading ? 'Searching...' : 'Search'}
          </button>
          
          {onGenerateNarrative && (
            <button
              onClick={() => onGenerateNarrative(searchTerm)}
              disabled={isGeneratingNarrative}
              className="bg-gray-400 text-gray-800 p-2 rounded disabled:opacity-50"
            >
              {isGeneratingNarrative ? 'Generating...' : 'Generate Narrative'}
            </button>
          )}
          
          {/* Render any extra buttons */}
          {extraButtons}
        </div>
      </div>
    </div>
  );
};

export default SearchBar;