// frontend/src/components/GeneratedText.jsx
import React from 'react';

const GeneratedText = ({ generatedContent, isLoading }) => {
  const renderKeyword = (keyword, index) => {
    const colorClasses = {
      primary: 'bg-red-200 text-red-800 font-bold',
      related: 'bg-yellow-200 text-yellow-800',
      connected: 'bg-gray-200 text-gray-800'
    };

    return (
      <span 
        key={`${keyword.text}-${index}`} 
        className={`px-2 py-1 rounded-md mr-2 mb-2 inline-block ${colorClasses[keyword.type]}`}
      >
        {keyword.text}
      </span>
    );
  };

  return (
    <div className="flex-1 bg-gray-50 rounded-lg p-4 overflow-y-auto">
      <h2 className="text-sm font-semibold text-gray-600 mb-3 uppercase tracking-wider">
        Generated Narrative
      </h2>
      
      {isLoading ? (
        <div className="flex justify-center items-center h-32">
          <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500"></div>
        </div>
      ) : generatedContent ? (
        <>
          <div className="mb-4 flex flex-wrap">
            {generatedContent.keywords.map((keyword, index) => renderKeyword(keyword, index))}
          </div>
          <div className="text-gray-700 leading-relaxed whitespace-pre-line">
            {generatedContent.text}
          </div>
        </>
      ) : (
        <p className="text-gray-500 italic text-sm">
          Search a location to generate a memory narrative...
        </p>
      )}
    </div>
  );
};

export default GeneratedText;