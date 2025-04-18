import React from 'react';

const GeneratedText = ({ generatedText, keywords }) => {
  const renderKeyword = (keyword) => {
    const colorClasses = {
      primary: 'bg-red-200 text-red-800 font-bold',
      related: 'bg-yellow-200 text-yellow-800',
      connected: 'bg-gray-200 text-gray-800'
    };

    return (
      <span 
        key={keyword.text} 
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
      {generatedText ? (
        <>
          <div className="mb-4 flex flex-wrap">
            {keywords.map(renderKeyword)}
          </div>
          <p className="text-gray-700 leading-relaxed">{generatedText}</p>
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