// src/components/KeywordTag.jsx
import React from 'react';

const KeywordTag = ({ keyword, type }) => {
  const colorClasses = {
    primary: 'bg-red-200 text-red-800 font-bold',
    related: 'bg-yellow-200 text-yellow-800',
    connected: 'bg-gray-200 text-gray-800'
  };

  return (
    <span 
      className={`px-2 py-1 rounded-md mr-2 mb-2 inline-block text-sm ${colorClasses[type]}`}
    >
      {keyword}
    </span>
  );
};

export default KeywordTag;