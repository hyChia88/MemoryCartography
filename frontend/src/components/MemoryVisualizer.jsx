// src/components/MemoryVisualizer.jsx
import React from 'react';
import KeywordTag from './KeywordTag';

const MemoryCard = ({ memory, isSmall = false }) => {
  return (
    <div className={`flex-shrink-0 ${isSmall ? 'w-36' : 'w-40'}`}>
      <div className="border border-gray-200 rounded overflow-hidden shadow-sm hover:shadow-md transition-shadow">
        <div className="relative">
          <img 
            src={`/${memory.image_path}`} 
            alt={memory.title} 
            className={`w-full ${isSmall ? 'h-28' : 'h-32'} object-cover`} 
          />
          {memory.connection_keyword && (
            <div className="absolute top-1 right-1">
              <KeywordTag keyword={memory.connection_keyword} type="connected" />
            </div>
          )}
        </div>
        <div className="p-2">
          <p className="text-xs font-medium truncate">{memory.title}</p>
          <p className="text-xs text-gray-500">{memory.location}</p>
          <p className="text-xs text-gray-400">{memory.date}</p>
        </div>
      </div>
    </div>
  );
};

const MemoryVisualizer = ({ primaryMemories, connectedMemories, activeType }) => {
  if (!primaryMemories?.length && !connectedMemories?.length) {
    return (
      <div className="bg-gray-50 rounded-lg p-4 flex flex-col items-center justify-center min-h-[200px]">
        <p className="text-gray-500 italic">
          Search for a location to visualize memory connections...
        </p>
      </div>
    );
  }

  // Group connected memories by their connection keyword
  const connectionGroups = {};
  connectedMemories.forEach(memory => {
    const keyword = memory.connection_keyword || 'unknown';
    if (!connectionGroups[keyword]) {
      connectionGroups[keyword] = [];
    }
    connectionGroups[keyword].push(memory);
  });

  return (
    <div className="bg-gray-50 rounded-lg p-4">
      <h2 className="text-sm font-semibold text-gray-600 mb-3 uppercase tracking-wider">
        Memory Connections
      </h2>
      
      {/* Primary Memories */}
      <div className="mb-6">
        <h3 className="text-sm text-gray-700 mb-2">
          Primary Memories ({activeType === 'user' ? 'Your Experiences' : 'Collective Experiences'})
        </h3>
        <div className="flex overflow-x-auto space-x-4 pb-2">
          {primaryMemories.map(memory => (
            <MemoryCard key={memory.id} memory={memory} />
          ))}
        </div>
      </div>
      
      {/* Connected Memories */}
      {Object.keys(connectionGroups).length > 0 && (
        <div>
          <h3 className="text-sm text-gray-700 mb-2">
            Connected Through
          </h3>
          
          {/* Connection Groups */}
          {Object.entries(connectionGroups).map(([keyword, memories]) => (
            <div key={keyword} className="mb-4">
              <div className="flex items-center mb-2">
                <span className="px-2 py-1 bg-yellow-200 text-yellow-800 text-xs rounded-full">
                  {keyword}
                </span>
                <div className="ml-2 flex-grow border-t border-dashed border-gray-300"></div>
              </div>
              
              <div className="flex overflow-x-auto space-x-4 pb-2 pl-6">
                {memories.map(memory => (
                  <MemoryCard key={memory.id} memory={memory} isSmall={true} />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
      
      {/* Description of memory processes */}
      <div className="mt-4 pt-4 border-t border-gray-200">
        <p className="text-xs text-gray-600 italic">
          {activeType === 'user' ? 
            "This visualization shows how your personal memories connect across time and space, revealing how memory stacks and distorts over time." :
            "This visualization shows how collective memories form patterns, demonstrating how different people remember the same places uniquely."
          }
        </p>
      </div>
    </div>
  );
};

export default MemoryVisualizer;