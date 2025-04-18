import React from 'react';

const MemoryList = ({ memories, onMemoryClick }) => {
  if (!memories || memories.length === 0) {
    return (
      <div className="text-sm text-gray-500 italic">
        No memories found. Try adding some.
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {memories.map(memory => (
        <div 
          key={memory.id} 
          className="group cursor-pointer hover:bg-gray-100 transition-colors duration-200 p-2 rounded-md"
          onClick={() => onMemoryClick(memory.location)}
        >
          <div className="flex justify-between items-center">
            <div>
              <p className="text-sm font-medium text-gray-800 group-hover:text-blue-600">
                {memory.title}
              </p>
              <p className="text-xs text-gray-500">
                {memory.location} | {memory.date}
              </p>
            </div>
          </div>
          {memory.keywords && memory.keywords.length > 0 && (
            <div className="mt-1 flex flex-wrap">
              {memory.keywords.slice(0, 3).map(kw => (
                <span key={kw} className="text-xs bg-gray-200 text-gray-700 px-1 rounded mr-1 mb-1">
                  {kw}
                </span>
              ))}
              {memory.keywords.length > 3 && (
                <span className="text-xs text-gray-500">+{memory.keywords.length - 3} more</span>
              )}
            </div>
          )}
        </div>
      ))}
    </div>
  );
};

export default MemoryList;