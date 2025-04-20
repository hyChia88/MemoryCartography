import React from 'react';

function MemoryList({ memories, onMemoryInteract }) {
  const handleMemoryClick = (memoryId) => {
    if (onMemoryInteract) {
      onMemoryInteract(memoryId);
    }
  };

  return (
    <div className="bg-gray-50 rounded-lg p-4">
      <h2 className="text-xl font-semibold mb-4 text-gray-800">Memory List</h2>
      
      {memories.length === 0 ? (
        <p className="text-gray-500 italic">No memories found</p>
      ) : (
        <ul className="space-y-3">
          {memories.map((memory) => (
            <li 
              key={memory.id || memory.original_path}
              onClick={() => handleMemoryClick(memory.id)}
              className="bg-white rounded-md shadow-sm p-3 cursor-pointer hover:bg-blue-50 transition-colors"
            >
              <div className="flex justify-between items-start">
                <div>
                  <h3 className="font-bold text-gray-800">
                    {memory.location || 'Unnamed Location'}
                  </h3>
                  <p className="text-sm text-gray-600">
                    {memory.date}
                  </p>
                </div>
                <span 
                  className={`px-2 py-1 rounded-full text-xs font-semibold 
                    ${memory.weight > 5 ? 'bg-green-200 text-green-800' : 'bg-gray-200 text-gray-800'}`}
                >
                  Weight: {memory.weight.toFixed(1)}
                </span>
              </div>
              
              {memory.keywords && (
                <div className="mt-2 flex flex-wrap gap-1">
                  {memory.keywords.slice(0, 5).map((keyword, index) => (
                    <span 
                      key={index} 
                      className="bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded"
                    >
                      {keyword}
                    </span>
                  ))}
                </div>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default MemoryList;