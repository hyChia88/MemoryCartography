import React from 'react';
import { ToggleLeft, ToggleRight, Database } from 'lucide-react';
import { useMemory } from '../context/MemoryContext';

const Header = () => {
  const { activeDatabase, toggleDatabase } = useMemory();

  return (
    <header className="flex justify-between items-center mb-4 pb-3 border-b border-gray-200">
      <div className="flex items-center">
        <Database className="mr-2 text-indigo-600" size={24} />
        <h1 className="text-2xl font-light text-gray-800">Memory Cartography</h1>
      </div>
      
      <div className="flex items-center space-x-2">
        <button 
          onClick={toggleDatabase} 
          className="flex items-center space-x-2 text-gray-600 hover:text-gray-900 px-3 py-1 rounded-full transition-all border border-gray-200 hover:border-gray-300"
        >
          {activeDatabase === 'user' ? (
            <>
              <ToggleLeft className="text-blue-500" />
              <span className="text-sm">User Memories</span>
            </>
          ) : (
            <>
              <ToggleRight className="text-green-500" />
              <span className="text-sm">Public Memories</span>
            </>
          )}
        </button>
      </div>
    </header>
  );
};

export default Header;