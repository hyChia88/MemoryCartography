import React from 'react';

const MemoryInput = ({ value, onChange, onSubmit, isDisabled }) => {
  return (
    <div className="bg-gray-50 rounded-lg p-4">
      <textarea 
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Add your own memory details..."
        className="w-full px-3 py-2 bg-transparent border-b border-gray-300 focus:outline-none focus:border-blue-500 resize-none"
        rows={4}
      />
      <div className="flex justify-end mt-2">
        <button 
          onClick={onSubmit}
          className={`text-sm font-medium px-3 py-1 rounded ${
            isDisabled 
              ? 'text-gray-400 cursor-not-allowed' 
              : 'text-blue-600 hover:text-blue-800 hover:bg-blue-50'
          }`}
          disabled={isDisabled}
        >
          Add Memory
        </button>
      </div>
    </div>
  );
};

export default MemoryInput;