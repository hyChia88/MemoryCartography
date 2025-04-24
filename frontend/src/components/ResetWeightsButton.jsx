// ResetWeightsButton.jsx
import React, { useState } from 'react';

const ResetWeightsButton = ({ memoryType, onResetComplete }) => {
  const [isResetting, setIsResetting] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  
  const handleResetClick = () => {
    setShowConfirm(true);
  };

  const handleConfirmReset = async () => {
    setIsResetting(true);
    
    try {
      // Call the reset API endpoint
      const response = await fetch(
        `/memories/reset_weights_from_json?memory_type=${memoryType}`, 
        { method: 'POST' }
      );
      
      const data = await response.json();
      
      // Notify parent component that reset is complete
      if (onResetComplete) {
        onResetComplete(data);
      }
      
    } catch (error) {
      console.error('Error resetting weights:', error);
    } finally {
      setIsResetting(false);
      setShowConfirm(false);
    }
  };

  return (
    <>
      <button
        onClick={handleResetClick}
        disabled={isResetting}
        className="bg-gray-200 text-gray-800 p-2 ml-2 rounded disabled:opacity-50 flex items-center"
      >
        {isResetting ? (
          <span>Resetting...</span>
        ) : (
          <>
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            Reset Weights
          </>
        )}
      </button>
      
      {/* Confirmation Dialog */}
      {showConfirm && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-4 max-w-sm w-full shadow-xl">
            <h3 className="font-medium mb-2">Confirm Reset</h3>
            <p className="text-sm text-gray-600 mb-4">
              Reset all memory weights to original values? This cannot be undone.
            </p>
            <div className="flex justify-end space-x-2">
              <button
                onClick={() => setShowConfirm(false)}
                className="px-3 py-1 bg-gray-200 rounded text-sm"
              >
                Cancel
              </button>
              <button
                onClick={handleConfirmReset}
                className="px-3 py-1 bg-red-500 text-white rounded text-sm"
              >
                Reset
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default ResetWeightsButton;