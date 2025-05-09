import React, { useState } from 'react';
import axios from 'axios';

/**
 * Button component to reset memory weights
 * Simplified version with no external dependencies
 */
const ResetWeightsButton = ({ memoryType = 'user', onResetComplete }) => {
  const [isResetting, setIsResetting] = useState(false);
  
  // API client
  const api = axios.create({
    baseURL: process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000',
    timeout: 15000,
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
  });
  
  const resetWeights = async () => {
    // Confirm before resetting
    if (!window.confirm(`Are you sure you want to reset all ${memoryType} memory weights to their default values?`)) {
      return;
    }
    
    setIsResetting(true);
    
    try {
      // Call API to reset weights
      const response = await api.post('/memories/reset_weights', {
        memory_type: memoryType
      });
      
      // Notify parent component of the result
      if (onResetComplete) {
        onResetComplete(response.data);
      }
    } catch (error) {
      console.error('Error resetting weights:', error);
      
      // Notify parent component of the error
      if (onResetComplete) {
        onResetComplete({
          status: 'error',
          message: error.message || 'Failed to reset weights',
          updated_count: 0
        });
      }
    } finally {
      setIsResetting(false);
    }
  };
  
  return (
    <button
      onClick={resetWeights}
      disabled={isResetting}
      className="bg-gray-200 text-gray-800 p-2 rounded disabled:opacity-50 hover:bg-gray-300"
      title={`Reset all ${memoryType} memory weights to default values`}
    >
      {isResetting ? 'Resetting...' : 'Reset Weights'}
    </button>
  );
};

export default ResetWeightsButton;