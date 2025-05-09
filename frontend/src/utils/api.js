import axios from 'axios';

// Create API client
const api = axios.create({
  baseURL: process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000',
  timeout: 15000,
  headers: {
    'Content-Type': 'application/json',
    Accept: 'application/json',
  },
});

/**
 * Function to get a thumbnail URL for a memory
 * 
 * @param {Object} memory - Memory object
 * @returns {string} - URL to the memory's image
 */
export const getThumbnailUrl = (memory) => {
  if (!memory.filename) {
    console.log('No filename found, using placeholder');
    return '/placeholder-image.jpg'; // Fallback image
  }
  
  // Add the API base URL
  const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000';
  const basePath = memory.type === 'user' ? '/user-photos/' : '/public-photos/';
  const url = `${apiUrl}${basePath}${memory.filename}`;
  
  return url;
};

/**
 * Function to search memories
 * 
 * @param {Object} params - Search parameters
 * @param {string} params.sessionId - Session ID
 * @param {string} params.query - Search query
 * @param {string} params.memoryType - Type of memories to search ('user', 'public', 'all')
 * @param {string} params.sortBy - How to sort results ('weight', 'date', 'relevance')
 * @returns {Promise<Array>} - Promise resolving to array of memories
 */
export const searchMemories = async ({
  sessionId,
  query,
  memoryType = 'user',
  sortBy = 'weight',
  limit = 30
}) => {
  if (!sessionId) {
    throw new Error('Session ID is required');
  }

  const response = await api.get('/api/memories/search', {
    params: {
      session_id: sessionId,
      query: query,
      memory_type: memoryType,
      sort_by: sortBy,
      limit: limit
    }
  });

  return response.data;
};

/**
 * Function to generate a narrative
 * 
 * @param {Object} params - Narrative parameters
 * @param {string} params.sessionId - Session ID
 * @param {string} params.query - Search query for narrative
 * @param {string} params.memoryType - Type of memories to include
 * @returns {Promise<Object>} - Promise resolving to narrative data
 */
export const generateNarrative = async ({
  sessionId,
  query,
  memoryType = 'user'
}) => {
  if (!sessionId) {
    throw new Error('Session ID is required');
  }

  const response = await api.get('/api/memories/narrative', {
    params: {
      session_id: sessionId,
      query: query,
      memory_type: memoryType
    }
  });

  return response.data;
};

/**
 * Function to adjust a memory's weight
 * 
 * @param {Object} params - Adjustment parameters
 * @param {string} params.sessionId - Session ID
 * @param {number} params.memoryId - ID of the memory to adjust
 * @param {number} params.adjustment - Weight adjustment value
 * @returns {Promise<Object>} - Promise resolving to adjustment result
 */
export const adjustMemoryWeight = async ({
  sessionId,
  memoryId,
  adjustment
}) => {
  if (!sessionId || !memoryId) {
    throw new Error('Session ID and memory ID are required');
  }

  const response = await api.post(
    `/api/memories/${memoryId}/adjust_weight`,
    {},
    {
      params: {
        session_id: sessionId,
        adjustment: adjustment
      }
    }
  );

  return response.data;
};

/**
 * Function to reset memory weights
 * 
 * @param {Object} params - Reset parameters
 * @param {string} params.memoryType - Type of memories to reset
 * @returns {Promise<Object>} - Promise resolving to reset result
 */
export const resetMemoryWeights = async ({
  memoryType = 'user'
}) => {
  const response = await api.post('/memories/reset_weights', {
    memory_type: memoryType
  });

  return response.data;
};

export default api;