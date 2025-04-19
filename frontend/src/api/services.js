// src/api/services.js
import apiClient from './index';

/**
 * Fetch memories from the API
 * @param {string} type - Type of memories ('user' or 'public')
 * @returns {Promise<Array>} - List of memories
 */
export const getMemories = async (type = 'user') => {
  try {
    const response = await apiClient.get('/memories', { 
      params: { type } 
    });
    return response.data;
  } catch (error) {
    console.error('Failed to fetch memories:', error);
    throw error;
  }
};

/**
 * Search memories based on a query
 * @param {string} query - Search query
 * @param {string} type - Type of memories ('user' or 'public')
 * @param {number} limit - Maximum number of results
 * @returns {Promise<Array>} - List of matching memories
 */
export const searchMemories = async (query, type = 'user', limit = 10) => {
  try {
    const response = await apiClient.get('/memories/search', {
      params: { 
        query, 
        type, 
        limit 
      }
    });
    return response.data;
  } catch (error) {
    console.error('Failed to search memories:', error);
    throw error;
  }
};

/**
 * Generate a narrative based on a search query
 * @param {string} query - Search query
 * @param {string} type - Type of memories ('user' or 'public')
 * @returns {Promise<Object>} - Generated narrative
 */
export const generateNarrative = async (query, type = 'user') => {
  try {
    const response = await apiClient.get(`/generate/${type}`, {
      params: { query }
    });
    return response.data;
  } catch (error) {
    console.error('Failed to generate narrative:', error);
    throw error;
  }
};

/**
 * Get a single memory by its ID
 * @param {number} id - Memory ID
 * @returns {Promise<Object>} - Memory details
 */
export const getMemoryById = async (id) => {
  try {
    const response = await apiClient.get(`/memories/${id}`);
    return response.data;
  } catch (error) {
    console.error(`Failed to fetch memory with ID ${id}:`, error);
    throw error;
  }
};

/**
 * Record an interaction with a memory
 * @param {number} id - Memory ID
 * @param {string} interactionType - Type of interaction
 * @returns {Promise<Object>} - Interaction result
 */
export const recordInteraction = async (id, interactionType = 'click') => {
  try {
    const response = await apiClient.post(`/memories/${id}/interact`, {
      interaction_type: interactionType
    });
    return response.data;
  } catch (error) {
    console.error('Failed to record interaction:', error);
    throw error;
  }
};

/**
 * Update the weight of a memory
 * @param {number} id - Memory ID
 * @param {number} weight - New weight value
 * @returns {Promise<Object>} - Update result
 */
export const updateMemoryWeight = async (id, weight) => {
  try {
    const response = await apiClient.post(`/memories/${id}/weight`, null, {
      params: { weight }
    });
    return response.data;
  } catch (error) {
    console.error('Failed to update memory weight:', error);
    throw error;
  }
};

/**
 * Find similar memories to a given memory
 * @param {number} id - Memory ID
 * @param {string} type - Type of memories ('user' or 'public')
 * @param {number} limit - Maximum number of results
 * @returns {Promise<Array>} - List of similar memories
 */
export const findSimilarMemories = async (id, type = 'user', limit = 5) => {
  try {
    const response = await apiClient.get(`/memories/similar/${id}`, {
      params: { 
        memory_type: type, 
        limit 
      }
    });
    return response.data;
  } catch (error) {
    console.error('Failed to find similar memories:', error);
    throw error;
  }
};

/**
 * Add a new memory
 * @param {Object} memory - Memory data
 * @returns {Promise<Object>} - Created memory
 */
export const addMemory = async (memory) => {
  try {
    const response = await apiClient.post('/memories', memory);
    return response.data;
  } catch (error) {
    console.error('Failed to add memory:', error);
    throw error;
  }
};