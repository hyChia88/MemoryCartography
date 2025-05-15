// src/services/api.js
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';

// Create an axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 15000,
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json'
  }
});

// Session management service
export const sessionService = {
  // Create a new session
  createSession: async () => {
    try {
      const response = await apiClient.post('/api/session/create');
      return response.data;
    } catch (error) {
      console.error('Error creating session:', error.response ? error.response.data : error.message);
      throw error;
    }
  },

  // Get session status
  getSessionStatus: async (sessionId) => {
    try {
      const response = await apiClient.get(`/api/session/${sessionId}/status`);
      return response.data;
    } catch (error) {
      console.error('Error fetching session status:', error.response ? error.response.data : error.message);
      throw error;
    }
  },

  // Delete a session
  deleteSession: async (sessionId) => {
    try {
      const response = await apiClient.delete(`/api/session/${sessionId}`);
      return response.data;
    } catch (error) {
      console.error('Error deleting session:', error.response ? error.response.data : error.message);
      throw error;
    }
  }
};

// File upload service
export const uploadService = {
  // Upload photos
  uploadPhotos: async (sessionId, files) => {
    try {
      const formData = new FormData();
      files.forEach(file => {
        formData.append('files', file);
      });

      const response = await axios({
        method: 'post',
        url: `${API_BASE_URL}/api/upload/photos?session_id=${sessionId}`,
        data: formData,
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      return response.data;
    } catch (error) {
      console.error('Error uploading photos:', error.response ? error.response.data : error.message);
      throw error;
    }
  },

  // Process uploaded photos
  processPhotos: async (sessionId) => {
    try {
      const response = await apiClient.post('/api/upload/process', {
        session_id: sessionId
      });
      return response.data;
    } catch (error) {
      console.error('Error processing photos:', error.response ? error.response.data : error.message);
      throw error;
    }
  },

  // Fetch public photos based on detected locations
  fetchPublicPhotos: async (sessionId, maxPhotosPerLocation = 10, totalLimit = 100) => {
    try {
      const response = await apiClient.post('/api/upload/fetch-public', {
        session_id: sessionId,
        max_photos_per_location: maxPhotosPerLocation,
        total_limit: totalLimit
      });
      return response.data;
    } catch (error) {
      console.error('Error fetching public photos:', error.response ? error.response.data : error.message);
      throw error;
    }
  },

  // Check processing status
  getUploadStatus: async (sessionId) => {
    try {
      const response = await apiClient.get(`/api/upload/status/${sessionId}`);
      return response.data;
    } catch (error) {
      console.error('Error checking upload status:', error.response ? error.response.data : error.message);
      throw error;
    }
  }
};

// Memory service
export const memoryService = {
  // Get all memories of a specific type
  getMemories: async (type = 'user') => {
    try {
      const response = await apiClient.get(`/memories?type=${type}`);
      return response.data;
    } catch (error) {
      console.error('Error fetching memories:', error.response ? error.response.data : error.message);
      throw error;
    }
  },

  // Search memories
  searchMemories: async (sessionId, query, memoryType = 'all', sortBy = 'weight', limit = 30) => {
    try {
      const response = await apiClient.get('/api/memories/search', {
        params: {
          session_id: sessionId,
          query,
          memory_type: memoryType,
          sort_by: sortBy,
          limit
        }
      });
      return response.data;
    } catch (error) {
      console.error('Error searching memories:', error.response ? error.response.data : error.message);
      throw error;
    }
  },

  // Get similar memories
  getSimilarMemories: async (sessionId, memoryId, limit = 5, memoryType = 'all') => {
    try {
      const response = await apiClient.get(`/api/memories/similar/${memoryId}`, {
        params: {
          session_id: sessionId,
          limit,
          memory_type: memoryType
        }
      });
      return response.data;
    } catch (error) {
      console.error('Error fetching similar memories:', error.response ? error.response.data : error.message);
      throw error;
    }
  },

  // Generate narrative for a search query
  generateNarrative: async (sessionId, query, memoryType = 'all', maxMemories = 5) => {
    try {
      const response = await apiClient.get('/api/memories/narrative', {
        params: {
          session_id: sessionId,
          query,
          memory_type: memoryType,
          max_memories: maxMemories
        }
      });
      return response.data;
    } catch (error) {
      console.error('Error generating narrative:', error.response ? error.response.data : error.message);
      throw error;
    }
  },

  // Get a single memory by ID
  getMemoryById: async (sessionId, memoryId) => {
    try {
      const response = await apiClient.get(`/api/memories/${memoryId}`, {
        params: {
          session_id: sessionId
        }
      });
      return response.data;
    } catch (error) {
      console.error('Error fetching memory:', error.response ? error.response.data : error.message);
      throw error;
    }
  },

  // Adjust memory weight
  adjustMemoryWeight: async (sessionId, memoryId, adjustment) => {
    try {
      const response = await apiClient.post(`/api/memories/${memoryId}/adjust_weight`, null, {
        params: {
          session_id: sessionId,
          adjustment
        }
      });
      return response.data;
    } catch (error) {
      console.error('Error adjusting memory weight:', error.response ? error.response.data : error.message);
      throw error;
    }
  },

  // Update these convenience methods to use the correct endpoint:
  increaseMemoryWeight: async (sessionId, memoryId) => {
    try {
      const response = await apiClient.post(`/api/memories/${memoryId}/adjust_weight`, null, {
        params: {
          session_id: sessionId,
          adjustment: 0.1
        }
      });
      return response.data;
    } catch (error) {
      console.error('Error increasing memory weight:', error.response ? error.response.data : error.message);
      throw error;
    }
  },

  decreaseMemoryWeight: async (sessionId, memoryId) => {
    try {
      const response = await apiClient.post(`/api/memories/${memoryId}/adjust_weight`, null, {
        params: {
          session_id: sessionId,
          adjustment: -0.1
        }
      });
      return response.data;
    } catch (error) {
      console.error('Error decreasing memory weight:', error.response ? error.response.data : error.message);
      throw error;
    }
  },

  // Reset memory weights
  resetMemoryWeights: async (memoryType = 'user') => {
    try {
      const response = await apiClient.post('/memories/reset_weights', {
        memory_type: memoryType
      });
      return response.data;
    } catch (error) {
      console.error('Error resetting memory weights:', error.response ? error.response.data : error.message);
      throw error;
    }
  },

  // Get a thumbnail URL for a memory
  getThumbnailUrl: (memory) => {
    if (!memory || !memory.filename) {
      return '/placeholder-image.jpg'; // Fallback image
    }
    
    const basePath = memory.type === 'user' ? '/user-photos/' : '/public-photos/';
    return `${API_BASE_URL}${basePath}${memory.filename}`;
  }
};

// Export a combined object with all services
const apiService = {
  session: sessionService,
  upload: uploadService,
  memory: memoryService
};

export default apiService;