// src/services/api.js
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';

export const memoryService = {
  getMemories: async (type = 'user') => {
    try {
      const response = await axios.get(`${API_BASE_URL}/memories?type=${type}`);
      return response.data;
    } catch (error) {
      console.error('Error fetching memories:', error.response ? error.response.data : error.message);
      throw error;
    }
  },
  // Add other methods as needed
};

export default memoryService;  // Add this line to make it a default export