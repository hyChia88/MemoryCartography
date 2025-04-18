import axios from 'axios';

const API_URL = 'http://localhost:8000';

// Create axios instance with base URL
const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  }
});

const memoryService = {
  // Get all memories of a specific type
  getMemories: async (type = 'user') => {
    try {
      const response = await api.get(`/memories/?type=${type}`);
      return response.data;
    } catch (error) {
      console.error('Error fetching memories:', error);
      throw error;
    }
  },
  
  // Add a new memory
  addMemory: async (memoryData) => {
    try {
      const response = await api.post('/memories/', memoryData);
      return response.data;
    } catch (error) {
      console.error('Error adding memory:', error);
      throw error;
    }
  },
  
  // Generate narrative for a location
  generateNarrative: async (location, databaseType = 'user') => {
    try {
      const response = await api.post('/generate/', {
        location,
        database_type: databaseType,
      });
      return response.data;
    } catch (error) {
      console.error('Error generating narrative:', error);
      throw error;
    }
  },
  
  // Seed sample data
  seedData: async () => {
    try {
      const response = await api.post('/memories/seed');
      return response.data;
    } catch (error) {
      console.error('Error seeding data:', error);
      throw error;
    }
  }
};

export default memoryService;