// src/services/api.js
const API_URL = 'http://localhost:8000';

export const fetchMemories = async (type = 'user') => {
  try {
    const response = await fetch(`${API_URL}/memories/?type=${type}`);
    if (!response.ok) throw new Error('Failed to fetch memories');
    return await response.json();
  } catch (error) {
    console.error('Error fetching memories:', error);
    throw error;
  }
};

export const generateNarrative = async (location, type = 'user') => {
  try {
    const encodedLocation = encodeURIComponent(location);
    const response = await fetch(`${API_URL}/generate/${type}/${encodedLocation}`);
    if (!response.ok) throw new Error('Failed to generate narrative');
    return await response.json();
  } catch (error) {
    console.error('Error generating narrative:', error);
    throw error;
  }
};

export const addMemory = async (memoryData) => {
  try {
    const response = await fetch(`${API_URL}/memories/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(memoryData),
    });
    if (!response.ok) throw new Error('Failed to add memory');
    return await response.json();
  } catch (error) {
    console.error('Error adding memory:', error);
    throw error;
  }
};