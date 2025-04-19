import { useState, useEffect } from 'react';
import { fetchMemories } from '../api/services';

export const useMemories = () => {
  const [memories, setMemories] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadMemories = async () => {
      try {
        console.log('Starting to fetch memories...'); // Debugging log
        setIsLoading(true);
        const data = await fetchMemories();
        console.log('Memories fetched:', data); // Debugging log
        setMemories(data);
        setError(null);
      } catch (err) {
        console.error('Failed to fetch memories:', err);
        setError(err);
        setMemories([]); // Ensure a fallback state
      } finally {
        setIsLoading(false);
        console.log('Finished loading memories'); // Debugging log
      }
    };

    loadMemories();
  }, []);

  return { memories, isLoading, error };
};