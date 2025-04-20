import React, { useState } from 'react';
import axios from 'axios';
interface Memory {
id: number;
title: string;
location: string;
date: string;
keywords: string[];
type: string;
description?: string;
}
const MemoryApp: React.FC = () => {
const [memories, setMemories] = useState<Memory[]>([]);
const [narrative, setNarrative] = useState<string>('');
const [searchTerm, setSearchTerm] = useState<string>('');
const [error, setError] = useState<string | null>(null);
const [loading, setLoading] = useState(false);
const [memoryType, setMemoryType] = useState<'user' | 'public'>('user');
const [highlightedKeywords, setHighlightedKeywords] = useState<string[]>([]);
// Configure axios with full URL and timeout
const api = axios.create({
baseURL: 'http://localhost:8000',
timeout: 5000,
headers: {
'Content-Type': 'application/json',
'Accept': 'application/json'
}
});
const searchMemories = async () => {
if (!searchTerm) {
setError('Please enter a search term');
return;
}
setLoading(true);
setError(null);

try {
  const response = await api.get('/memories/search', {
    params: { 
      query: searchTerm,
      memory_type: memoryType
    }
  });
  
  setMemories(response.data);
} catch (err) {
  console.error('Search error', err);
  setError(err instanceof Error ? err.message : 'An unknown error occurred');
} finally {
  setLoading(false);
}
};
const generateNarrative = async () => {
if (!searchTerm) {
setError('Please enter a search term');
return;
}
setLoading(true);
setError(null);

try {
  const response = await api.get('/memories/narrative', {
    params: { 
      query: searchTerm,
      memory_type: memoryType
    }
  });
  
  setNarrative(response.data.text);
  // Extract direct related keywords
  const keywords = response.data.keywords
    .filter((kw: any) => kw.type === 'primary')
    .map((kw: any) => kw.text);
  setHighlightedKeywords(keywords);
} catch (err) {
  console.error('Narrative generation error', err);
  setError(err instanceof Error ? err.message : 'An unknown error occurred');
} finally {
  setLoading(false);
}
};
// Helper function to highlight keywords in narrative
const highlightNarrative = () => {
if (!narrative) return narrative;
let highlightedText = narrative;
highlightedKeywords.forEach(keyword => {
  // Create a regex that matches the keyword as a whole word
  const regex = new RegExp(`\\b${keyword}\\b`, 'gi');
  highlightedText = highlightedText.replace(
    regex, 
    `<span class="text-red-600 font-bold">$&</span>`
  );
});

return highlightedText;
};
return (
<div className="bg-white min-h-screen p-4">
<div className="container mx-auto max-w-4xl">
<h1 className="text-3xl font-bold mb-4 text-gray-800">Memory Cartography</h1>
    {/* Memory Type Toggle */}
    <div className="flex justify-center mb-4">
      <div className="bg-gray-100 rounded-full p-1 flex">
        <button
          onClick={() => setMemoryType('user')}
          className={`px-4 py-2 rounded-full transition-colors ${
            memoryType === 'user' 
              ? 'bg-yellow-400 text-gray-800' 
              : 'text-gray-600 hover:bg-gray-200'
          }`}
        >
          User Memories
        </button>
        <button
          onClick={() => setMemoryType('public')}
          className={`px-4 py-2 rounded-full transition-colors ${
            memoryType === 'public' 
              ? 'bg-yellow-400 text-gray-800' 
              : 'text-gray-600 hover:bg-gray-200'
          }`}
        >
          Public Memories
        </button>
      </div>
    </div>
    
    {/* Search Bar */}
    <div className="mb-4 flex">
      <input 
        type="text" 
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
        placeholder={`Search ${memoryType} memories...`}
        className="flex-grow p-2 border rounded-l bg-white text-gray-800"
      />
      <button 
        onClick={searchMemories} 
        disabled={loading}
        className="bg-yellow-400 text-gray-800 p-2 rounded-r disabled:opacity-50"
      >
        {loading ? 'Searching...' : 'Search'}
      </button>
      <button 
        onClick={generateNarrative}
        disabled={loading}
        className="bg-yellow-500 text-gray-800 p-2 ml-2 rounded disabled:opacity-50"
      >
        {loading ? 'Generating...' : 'Generate Narrative'}
      </button>
    </div>

    {/* Error Handling */}
    {error && (
      <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-4">
        <span className="block sm:inline">{error}</span>
      </div>
    )}

    {/* Narrative Display */}
    {narrative && (
      <div className="bg-yellow-50 p-4 rounded mb-4">
        <h2 className="font-bold mb-2 text-gray-800">Generated Narrative</h2>
        <p 
          className="text-gray-800"
          dangerouslySetInnerHTML={{ __html: highlightNarrative() }}
        />
      </div>
    )}

    {/* Memory List */}
    <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
      {memories.map(memory => (
        <div 
          key={memory.id} 
          className="bg-white border rounded p-3 hover:shadow-lg transition-shadow"
        >
          <h2 className="font-bold text-gray-800">{memory.title}</h2>
          <p className="text-sm text-gray-600">{memory.location}</p>
          <p className="text-sm text-gray-600">{memory.date}</p>
          <div className="mt-2">
            {memory.keywords.slice(0,3).map(keyword => (
              <span 
                key={keyword} 
                className="inline-block bg-yellow-100 text-yellow-800 text-xs px-2 py-1 rounded mr-1 mb-1"
              >
                {keyword}
              </span>
            ))}
          </div>
        </div>
      ))}
    </div>
  </div>
</div>
);
};
export default MemoryApp;