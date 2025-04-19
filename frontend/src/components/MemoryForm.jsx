import React, { useState } from 'react';
import { addMemory } from '../api/services';
import { useMemory } from '../context/MemoryContext';

const MemoryForm = () => {
  const { activeDatabase } = useMemory();
  const [isExpanded, setIsExpanded] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);
  
  // Form state
  const [formData, setFormData] = useState({
    title: '',
    location: '',
    date: new Date().toISOString().split('T')[0], // Default to today
    description: '',
    keywords: '',
    image: null
  });
  
  // Handle input changes
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };
  
  // Handle file input
  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFormData(prev => ({ ...prev, image: e.target.files[0] }));
    }
  };
  
  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError(null);
    setSuccess(false);
    
    try {
      // Format the data for the API
      const keywords = formData.keywords
        .split(',')
        .map(kw => kw.trim())
        .filter(kw => kw.length > 0);
      
      // Create a temporary filename for the image
      // In a real app, you'd upload the image to a server
      const filename = formData.image 
        ? `${activeDatabase}_${Date.now()}_${formData.image.name}` 
        : null;
      
      // Create the memory
      const memoryData = {
        title: formData.title,
        location: formData.location,
        date: formData.date,
        description: formData.description,
        keywords,
        filename,
        type: activeDatabase
      };
      
      // Add the memory
      await addMemory(memoryData);
      
      // Show success message and reset form
      setSuccess(true);
      setFormData({
        title: '',
        location: '',
        date: new Date().toISOString().split('T')[0],
        description: '',
        keywords: '',
        image: null
      });
      
      // Collapse the form after submission
      setTimeout(() => {
        setIsExpanded(false);
        setSuccess(false);
      }, 3000);
      
    } catch (err) {
      setError('Failed to add memory. Please try again.');
      console.error(err);
    } finally {
      setIsSubmitting(false);
    }
  };
  
  // Render a simple form or expanded form based on state
  if (!isExpanded) {
    return (
      <div className="bg-gray-50 rounded-lg p-4">
        <button 
          onClick={() => setIsExpanded(true)}
          className="w-full px-3 py-2 bg-transparent border-b border-gray-300 focus:outline-none focus:border-blue-500 text-gray-500 text-left"
        >
          Add your own memory...
        </button>
      </div>
    );
  }
  
  return (
    <div className="bg-gray-50 rounded-lg p-4">
      <h2 className="text-sm font-semibold text-gray-600 mb-3 uppercase tracking-wider">
        Add New Memory
      </h2>
      
      {success && (
        <div className="mb-3 p-2 bg-green-100 text-green-700 rounded-md">
          Memory added successfully!
        </div>
      )}
      
      {error && (
        <div className="mb-3 p-2 bg-red-100 text-red-700 rounded-md">
          {error}
        </div>
      )}
      
      <form onSubmit={handleSubmit}>
        <div className="space-y-3">
          <div>
            <label className="block text-xs text-gray-600 mb-1">Title</label>
            <input 
              type="text"
              name="title"
              value={formData.title}
              onChange={handleChange}
              required
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-1 focus:ring-blue-500"
              placeholder="Give your memory a title"
            />
          </div>
          
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-gray-600 mb-1">Location</label>
              <input 
                type="text"
                name="location"
                value={formData.location}
                onChange={handleChange}
                required
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-1 focus:ring-blue-500"
                placeholder="Where did this happen?"
              />
            </div>
            
            <div>
              <label className="block text-xs text-gray-600 mb-1">Date</label>
              <input 
                type="date"
                name="date"
                value={formData.date}
                onChange={handleChange}
                required
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-1 focus:ring-blue-500"
              />
            </div>
          </div>
          
          <div>
            <label className="block text-xs text-gray-600 mb-1">Description</label>
            <textarea 
              name="description"
              value={formData.description}
              onChange={handleChange}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-1 focus:ring-blue-500 resize-none"
              rows={3}
              placeholder="Describe your memory..."
            />
          </div>
          
          <div>
            <label className="block text-xs text-gray-600 mb-1">Keywords (comma separated)</label>
            <input 
              type="text"
              name="keywords"
              value={formData.keywords}
              onChange={handleChange}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-1 focus:ring-blue-500"
              placeholder="e.g. sunset, beach, relaxation"
            />
          </div>
          
          <div>
            <label className="block text-xs text-gray-600 mb-1">Image (optional)</label>
            <input 
              type="file"
              name="image"
              onChange={handleFileChange}
              accept="image/*"
              className="w-full text-sm text-gray-500"
            />
          </div>
          
          <div className="flex justify-end space-x-2">
            <button
              type="button"
              onClick={() => setIsExpanded(false)}
              className="px-4 py-2 text-sm text-gray-600 hover:text-gray-800"
              disabled={isSubmitting}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-4 py-2 text-sm bg-blue-500 text-white rounded-md hover:bg-blue-600 disabled:opacity-50"
              disabled={isSubmitting}
            >
              {isSubmitting ? 'Adding...' : 'Add Memory'}
            </button>
          </div>
        </div>
      </form>
    </div>
  );
};

export default MemoryForm;