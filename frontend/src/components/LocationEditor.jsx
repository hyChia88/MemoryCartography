import React, { useState, useRef, useEffect } from 'react';

const LocationEditor = ({ memory, onLocationUpdate, sessionId }) => {
  const [isEditing, setIsEditing] = useState(false);
  const [newLocation, setNewLocation] = useState(memory.location || '');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);

  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [isEditing]);

  const handleSave = async () => {
    if (!newLocation.trim()) {
      setError('Location cannot be empty');
      return;
    }

    if (newLocation.trim() === memory.location) {
      setIsEditing(false);
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`/api/memories/${memory.id}/location?session_id=${sessionId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ location: newLocation.trim() })
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      
      // Update the memory data in parent component
      if (onLocationUpdate) {
        onLocationUpdate(memory.id, data.new_location, data.updated_memory);
      }
      
      setIsEditing(false);
    } catch (err) {
      console.error('Error updating location:', err);
      setError(err.message || 'Failed to update location');
    } finally {
      setIsLoading(false);
    }
  };

  const handleCancel = () => {
    setNewLocation(memory.location || '');
    setIsEditing(false);
    setError(null);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSave();
    } else if (e.key === 'Escape') {
      handleCancel();
    }
  };

  const isUnknownLocation = memory.location === 'Unknown Location';

  if (isEditing) {
    return (
      <div className="w-full">
        <div className="flex flex-col gap-1">
          <div className="flex gap-1">
            <input
              ref={inputRef}
              type="text"
              value={newLocation}
              onChange={(e) => setNewLocation(e.target.value)}
              onKeyDown={handleKeyPress}
              className="flex-1 text-xs px-2 py-1 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Enter location..."
              disabled={isLoading}
              maxLength={200}
            />
            <button
              onClick={handleSave}
              disabled={isLoading}
              className="px-2 py-1 bg-green-500 text-white text-xs rounded hover:bg-green-600 disabled:opacity-50"
              title="Save location"
            >
              ✓
            </button>
            <button
              onClick={handleCancel}
              disabled={isLoading}
              className="px-2 py-1 bg-gray-500 text-white text-xs rounded hover:bg-gray-600 disabled:opacity-50"
              title="Cancel editing"
            >
              ✕
            </button>
          </div>
          {error && (
            <div className="text-xs text-red-500">{error}</div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div 
      className={`flex items-center gap-1 group cursor-pointer text-xs text-gray-600 ${
        isUnknownLocation ? 'text-orange-600 hover:text-orange-700' : 'hover:text-blue-600'
      }`}
      onClick={() => setIsEditing(true)}
      title={isUnknownLocation ? 'Click to set location' : 'Click to edit location'}
    >
      <span>📍</span>
      <span className={isUnknownLocation ? 'italic' : ''}>{memory.location}</span>
      {isUnknownLocation ? (
        <span className="opacity-0 group-hover:opacity-100 text-blue-500 text-xs">✏️</span>
      ) : (
        <span className="opacity-0 group-hover:opacity-100 text-blue-500 text-xs">✏️</span>
      )}
    </div>
  );
};

export default LocationEditor;