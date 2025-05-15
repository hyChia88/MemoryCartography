// Update ProcessingComplete.jsx with better error handling

import React, { useState, useEffect } from 'react';

const ProcessingComplete = ({ sessionId }) => {
  const [processedPhotos, setProcessedPhotos] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchProcessedPhotos = async () => {
      try {
        console.log(`Fetching processed photos for session: ${sessionId}`);
        
        // Check different possible endpoint configurations
        const endpoints = [
          `/api/upload/processed/${sessionId}`,
          `${process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000'}/api/upload/processed/${sessionId}`,
        ];
        
        let response;
        let lastError;
        
        for (const endpoint of endpoints) {
          try {
            console.log(`Trying endpoint: ${endpoint}`);
            response = await fetch(endpoint, {
              headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json',
              },
            });
            
            if (response.ok) {
              break; // Success, exit loop
            } else {
              lastError = `HTTP ${response.status}: ${response.statusText}`;
              continue; // Try next endpoint
            }
          } catch (err) {
            lastError = err.message;
            continue; // Try next endpoint
          }
        }
        
        if (!response || !response.ok) {
          throw new Error(lastError || 'All endpoints failed');
        }
        
        // Check if response is JSON
        const contentType = response.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
          const text = await response.text();
          console.error('Non-JSON response received:', text.substring(0, 200));
          throw new Error('Server returned HTML instead of JSON. This usually means the API endpoint doesn\'t exist or the session is invalid.');
        }
        
        const data = await response.json();
        setProcessedPhotos(data);
        console.log('Processed photos fetched successfully:', data);
      } catch (err) {
        console.error('Error fetching processed photos:', err);
        setError(`Failed to load processed photos: ${err.message}`);
      } finally {
        setLoading(false);
      }
    };

    if (sessionId) {
      fetchProcessedPhotos();
    }
  }, [sessionId]);

  if (loading) {
    return (
      <div className="mt-6 bg-blue-50 border border-blue-200 rounded-lg p-4">
        <div className="flex items-center">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500"></div>
          <span className="ml-2">Loading processed photos...</span>
        </div>
      </div>
    );
  }
  
  if (error) {
    return (
      <div className="mt-6 bg-red-50 border border-red-200 rounded-lg p-4">
        <div className="text-red-700">
          <strong>Error loading processed photos:</strong>
          <p className="mt-1">{error}</p>
          <p className="mt-2 text-sm">
            Try refreshing the page or check that the session is still active.
          </p>
        </div>
      </div>
    );
  }
  
  if (!processedPhotos || !processedPhotos.photos || processedPhotos.photos.length === 0) {
    return (
      <div className="mt-6 bg-yellow-50 border border-yellow-200 rounded-lg p-4">
        <div className="text-yellow-700">
          <strong>No processed photos found</strong>
          <p className="mt-1">The processing may still be in progress, or no photos were processed successfully.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="mt-6 bg-white rounded-lg border p-4">
      <h2 className="text-xl font-bold mb-4 text-gray-800">Processing Complete! 🎉</h2>
      
      {/* Summary Stats */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="bg-blue-50 p-3 rounded text-center">
          <div className="text-2xl font-bold text-blue-600">{processedPhotos.total_count}</div>
          <div className="text-sm text-gray-600">Total Photos</div>
        </div>
        <div className="bg-green-50 p-3 rounded text-center">
          <div className="text-2xl font-bold text-green-600">{processedPhotos.user_count}</div>
          <div className="text-sm text-gray-600">Your Photos</div>
        </div>
        <div className="bg-purple-50 p-3 rounded text-center">
          <div className="text-2xl font-bold text-purple-600">{processedPhotos.public_count}</div>
          <div className="text-sm text-gray-600">Public Photos</div>
        </div>
      </div>

      {/* Photo Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {processedPhotos.photos.map((photo) => (
          <div key={photo.id} className="border rounded-lg overflow-hidden shadow-sm hover:shadow-md transition-shadow">
            <div className="aspect-w-16 aspect-h-9 bg-gray-200">
              <img
                src={photo.image_url ? `${process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000'}${photo.image_url}` : '/placeholder-image.jpg'}
                alt={photo.title}
                className="w-full h-48 object-cover"
                onError={(e) => {
                  e.target.src = '/placeholder-image.jpg';
                }}
              />
            </div>
            
            <div className="p-3">
              <h3 className="font-semibold text-gray-800 text-sm mb-1">{photo.title}</h3>
              
              {/* Location and Date */}
              <div className="text-xs text-gray-600 mb-2">
                <div>📍 {photo.location}</div>
                <div>📅 {photo.date}</div>
              </div>

              {/* Impact Weight */}
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-gray-500">Impact Score:</span>
                <span className="text-sm font-semibold text-blue-600">
                  {photo.impact_weight ? photo.impact_weight.toFixed(1) : '1.0'}
                </span>
              </div>

              {/* Description */}
              {photo.openai_description && (
                <p className="text-xs text-gray-700 mb-2 italic">
                  "{photo.openai_description}"
                </p>
              )}

              {/* Keywords */}
              <div className="flex flex-wrap gap-1">
                {(photo.openai_keywords || []).slice(0, 3).map((keyword, index) => (
                  <span
                    key={index}
                    className="inline-block bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded-full"
                  >
                    {keyword}
                  </span>
                ))}
                {(photo.openai_keywords || []).length > 3 && (
                  <span className="text-xs text-gray-500">
                    +{photo.openai_keywords.length - 3} more
                  </span>
                )}
              </div>

              {/* Detected Objects */}
              {photo.detected_objects && photo.detected_objects.length > 0 && (
                <div className="mt-2">
                  <div className="text-xs text-gray-500 mb-1">Detected:</div>
                  <div className="flex flex-wrap gap-1">
                    {photo.detected_objects.slice(0, 2).map((obj, index) => (
                      <span
                        key={index}
                        className="inline-block bg-green-100 text-green-800 text-xs px-2 py-1 rounded-full"
                      >
                        {obj}
                      </span>
                    ))}
                    {photo.detected_objects.length > 2 && (
                      <span className="text-xs text-gray-500">
                        +{photo.detected_objects.length - 2} more
                      </span>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Continue Button */}
      <div className="mt-6 text-center">
        <button
          onClick={() => {
            // Navigate to memory exploration
            window.location.href = `/memories?session=${sessionId}`;
          }}
          className="bg-blue-500 text-white px-6 py-2 rounded hover:bg-blue-600 transition-colors"
        >
          Explore Your Memories
        </button>
      </div>
    </div>
  );
};

export default ProcessingComplete;