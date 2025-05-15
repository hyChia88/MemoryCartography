import React, { useState, useEffect } from 'react';

const ProcessingComplete = ({ sessionId }) => {
  const [processedPhotos, setProcessedPhotos] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchProcessedPhotos = async () => {
      try {
        const response = await fetch(`/api/upload/processed/${sessionId}`);
        if (response.ok) {
          const data = await response.json();
          setProcessedPhotos(data);
        } else {
          setError('Failed to fetch processed photos');
        }
      } catch (err) {
        setError('Error fetching processed photos');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    if (sessionId) {
      fetchProcessedPhotos();
    }
  }, [sessionId]);

  if (loading) return <div>Loading processed photos...</div>;
  if (error) return <div className="text-red-500">Error: {error}</div>;
  if (!processedPhotos) return null;

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
                src={photo.image_url}
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
                  {photo.impact_weight.toFixed(1)}
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
                {photo.openai_keywords.slice(0, 3).map((keyword, index) => (
                  <span
                    key={index}
                    className="inline-block bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded-full"
                  >
                    {keyword}
                  </span>
                ))}
                {photo.openai_keywords.length > 3 && (
                  <span className="text-xs text-gray-500">
                    +{photo.openai_keywords.length - 3} more
                  </span>
                )}
              </div>

              {/* Detected Objects */}
              {photo.detected_objects.length > 0 && (
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
            // Navigate to memory exploration or whatever comes next
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