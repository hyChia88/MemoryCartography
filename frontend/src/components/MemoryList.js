import React from 'react';
import LocationEditor from './LocationEditor';

/**
 * Component that displays a grid of memory cards
 * 
 * @param {Object} props
 * @param {Array} props.memories - Array of memory objects to display
 * @param {Function} props.onIncreaseWeight - Handler for increasing memory weight
 * @param {Function} props.onDecreaseWeight - Handler for decreasing memory weight
 * @param {Function} props.getThumbnailUrl - Function to get the thumbnail URL for a memory
 * @param {Function} props.onLocationUpdate - Handler for location updates
 * @param {string} props.sessionId - Current session ID
 * @returns {JSX.Element} - Grid of memory cards
 */
const MemoryList = ({ 
  memories, 
  onIncreaseWeight, 
  onDecreaseWeight, 
  getThumbnailUrl, 
  onLocationUpdate,
  sessionId 
}) => {
  // Format date for display
  const formatDate = (dateString) => {
    try {
      const date = new Date(dateString);
      return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
      });
    } catch (e) {
      return dateString;
    }
  };

  if (memories.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        No memories found
      </div>
    );
  }

  return (
    <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
      {memories.map((memory) => (
        <div
          key={memory.id}
          className="bg-white border rounded overflow-hidden hover:shadow-lg transition-shadow relative"
          // Pass the event to handlers
          onContextMenu={(e) => onDecreaseWeight(memory.id, e)}
          onClick={(e) => {
            // Prevent weight adjustment if clicking on location editor
            if (e.target.closest('.location-editor')) {
              e.stopPropagation();
              return;
            }
            onIncreaseWeight(memory.id, e);
          }}
        >
          {/* Image Thumbnail */}
          <div className="w-full h-40 bg-gray-200 relative overflow-hidden">
            <img
              src={getThumbnailUrl(memory)}
              alt={memory.title}
              className="w-full h-full object-cover"
              onError={(e) => {
                // Fallback for failed image loads
                e.target.src = '/placeholder-image.jpg';
              }}
            />
            {/* Weight indicator positioned on top of image */}
            <div 
              className="absolute top-2 right-2 h-6 w-6 rounded-full flex items-center justify-center"
              style={{ 
                backgroundColor: `rgba(255, 204, 0, ${(memory.weight || memory.impact_weight || 1.0) / 2})`,
                border: '1px solid #ffffff',
                boxShadow: '0 1px 3px rgba(0,0,0,0.2)'
              }}
              title={`Memory weight: ${(memory.weight || memory.impact_weight || 1.0).toFixed(1)}`}
            >
              <span className="text-xs font-semibold text-gray-800">
                {(memory.weight || memory.impact_weight || 1.0).toFixed(1)}
              </span>
            </div>
          </div>
          
          {/* Memory Details */}
          <div className="p-3">
            <h2 className="font-bold text-gray-800 truncate">{memory.title}</h2>
            
            {/* Location with editor - wrap in div with class for event handling */}
            <div className="location-editor">
              <LocationEditor
                memory={memory}
                onLocationUpdate={onLocationUpdate}
                sessionId={sessionId}
              />
            </div>
            
            <p className="text-sm text-gray-600">{formatDate(memory.date)}</p>
            <div className="mt-2">
              {(memory.keywords || memory.openai_keywords || []).slice(0, 3).map((keyword, index) => (
                <span
                  key={index}
                  className="inline-block bg-gray-100 text-gray-800 text-xs px-2 py-1 rounded mr-1 mb-1"
                >
                  {keyword}
                </span>
              ))}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};

export default MemoryList;