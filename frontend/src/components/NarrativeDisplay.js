import React from 'react';
import highlightUtils from '../utils/highlightUtils';

function NarrativeDisplay({ narrative, searchTerms = [] }) {
  if (!narrative) return null;

  // Extract text and keywords
  const { text, keywords } = narrative;

  // Highlight the narrative
  const highlightedNarrative = highlightUtils.highlightKeywords(
    text, 
    keywords, 
    { 
      highlightColor: 'red', 
      highlightIntensity: 600 
    }
  );

  return (
    <div className="mt-4 p-4 bg-gray-50 rounded-lg">
      <h2 className="text-xl font-semibold mb-4 text-gray-800">
        Synthetic Memory Narrative
      </h2>
      
      {/* Keywords Visualization */}
      {keywords && keywords.length > 0 && (
        <div className="mb-4">
          <h3 className="text-sm font-semibold text-gray-600 mb-2">
            Key Moments:
          </h3>
          <div className="flex flex-wrap gap-2">
            {keywords.map((keyword, index) => (
              <span 
                key={index} 
                className="px-2 py-1 bg-blue-100 text-blue-800 rounded-full text-xs"
              >
                {keyword}
              </span>
            ))}
          </div>
        </div>
      )}
      
      {/* Narrative Text */}
      <div className="text-gray-800 leading-relaxed">
        {highlightedNarrative}
      </div>
    </div>
  );
}

export default NarrativeDisplay;