import React from 'react';
import KeywordHighlight from './KeywordHighlight';

/**
 * Component to display generated narrative with highlighted keywords
 * 
 * @param {Object} props
 * @param {string} props.narrative - Generated narrative text
 * @param {Array} props.keywords - Keywords to highlight in the narrative
 * @param {Array} props.highlightedTerms - Specific terms to highlight
 * @param {Array} props.sourceMemories - IDs of memories used to generate the narrative
 * @param {Function} props.onClose - Handler for closing the narrative
 * @returns {JSX.Element} - Narrative display component
 */
const NarrativeDisplay = ({ 
  narrative, 
  keywords = [], 
  highlightedTerms = [], 
  sourceMemories = [],
  onClose 
}) => {
  if (!narrative) {
    return null;
  }
  
  // Extract keywords from the keyword objects if necessary
  const keywordsToHighlight = keywords.map(k => 
    typeof k === 'object' && k.text ? k.text : k
  );
  
  // Combine with highlighted terms for full highlighting
  const allHighlightTerms = [...keywordsToHighlight, ...highlightedTerms];

  return (
    <div className="bg-gray-50 p-4 rounded mb-4 relative">
      {/* Close button */}
      {onClose && (
        <button
          onClick={onClose}
          className="absolute top-2 right-2 text-gray-400 hover:text-gray-600"
          aria-label="Close narrative"
        >
          &times;
        </button>
      )}
      
      <h2 className="font-bold mb-2 text-gray-800">Generated Narrative</h2>
      
      {/* Narrative text with highlighted keywords */}
      <div className="mb-3">
        <KeywordHighlight 
          text={narrative} 
          keywords={allHighlightTerms} 
          highlightClass="text-red-600 font-bold"
        />
      </div>
      
      {/* Keywords display */}
      {keywords && keywords.length > 0 && (
        <div className="flex flex-wrap gap-2 mb-2">
          {keywords.map((keyword, index) => {
            // Handle both string keywords and keyword objects
            const text = typeof keyword === 'object' ? keyword.text : keyword;
            const type = typeof keyword === 'object' ? keyword.type : 'primary';
            
            return (
              <span 
                key={index} 
                className={`inline-block text-xs px-2 py-1 rounded-full ${
                  type === 'primary' 
                    ? 'bg-yellow-200 text-yellow-800' 
                    : 'bg-gray-100 text-gray-800'
                }`}
              >
                {text}
              </span>
            );
          })}
        </div>
      )}
      
      {/* Source memory count */}
      {sourceMemories && sourceMemories.length > 0 && (
        <p className="text-xs text-gray-500">
          Based on {sourceMemories.length} memories
        </p>
      )}
    </div>
  );
};

export default NarrativeDisplay;