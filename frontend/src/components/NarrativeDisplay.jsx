import React from 'react';
import { useMemory } from '../context/MemoryContext';
import KeywordTag from './KeywordTag';

// Function to highlight terms in the narrative
const highlightNarrativeTerms = (text, highlightedTerms = []) => {
  if (!text || !highlightedTerms.length) return text;
  
  // Create a regex pattern that matches any of the highlighted terms
  const pattern = new RegExp(
    `(${highlightedTerms.map(term => term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})`, 
    'gi'
  );
  
  // Split the text by the pattern
  const parts = text.split(pattern);
  
  return (
    <>
      {parts.map((part, index) => {
        // Check if this part matches any highlighted term (case insensitive)
        const isHighlighted = highlightedTerms.some(
          term => part.toLowerCase() === term.toLowerCase()
        );
        
        return isHighlighted ? (
          <span key={index} className="font-bold text-red-600">
            {part}
          </span>
        ) : (
          <span key={index}>{part}</span>
        );
      })}
    </>
  );
};

// Function to highlight terms wrapped in asterisks
const highlightAsterisks = (text) => {
  if (!text) return text;
  
  // Pattern to match text between asterisks
  const parts = text.split(/(\*[^*]+\*)/g);
  
  return (
    <>
      {parts.map((part, index) => {
        if (part.startsWith('*') && part.endsWith('*')) {
          // Remove asterisks and return highlighted text
          const content = part.substring(1, part.length - 1);
          return (
            <span key={index} className="font-bold text-red-600">
              {content}
            </span>
          );
        }
        return <span key={index}>{part}</span>;
      })}
    </>
  );
};

const NarrativeDisplay = () => {
  const { narrative, searchQuery } = useMemory();
  
  // If no narrative, show placeholder
  if (!narrative) {
    return (
      <div className="flex-1 bg-gray-50 rounded-lg p-4 overflow-y-auto">
        <h2 className="text-sm font-semibold text-gray-600 mb-3 uppercase tracking-wider">
          Generated Narrative
        </h2>
        <p className="text-gray-500 italic text-sm">
          Search a location or keyword to generate a memory narrative...
        </p>
      </div>
    );
  }
  
  // Extract data from narrative
  const { text, keywords, highlighted_terms = [], primary_memories = [] } = narrative;
  
  return (
    <div className="flex-1 bg-gray-50 rounded-lg p-4 overflow-y-auto">
      <h2 className="text-sm font-semibold text-gray-600 mb-3 uppercase tracking-wider">
        Generated Narrative
        {searchQuery && (
          <span className="ml-2 text-xs normal-case font-normal">
            based on "{searchQuery}"
          </span>
        )}
      </h2>
      
      {/* Display keywords as tags */}
      <div className="mb-4 flex flex-wrap gap-2">
        {keywords && keywords.map((keyword, index) => (
          <KeywordTag key={index} keyword={keyword} />
        ))}
      </div>
      
      {/* Display narrative with highlighted terms */}
      <div className="text-gray-700 leading-relaxed">
        {text && highlighted_terms.length > 0 
          ? highlightNarrativeTerms(text, highlighted_terms)
          : highlightAsterisks(text)
        }
      </div>
      
      {/* Display source information */}
      {primary_memories && primary_memories.length > 0 && (
        <div className="mt-4 text-xs text-gray-500">
          <p>This narrative is based on {primary_memories.length} memories.</p>
        </div>
      )}
    </div>
  );
};

export default NarrativeDisplay;