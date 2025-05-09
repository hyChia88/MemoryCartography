import React from 'react';
import DOMPurify from 'dompurify';

/**
 * Component that highlights keywords in a text
 * 
 * @param {Object} props
 * @param {string} props.text - The text to highlight
 * @param {Array} props.keywords - List of keywords to highlight
 * @param {string} props.highlightClass - CSS class to apply to highlighted words (default: "text-red-600 font-bold")
 * @returns {JSX.Element} - Text with highlighted keywords
 */
const KeywordHighlight = ({ text, keywords, highlightClass = "text-red-600 font-bold" }) => {
  if (!text || !keywords || keywords.length === 0) {
    return <p>{text}</p>;
  }

  const highlightText = () => {
    let highlightedText = text;

    // Sort keywords by length (longest first) to avoid highlighting issues
    const sortedKeywords = [...keywords].sort((a, b) => b.length - a.length);

    sortedKeywords.forEach((keyword) => {
      if (!keyword) return;
      
      // Create regex to match the keyword with word boundaries
      const regex = new RegExp(`\\b${keyword}\\b`, 'gi');
      
      // Replace the keyword with a highlighted version
      highlightedText = highlightedText.replace(
        regex,
        `<span class="${highlightClass}">${keyword}</span>`
      );
    });

    // Sanitize the HTML to prevent XSS attacks
    return DOMPurify.sanitize(highlightedText);
  };

  return (
    <p dangerouslySetInnerHTML={{ __html: highlightText() }} />
  );
};

export default KeywordHighlight;