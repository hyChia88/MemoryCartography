import axios from 'axios';

class OpenAIService {
  constructor() {
    this.apiKey = process.env.REACT_APP_OPENAI_API_KEY;
    this.apiUrl = 'https://api.openai.com/v1/chat/completions';
  }

  // Generate keywords for an image
  async generateImageKeywords(imageDetails) {
    try {
      const prompt = `Generate a comprehensive list of keywords for an image with the following details:
Location: ${imageDetails.location}
Date: ${imageDetails.date}

Provide keywords in the following categories:
1. Location-specific details
2. Sensory experiences
3. Emotional associations
4. Abstract concepts
5. Concrete objects

Return as a JSON array with at least 15 unique keywords.`;

      const response = await axios.post(
        this.apiUrl,
        {
          model: "gpt-3.5-turbo",
          messages: [{ role: "user", content: prompt }],
          max_tokens: 150
        },
        {
          headers: {
            'Authorization': `Bearer ${this.apiKey}`,
            'Content-Type': 'application/json'
          }
        }
      );

      // Parse the keywords from the response
      const keywordsResponse = response.data.choices[0].message.content.trim();
      return JSON.parse(keywordsResponse);
    } catch (error) {
      console.error('Error generating keywords:', error);
      return [];
    }
  }

  // Generate synthetic memory narrative
  async generateSyntheticMemory(memories) {
    try {
      // Sort memories by weight in descending order
      const sortedMemories = memories.sort((a, b) => b.weight - a.weight);
      
      // Take top memories that sum to at least 3.0
      const selectedMemories = [];
      let totalWeight = 0;
      for (let memory of sortedMemories) {
        if (totalWeight < 3.0) {
          selectedMemories.push(memory);
          totalWeight += memory.weight;
        } else {
          break;
        }
      }

      const prompt = `Create a synthetic memory narrative based on these memory events:

${selectedMemories.map((m, index) => 
  `Memory ${index + 1}:
- Location: ${m.location}
- Date: ${m.date}
- Keywords: ${m.keywords.join(', ')}
- Weight: ${m.weight}`
).join('\n\n')}

Write a narrative that:
1. Follows the chronological order of these memories
2. Captures the emotional essence of the experiences
3. Uses vivid, personal language
4. Highlights the most significant moments
5. Creates a cohesive story that feels like a personal diary entry

Include some of the keywords naturally in the narrative.`;

      const response = await axios.post(
        this.apiUrl,
        {
          model: "gpt-3.5-turbo",
          messages: [{ role: "user", content: prompt }],
          max_tokens: 250
        },
        {
          headers: {
            'Authorization': `Bearer ${this.apiKey}`,
            'Content-Type': 'application/json'
          }
        }
      );

      // Return the narrative and the keywords to highlight
      const narrative = response.data.choices[0].message.content.trim();
      const allKeywords = selectedMemories.flatMap(m => m.keywords);

      return {
        text: narrative,
        keywords: allKeywords
      };
    } catch (error) {
      console.error('Error generating synthetic memory:', error);
      return null;
    }
  }
}

export default new OpenAIService();