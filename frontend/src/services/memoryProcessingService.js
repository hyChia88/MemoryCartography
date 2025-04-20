import OpenAIService from './openaiService';

class MemoryProcessingService {
  // Process an uploaded image to create a memory event
  async processImage(imageFile) {
    try {
      // Extract location details (you might want to use a more sophisticated method)
      const imageDetails = await this.extractImageDetails(imageFile);

      // Generate keywords using OpenAI
      const generatedKeywords = await OpenAIService.generateImageKeywords(imageDetails);

      // Create memory event object
      const memoryEvent = {
        original_path: imageFile.path,
        location: imageDetails.location,
        date: imageDetails.date || new Date().toISOString().split('T')[0],
        keywords: generatedKeywords,
        weight: 1.0  // Initial weight
      };

      return memoryEvent;
    } catch (error) {
      console.error('Error processing image:', error);
      return null;
    }
  }

  // Extract image details (this is a placeholder - you'd replace with actual image analysis)
  async extractImageDetails(imageFile) {
    // In a real-world scenario, you might use:
    // - EXIF data for location and date
    // - Computer vision API to detect location
    // - GPS metadata
    return {
      location: this.guessLocationFromFilename(imageFile.name),
      date: this.extractDateFromFilename(imageFile.name)
    };
  }

  // Guess location from filename (very basic implementation)
  guessLocationFromFilename(filename) {
    // Look for location-like patterns in the filename
    const locationPatterns = [
      'kuala_lumpur', 'malaysia', 'kl', 'city', 'trip', 
      'vacation', 'travel', 'journey'
    ];

    const matchedLocation = locationPatterns.find(pattern => 
      filename.toLowerCase().includes(pattern)
    );

    return matchedLocation 
      ? this.formatLocation(matchedLocation)
      : 'Unknown Location';
  }

  // Format location to be more descriptive
  formatLocation(location) {
    const locationMappings = {
      'kuala_lumpur': 'Kuala Lumpur City Center, Malaysia',
      'kl': 'Kuala Lumpur Metropolitan Area, Malaysia',
      'malaysia': 'Various Locations, Malaysia'
    };

    return locationMappings[location] || location;
  }

  // Extract date from filename (very basic implementation)
  extractDateFromFilename(filename) {
    // Look for date patterns like YYYYMMDD or YYYY-MM-DD
    const datePatterns = [
      /(\d{4})(\d{2})(\d{2})/,  // YYYYMMDD
      /(\d{4})-(\d{2})-(\d{2})/  // YYYY-MM-DD
    ];

    for (let pattern of datePatterns) {
      const match = filename.match(pattern);
      if (match) {
        return `${match[1]}-${match[2]}-${match[3]}`;
      }
    }

    // If no date found, return today's date
    return new Date().toISOString().split('T')[0];
  }

  // Merge multiple memory events
  mergeMemoryEvents(events) {
    // Sort events by date
    const sortedEvents = events.sort((a, b) => 
      new Date(a.date) - new Date(b.date)
    );

    // Aggregate keywords
    const allKeywords = sortedEvents.flatMap(event => event.keywords);
    const uniqueKeywords = [...new Set(allKeywords)];

    // Calculate average weight
    const averageWeight = sortedEvents.reduce((sum, event) => sum + event.weight, 0) / sortedEvents.length;

    // Create merged event
    return {
      location: sortedEvents.map(e => e.location).join(', '),
      date: sortedEvents[0].date,  // Use the earliest date
      keywords: uniqueKeywords,
      weight: averageWeight,
      original_paths: sortedEvents.map(e => e.original_path)
    };
  }

  // Save memory events to local storage or backend
  saveMemoryEvents(events) {
    try {
      // In a real app, this would interact with your backend API
      const existingEvents = JSON.parse(localStorage.getItem('memoryEvents') || '{}');
      
      events.forEach(event => {
        const key = `memory_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        existingEvents[key] = event;
      });

      localStorage.setItem('memoryEvents', JSON.stringify(existingEvents));
      return existingEvents;
    } catch (error) {
      console.error('Error saving memory events:', error);
      return null;
    }
  }

  // Retrieve memory events from storage or backend
  getMemoryEvents() {
    try {
      const storedEvents = localStorage.getItem('memoryEvents');
      return storedEvents ? JSON.parse(storedEvents) : {};
    } catch (error) {
      console.error('Error retrieving memory events:', error);
      return {};
    }
  }
}

export default new MemoryProcessingService();