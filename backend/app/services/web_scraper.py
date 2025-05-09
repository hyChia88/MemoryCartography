# app/services/web_scraper.py
import os
import requests
import time
import random
import logging
import urllib.parse
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json

# Try to import optional dependencies
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    logging.warning("BeautifulSoup not available. Web scraping capabilities will be limited.")
    
try:
    from PIL import Image
    from io import BytesIO
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("PIL not available. Image processing capabilities will be limited.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('logs/web_scraping.log', mode='a'),
        logging.StreamHandler()
    ]
)

class LocationImageScraper:
    """
    Scrape public images for locations to supplement user photos.
    Uses multiple sources including Pexels API and web search.
    """
    
    def __init__(self, pexels_api_key=None, user_agent=None):
        """
        Initialize the scraper with API keys and settings.
        
        Args:
            pexels_api_key: API key for Pexels (optional)
            user_agent: User agent for HTTP requests
        """
        # Set up Pexels API key (can be provided or loaded from environment)
        self.pexels_api_key = pexels_api_key or os.environ.get("PEXELS_API_KEY")
        
        # Set up HTTP session
        self.user_agent = user_agent or 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.user_agent})
        
        # app/services/web_scraper.py (continued)
        # If Pexels API key is available, add it to headers
        if self.pexels_api_key:
            self.session.headers.update({'Authorization': self.pexels_api_key})
            logging.info("Pexels API key configured for image scraping")
        else:
            logging.warning("No Pexels API key provided. Using fallback methods.")
        
        # Add delay between requests to avoid getting blocked
        self.min_delay = 1.0
        self.max_delay = 3.0
    
    def search_pexels(self, query: str, per_page: int = 15, page: int = 1) -> Dict[str, Any]:
        """
        Search Pexels API for images related to a location.
        
        Args:
            query: Location or keyword to search for
            per_page: Number of results per page
            page: Page number for pagination
            
        Returns:
            Dictionary with search results or empty dict if failed
        """
        if not self.pexels_api_key:
            logging.warning("Pexels API key not available, skipping Pexels search")
            return {}
        
        endpoint = "https://api.pexels.com/v1/search"
        
        params = {
            "query": query,
            "per_page": per_page,
            "page": page
        }
        
        try:
            logging.info(f"Searching Pexels for: {query}")
            response = self.session.get(endpoint, params=params, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                logging.error(f"Pexels API error: {response.status_code}, {response.text}")
                return {}
        except Exception as e:
            logging.error(f"Error searching Pexels: {e}")
            return {}
    
    def search_web_images(self, location: str, max_images: int = 20) -> List[str]:
        """
        Search for images related to a location using web scraping.
        Fallback method when Pexels API is not available.
        
        Args:
            location: Location name to search for
            max_images: Maximum number of image URLs to return
            
        Returns:
            List of image URLs
        """
        if not BS4_AVAILABLE:
            logging.warning("BeautifulSoup not available, skipping web search")
            return []
        
        logging.info(f"Searching web for images of: {location}")
        
        # Create search query
        query = f"{location} landmark site:wikimedia.org OR site:commons.wikimedia.org OR site:pexels.com"
        encoded_query = urllib.parse.quote(query)
        
        # Construct search URL
        search_url = f"https://www.google.com/search?q={encoded_query}&tbm=isch"
        
        # Get search results page
        try:
            # Add random delay to avoid being blocked
            time.sleep(random.uniform(self.min_delay, self.max_delay))
            
            response = self.session.get(search_url, timeout=10)
            
            if response.status_code != 200:
                logging.error(f"Error searching for images: {response.status_code}")
                return []
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract image URLs
            image_urls = []
            
            # Google stores image data in different ways, try to find it
            # Look for img tags with src attribute
            images = soup.find_all('img')
            for img in images:
                if 'src' in img.attrs and img['src'].startswith('http'):
                    image_urls.append(img['src'])
            
            # Limit the number of images
            return image_urls[:max_images]
            
        except Exception as e:
            logging.error(f"Error searching for images of {location}: {e}")
            return []
    
    def download_image(self, url: str, output_path: str) -> bool:
        """
        Download an image from a URL to a specified path.
        
        Args:
            url: URL of the image to download
            output_path: Path where the image should be saved
            
        Returns:
            True if download was successful, False otherwise
        """
        try:
            logging.info(f"Downloading image from: {url}")
            
            # Add random delay to avoid rate limiting
            time.sleep(random.uniform(self.min_delay, self.max_delay))
            
            # Download the image
            response = self.session.get(url, timeout=10)
            
            if response.status_code != 200:
                logging.error(f"Failed to download {url}: {response.status_code}")
                return False
            
            # Verify it's an actual image
            if PIL_AVAILABLE:
                try:
                    image = Image.open(BytesIO(response.content))
                    
                    # Save the image
                    image.save(output_path)
                    logging.info(f"Image saved to: {output_path}")
                    return True
                except Exception as e:
                    logging.error(f"Not a valid image: {url}, Error: {e}")
                    return False
            else:
                # If PIL is not available, just save the raw content
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                logging.info(f"Image saved to: {output_path} (without validation)")
                return True
                
        except Exception as e:
            logging.error(f"Error downloading image from {url}: {e}")
            return False
    
    def scrape_location_images_pexels(self, location: str, output_dir: str, max_images: int = 20) -> List[str]:
        """
        Scrape images for a location from Pexels API.
        
        Args:
            location: Location name to search for
            output_dir: Directory to save downloaded images
            max_images: Maximum number of images to download
            
        Returns:
            List of paths to downloaded images
        """
        if not self.pexels_api_key:
            logging.warning("Pexels API key not available, skipping Pexels search")
            return []
        
        os.makedirs(output_dir, exist_ok=True)
        
        downloaded_paths = []
        images_downloaded = 0
        page = 1
        
        while images_downloaded < max_images:
            # Get JSON response from Pexels
            search_query = f"{location}"
            result = self.search_pexels(search_query, per_page=15, page=page)
            
            if "photos" not in result or not result["photos"]:
                logging.info(f"No more photos found for {location} on page {page}")
                break
            
            photos = result["photos"]
            
            # Download each photo
            for photo in photos:
                if images_downloaded >= max_images:
                    break
                
                # Get medium size image URL
                url = photo["src"]["large"]
                
                # Create a filename based on photo ID
                location_slug = location.replace(" ", "_").replace(",", "").lower()
                filename = f"{location_slug}_pexels_{photo['id']}.jpg"
                output_path = os.path.join(output_dir, filename)
                
                # Download the image
                if self.download_image(url, output_path):
                    downloaded_paths.append(output_path)
                    images_downloaded += 1
                    logging.info(f"Downloaded {images_downloaded}/{max_images}: {output_path}")
                
                # Be nice to the API
                time.sleep(0.5)
            
            # Go to next page
            page += 1
            
            # Respect API rate limits
            if page % 5 == 0:
                logging.info("Pausing to respect API rate limits...")
                time.sleep(2)
        
        logging.info(f"Downloaded {len(downloaded_paths)} Pexels images for {location}")
        return downloaded_paths
    
    def scrape_location_images_web(self, location: str, output_dir: str, max_images: int = 20) -> List[str]:
        """
        Scrape images for a location using web search as fallback.
        
        Args:
            location: Location name to search for
            output_dir: Directory to save downloaded images
            max_images: Maximum number of images to download
            
        Returns:
            List of paths to downloaded images
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Search for image URLs
        image_urls = self.search_web_images(location, max_images=max_images*2)  # Get extra URLs in case some fail
        
        if not image_urls:
            logging.warning(f"No images found for location: {location}")
            return []
        
        # Download images (up to the maximum requested)
        downloaded_paths = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for i, url in enumerate(image_urls):
                if i >= max_images:
                    break
                    
                # Create output filename
                image_extension = url.split('.')[-1].lower()
                if image_extension not in ['jpg', 'jpeg', 'png', 'webp', 'gif']:
                    image_extension = 'jpg'  # Default extension
                
                location_slug = location.replace(" ", "_").replace(",", "").lower()
                output_filename = f"{location_slug}_web_{i+1}.{image_extension}"
                output_path = os.path.join(output_dir, output_filename)
                
                # Submit download task
                future = executor.submit(self.download_image, url, output_path)
                futures.append((future, output_path))
            
            # Collect results
            for future, path in futures:
                if future.result():
                    downloaded_paths.append(path)
        
        logging.info(f"Downloaded {len(downloaded_paths)} web images for {location}")
        return downloaded_paths
    
    def scrape_location_images(self, location: str, output_dir: str, max_images: int = 20) -> List[str]:
        """
        Scrape images for a location using all available methods.
        
        Args:
            location: Location name to search for
            output_dir: Directory to save downloaded images
            max_images: Maximum number of images to download
            
        Returns:
            List of paths to downloaded images
        """
        os.makedirs(output_dir, exist_ok=True)
        downloaded_paths = []
        
        # First try Pexels API if available
        if self.pexels_api_key:
            pexels_paths = self.scrape_location_images_pexels(
                location, 
                output_dir, 
                max_images=max_images//2  # Use half the quota for Pexels
            )
            downloaded_paths.extend(pexels_paths)
        
        # If we need more images, use web search
        remaining_images = max_images - len(downloaded_paths)
        if remaining_images > 0:
            web_paths = self.scrape_location_images_web(
                location, 
                output_dir, 
                max_images=remaining_images
            )
            downloaded_paths.extend(web_paths)
        
        return downloaded_paths
    
    def scrape_multiple_locations(self, locations: List[str], base_output_dir: str, images_per_location: int = 10, total_images: int = 100) -> List[str]:
        """
        Scrape images for multiple locations with a total image limit.
        
        Args:
            locations: List of location names to search for
            base_output_dir: Base directory for saving images
            images_per_location: Maximum images per location
            total_images: Overall maximum images to download
            
        Returns:
            List of paths to all downloaded images
        """
        os.makedirs(base_output_dir, exist_ok=True)
        
        if not locations:
            logging.warning("No locations provided")
            return []
        
        # Calculate how many images to download per location
        num_locations = len(locations)
        if num_locations > 0:
            images_per_location = min(images_per_location, total_images // num_locations)
        
        all_downloaded_paths = []
        
        for location in locations:
            # Stop if we've reached the total limit
            if len(all_downloaded_paths) >= total_images:
                break
                
            location_dir = os.path.join(base_output_dir, location.replace(" ", "_").replace(",", "").lower())
            
            # Calculate how many more images we can download
            remaining = total_images - len(all_downloaded_paths)
            if remaining <= 0:
                break
                
            # Download images for this location
            location_max = min(images_per_location, remaining)
            downloaded_paths = self.scrape_location_images(
                location, 
                location_dir, 
                max_images=location_max
            )
            
            all_downloaded_paths.extend(downloaded_paths)
        
        return all_downloaded_paths