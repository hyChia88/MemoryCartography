# backend/scripts/pexels_scraper.py
import os
import requests
import time
from pathlib import Path

# You'll need to get a free API key from https://www.pexels.com/api/
API_KEY = "ykXqFs4v1PChOjgGaVsBRhCVAsOA812H36oq6Z7Fhka7QoIf0U0toCwk"  # Register for free at pexels.com

def search_pexels(query, per_page=15, page=1):
    """Search Pexels for images and return JSON response."""
    endpoint = "https://api.pexels.com/v1/search"
    
    headers = {
        "Authorization": API_KEY
    }
    
    params = {
        "query": query,
        "per_page": per_page,
        "page": page
    }
    
    response = requests.get(endpoint, headers=headers, params=params)
    return response.json()

def download_image(url, save_dir, filename):
    """Download an image from URL and save to directory."""
    try:
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            print(f"Failed to download {url}: {response.status_code}")
            return None
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Save the image
        save_path = os.path.join(save_dir, filename)
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        
        return save_path
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None

def scrape_location_images(location, output_dir, num_images=100):
    """Scrape images for a location and save to directory."""
    print(f"Scraping images for {location}...")
    
    images_downloaded = 0
    page = 1
    
    while images_downloaded < num_images:
        # Get JSON response from Pexels
        result = search_pexels(f"{location} Malaysia", per_page=15, page=page)
        
        if "photos" not in result or not result["photos"]:
            print(f"No more photos found for {location}")
            break
        
        photos = result["photos"]
        
        # Download each photo
        for photo in photos:
            # Get medium size image URL
            url = photo["src"]["large"]
            
            # Create a filename based on photo ID
            filename = f"pexels_{photo['id']}.jpg"
            
            # Download the image
            save_path = download_image(url, output_dir, filename)
            if save_path:
                images_downloaded += 1
                print(f"Downloaded {images_downloaded}/{num_images}: {save_path}")
            
            # Break if we've downloaded enough images
            if images_downloaded >= num_images:
                break
            
            # Be nice to the API
            time.sleep(0.5)
        
        # Go to next page
        page += 1
        
        # Respect API rate limits
        if page % 5 == 0:
            print("Pausing to respect API rate limits...")
            time.sleep(2)
    
    print(f"Downloaded {images_downloaded} images for {location}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape images from Pexels")
    parser.add_argument("location", help="Location to search for")
    parser.add_argument("--output", "-o", help="Output directory", default="data/public_photos")
    parser.add_argument("--num", "-n", type=int, help="Number of images to download", default=50)
    
    args = parser.parse_args()
    
    # Create the output directory for this location
    output_dir = os.path.join(args.output, args.location.replace(" ", "_"))
    
    # Scrape images
    scrape_location_images(args.location, output_dir, args.num)