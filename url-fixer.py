import os
import re
import argparse
import urllib.request
from collections import defaultdict
from tqdm import tqdm

def get_image_suffixes(base_url, cache_file=None):
    """
    Get a dictionary mapping from image numbers to their suffixes by parsing directory listings.
    
    Args:
        base_url: The base URL for the imagesets, e.g., 'https://prism-static.aruw.org/images/1_145/'
        cache_file: Optional file to cache results to avoid repeated downloads
        
    Returns:
        Dictionary mapping from image numbers to their suffixes, e.g., {'0000601': 'EGD3NF'}
    """
    if cache_file and os.path.exists(cache_file):
        print(f"Loading image suffixes from cache file: {cache_file}")
        suffixes = {}
        with open(cache_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 3:  # imageset_id, image_number, suffix
                    imageset_id, image_number, suffix = parts
                    if imageset_id not in suffixes:
                        suffixes[imageset_id] = {}
                    suffixes[imageset_id][image_number] = suffix
        return suffixes
    
    # Initialize an empty dictionary to store suffixes for each imageset
    suffixes = defaultdict(dict)
    
    # If base_url ends with a specific imageset (e.g., 1_145/), process just that one
    base_match = re.search(r'images/1_(\d+)/?$', base_url)
    if base_match:
        imageset_ids = [base_match.group(1)]
        # Ensure the base URL ends with a slash
        base_url = base_url[:-1] if base_url.endswith('/') else base_url
        base_url = base_url.rsplit('/', 1)[0]  # Remove the imageset part
    else:
        # Get all imagesets from the pattern "1_NNN"
        # This assumes there's a way to list all available imagesets
        print("Base URL doesn't specify an imageset. Using predefined imagesets.")
        imageset_ids = [
            '145', '146', '147', '148', '149', '152', 
            '153', '154', '158', '175'
        ]
    
    print(f"Processing {len(imageset_ids)} imagesets")
    
    for imageset_id in tqdm(imageset_ids, desc="Fetching imagesets"):
        # Get the directory listing for this imageset
        imageset_url = f"{base_url}/1_{imageset_id}/"
        try:
            with urllib.request.urlopen(imageset_url) as response:
                directory_listing = response.read().decode('utf-8')
                
                # Extract all image filenames and their suffixes
                pattern = r'image-(\d+)_([A-Za-z0-9]+)\.jpg'
                matches = re.findall(pattern, directory_listing)
                
                # Store the mapping from image number to suffix
                for image_number, suffix in matches:
                    suffixes[imageset_id][image_number] = suffix
                
                print(f"Found {len(matches)} images for imageset {imageset_id}")
        except Exception as e:
            print(f"Error processing imageset {imageset_id}: {e}")
    
    # Save to cache file if specified
    if cache_file:
        print(f"Saving image suffixes to cache file: {cache_file}")
        with open(cache_file, 'w') as f:
            for imageset_id in suffixes:
                for image_number, suffix in suffixes[imageset_id].items():
                    f.write(f"{imageset_id},{image_number},{suffix}\n")
    
    return suffixes

def update_urls_in_file(input_file, output_file, image_suffixes):
    """
    Update URLs in a file to include the proper image suffixes.
    
    Args:
        input_file: Path to the input file (e.g., train.txt)
        output_file: Path to the output file
        image_suffixes: Dictionary mapping from imagesets to image numbers to suffixes
    """
    print(f"Updating URLs in {input_file}")
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in tqdm(f_in, desc="Processing URLs"):
            line = line.strip()
            
            # Extract imageset and image number
            match = re.search(r'images/1_(\d+)/image-(\d+)\.jpg', line)
            if match:
                imageset_id, image_number = match.groups()
                
                # Check if we have a suffix for this image
                if imageset_id in image_suffixes and image_number in image_suffixes[imageset_id]:
                    suffix = image_suffixes[imageset_id][image_number]
                    
                    # Replace the URL with the suffixed version
                    updated_line = re.sub(
                        r'(images/1_\d+/image-\d+)\.jpg',
                        r'\1_' + suffix + '.jpg',
                        line
                    )
                    f_out.write(updated_line + '\n')
                else:
                    # Keep the original URL if no suffix is found
                    print(f"Warning: No suffix found for imageset {imageset_id}, image {image_number}")
                    f_out.write(line + '\n')
            else:
                # Keep lines that don't match the expected URL pattern
                f_out.write(line + '\n')
    
    print(f"Updated URLs saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Update image URLs to include proper suffixes')
    parser.add_argument('--url-file', required=True, help='Path to file containing URLs (train.txt or val.txt)')
    parser.add_argument('--output-file', help='Path to save updated URLs (defaults to original filename with .new suffix)')
    parser.add_argument('--base-url', default='https://prism-static.aruw.org/images/', help='Base URL for the images')
    parser.add_argument('--cache-file', default='image_suffixes.cache', help='File to cache image suffixes')
    
    args = parser.parse_args()
    
    # Set default output file if not specified
    if not args.output_file:
        args.output_file = args.url_file + '.new'
    
    # Get the image suffixes
    image_suffixes = get_image_suffixes(args.base_url, args.cache_file)
    
    # Update URLs in the file
    update_urls_in_file(args.url_file, args.output_file, image_suffixes)
    
    print("Done!")

if __name__ == "__main__":
    main()