import os
import requests
from concurrent.futures import ThreadPoolExecutor

dataset_dir = 'datasets/url_dataset'

def download_images(split):
    image_dir = os.path.join(dataset_dir, 'images', split)
    os.makedirs(image_dir, exist_ok=True)

    url_file = os.path.join(dataset_dir, f'{split}.txt')

    with open(url_file, 'r') as f:
        urls = [line.strip() for line in f.readlines()]

    def download_image(url):
        filename = os.path.basename(url.split('?')[0])
        filepath = os.path.join(image_dir, filename)

        if os.path.exists(filepath):
            print(f"[{split}] Already downloaded: {filename}")
            return

        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print(f"[{split}] Downloaded: {filename}")
            else:
                print(f"[{split}] Failed {url} - Status: {response.status_code}")
        except Exception as e:
            print(f"[{split}] Error {url} - {e}")

    with ThreadPoolExecutor(max_workers=20) as executor:
        executor.map(download_image, urls)

download_images('train')
download_images('val')
