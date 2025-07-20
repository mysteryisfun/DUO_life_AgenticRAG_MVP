import requests
from bs4 import BeautifulSoup
import json

TAVILY_API_KEY = "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"  # <-- User's actual API key
INPUT_FILE = "urls.txt"
OUTPUT_FILE = "productImages.js"

def fetch_html_with_tavily(url):
    api_url = "https://api.tavily.com/v1/search"
    headers = {"Authorization": f"Bearer {TAVILY_API_KEY}"}
    data = {
        "query": url,  # or some text query
        "include_html": True
    }
    response = requests.post(api_url, headers=headers, json=data)
    if response.status_code != 200:
        print("Tavily API error:", response.text)
    response.raise_for_status()
    return response.json().get("html", "")

def extract_image_url(html):
    soup = BeautifulSoup(html, "html.parser")
    # Try the main selector first
    img = soup.select_one("img.details__image__product")
    if img and img.get("src"):
        return img["src"]
    # Fallbacks
    og_img = soup.find("meta", property="og:image")
    if og_img and og_img.get("content"):
        return og_img["content"]
    return None

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    mapping = {}
    for url in urls:
        print(f"Fetching: {url}")
        try:
            html = fetch_html_with_tavily(url)
            img_url = extract_image_url(html)
            if img_url:
                # Make relative URLs absolute
                if img_url.startswith("/"):
                    from urllib.parse import urljoin
                    img_url = urljoin(url, img_url)
                mapping[url] = img_url
                print(f"  Found image: {img_url}")
            else:
                print(f"  No image found for: {url}")
        except Exception as e:
            print(f"  Error fetching {url}: {e}")

    # Write to JS file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("export const productImages = ")
        f.write(json.dumps(mapping, indent=2, ensure_ascii=False))
        f.write(";\n")
    print(f"Done! Mapping saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

