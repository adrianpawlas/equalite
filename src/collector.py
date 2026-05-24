"""Collection page scraper for Equalite Shopify store.

Handles paginated collection pages and extracts all product URLs.
"""

import logging
import re
import time
from typing import List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE_URL = "https://equalite.nl"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,nl;q=0.8",
}


def extract_product_urls_from_html(html: str) -> List[str]:
    """Extract all unique product URLs from collection page HTML."""
    soup = BeautifulSoup(html, "lxml")
    urls: set[str] = set()

    # Method 1: Find all <a> tags with href containing /en/products/
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if "/en/products/" in href:
            full_url = urljoin(BASE_URL, href)
            # Remove any URL fragments and query params
            parsed = urlparse(full_url)
            clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            urls.add(clean_url)

    # Method 2: Regex search for any remaining product URLs
    # (catches URLs in script tags, data attributes, etc.)
    pattern = re.compile(r'/en/products/[a-zA-Z0-9_-]+')
    for match in pattern.finditer(html):
        full_url = urljoin(BASE_URL, match.group(0))
        parsed = urlparse(full_url)
        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        urls.add(clean_url)

    return sorted(urls)


def has_next_page(html: str) -> bool:
    """Check if there's a next page via <link rel='next'>."""
    soup = BeautifulSoup(html, "lxml")
    link_next = soup.find("link", rel="next")
    if link_next and link_next.get("href"):
        return True
    return False


def get_next_page_url(html: str) -> Optional[str]:
    """Get the next page URL from <link rel='next'>."""
    soup = BeautifulSoup(html, "lxml")
    link_next = soup.find("link", rel="next")
    if link_next and link_next.get("href"):
        return urljoin(BASE_URL, link_next["href"])
    return None


def scrape_collection_page(url: str, max_retries: int = 3) -> Optional[str]:
    """Fetch a single collection page and return its HTML."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(
                url,
                headers=DEFAULT_HEADERS,
                timeout=30,
            )
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            logger.warning(
                "Attempt %d/%d failed for %s: %s",
                attempt + 1,
                max_retries,
                url,
                e,
            )
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
    return None


def scrape_all_product_urls(
    start_url: str = f"{BASE_URL}/en/collections/shop-everything",
    delay: float = 1.0,
) -> List[str]:
    """Scrape all collection pages and return all product URLs."""
    all_product_urls: set[str] = set()
    current_url: Optional[str] = start_url
    page_num = 1

    logger.info("Starting collection scraping from: %s", start_url)

    while current_url:
        logger.info("Scraping collection page %d: %s", page_num, current_url)

        html = scrape_collection_page(current_url)
        if not html:
            logger.error("Failed to fetch collection page: %s", current_url)
            break

        product_urls = extract_product_urls_from_html(html)
        logger.info(
            "Found %d product URLs on page %d",
            len(product_urls),
            page_num,
        )

        if not product_urls:
            logger.info(
                "No products found on page %d — stopping pagination", page_num
            )
            break

        before_count = len(all_product_urls)
        all_product_urls.update(product_urls)
        new_count = len(all_product_urls) - before_count
        logger.info(
            "Added %d new product URLs (total unique: %d)",
            new_count,
            len(all_product_urls),
        )

        # Check for next page
        if has_next_page(html):
            current_url = get_next_page_url(html)
            page_num += 1
            time.sleep(delay)
        else:
            logger.info("No next page found — scraping complete")
            break

    result = sorted(all_product_urls)
    logger.info(
        "Collection scraping complete: %d total product URLs found",
        len(result),
    )
    return result
