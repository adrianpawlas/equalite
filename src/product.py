"""Product page scraper for Equalite Shopify store.

Extracts all product data from individual product pages using JSON-LD,
meta tags, and HTML parsing.
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlencode, urljoin, urlparse

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

SOURCE_NAME = "scraper-equalite"
BRAND_NAME = "Equalité"

# Delay between product page requests (seconds)
REQUEST_DELAY = 0.5


def fetch_product_page(url: str, max_retries: int = 3) -> Optional[str]:
    """Fetch a single product page and return its HTML."""
    time.sleep(REQUEST_DELAY)
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=30)
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
                time.sleep(2**attempt)
    return None


def parse_json_ld(html: str) -> Optional[Dict[str, Any]]:
    """Extract the Product JSON-LD from the page HTML."""
    pattern = re.compile(
        r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        re.DOTALL,
    )
    for match in pattern.finditer(html):
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict) and data.get("@type") == "Product":
                return data
            # Sometimes it's wrapped in an array
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and item.get("@type") == "Product":
                        return item
        except (json.JSONDecodeError, KeyError):
            continue
    return None


def extract_meta_tags(html: str) -> Dict[str, str]:
    """Extract Open Graph and other meta tags."""
    soup = BeautifulSoup(html, "lxml")
    meta = {}
    for tag in soup.find_all("meta"):
        if tag.get("property"):
            meta[tag["property"]] = tag.get("content", "")
        if tag.get("name"):
            meta[tag["name"]] = tag.get("content", "")
    return meta


def parse_category(category_str: Optional[str]) -> Optional[str]:
    """Parse category string. Split compound categories like 'Sweaters & Hoodies'."""
    if not category_str or not category_str.strip():
        return None

    cleaned = category_str.strip()

    # Compound categories like "Sweaters & Hoodies" -> split
    if " & " in cleaned:
        parts = [p.strip() for p in cleaned.split(" & ")]
        return ", ".join(parts)
    if " / " in cleaned:
        parts = [p.strip() for p in cleaned.split(" / ")]
        return ", ".join(parts)

    return cleaned


def parse_price(price_value) -> Optional[float]:
    """Parse price to a float. Handles various formats."""
    if price_value is None:
        return None
    try:
        return float(price_value)
    except (ValueError, TypeError):
        return None


def extract_price_info(
    json_ld: Dict[str, Any],
    meta: Dict[str, str],
) -> Tuple[Optional[str], Optional[str]]:
    """Extract price and sale price from product data.

    Returns (price_str, sale_str) where:
    - price_str is the original/regular price string (e.g. "90.00EUR")
    - sale_str is the sale price string (None if no sale, e.g. "65.00EUR")
    Currency priority: EUR first, then any others found.
    """
    offers = json_ld.get("offers", [])

    # If single offer object (not array), wrap in list
    if isinstance(offers, dict):
        offers = [offers]

    # Collect all unique prices from variants
    prices = set()
    for offer in offers:
        p = parse_price(offer.get("price"))
        if p is not None:
            prices.add(p)

    # Determine currency
    currency = "EUR"
    if offers and isinstance(offers[0], dict):
        currency = offers[0].get("priceCurrency", "EUR")

    # --- Sale Detection ---
    # Method 1: OG meta tags (most reliable for Shopify)
    og_price = meta.get("og:price:amount")
    og_original = meta.get("product:original_price") or meta.get(
        "product:price:original_amount"
    )

    if og_original and og_price:
        try:
            original = float(og_original)
            current = float(og_price)
            if original > current:
                return (
                    f"{original:.2f}{currency}",
                    f"{current:.2f}{currency}",
                )
        except (ValueError, TypeError):
            pass

    # Method 2: Check Shopify's JSON product data (often in script tags)
    # Some themes embed a `compare_at_price` in the initial product JSON

    # Method 3: No sale detected — use the variant price(s)
    if prices:
        sorted_prices = sorted(prices)
        if len(sorted_prices) == 1:
            return f"{sorted_prices[0]:.2f}{currency}", None

        # Multiple variant prices — join them as the price
        price_str = ", ".join(
            f"{p:.2f}{currency}" for p in sorted_prices
        )
        return price_str, None

    # Fallback: use OG meta tag
    og_price_val = og_price or meta.get("product:price:amount")
    if og_price_val:
        try:
            p = float(og_price_val)
            return f"{p:.2f}{currency}", None
        except (ValueError, TypeError):
            pass

    return None, None


def clean_image_url(src: str) -> str:
    """Clean and normalize an image URL to full resolution."""
    if not src:
        return src
    if src.startswith("//"):
        src = "https:" + src
    elif src.startswith("/"):
        src = urljoin(BASE_URL, src)

    # Upgrade to full resolution
    parsed = urlparse(src)
    qs = parse_qs(parsed.query)
    qs["width"] = ["2048"]
    new_query = urlencode(qs, doseq=True)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{new_query}"


def extract_images(json_ld: Dict[str, Any], html: str) -> Tuple[
    Optional[str], Optional[str]
]:
    """Extract main image URL and additional images.

    Returns (main_image_url, additional_images_str)
    Additional images are separated by ' , ' as requested.
    """
    # Get main image from JSON-LD
    main_image = None
    ld_image = json_ld.get("image")
    if isinstance(ld_image, dict):
        main_image = ld_image.get("url") or ld_image.get("image")
    elif isinstance(ld_image, str):
        main_image = ld_image

    if main_image:
        main_image = clean_image_url(main_image)

    # Get additional images — focus on product gallery images
    soup = BeautifulSoup(html, "lxml")
    additional_images = set()

    # Strategy: look for images in product media/gallery containers
    # Shopify uses various selectors for product media
    product_media_selectors = [
        "img[data-media-type]",            # Shopify media attribute
        "img.product__media",              # Common Shopify class
        ".product-gallery img",            # Gallery container
        ".product-media img",              # Media container
        "[data-product-gallery] img",      # Data attribute gallery
        ".media img",                      # Generic media container
        ".product__media img",             # Product media section
        ".product-single__media img",      # Alternative product media
    ]

    found_srcs = set()
    for selector in product_media_selectors:
        for img in soup.select(selector):
            src = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
            if src:
                found_srcs.add(src)

    # Also look for all images in the product form/main section
    product_section = soup.select_one(
        ".product, .product-single, main, [data-section-type='product'], "
        ".product-section, #shopify-section-product-template, "
        ".shopify-section--product"
    )
    if product_section:
        for img in product_section.find_all("img"):
            src = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
            if src:
                found_srcs.add(src)

    for src in found_srcs:
        if not src or "/cdn/shop/files/" not in src:
            continue
        clean_src = clean_image_url(src)
        if clean_src and clean_src != main_image:
            additional_images.add(clean_src)

    additional_images.discard(main_image)

    additional_str = None
    if additional_images:
        additional_str = " , ".join(sorted(additional_images))

    return main_image, additional_str


def extract_gender(json_ld: Dict[str, Any], category: Optional[str]) -> Optional[str]:
    """Extract gender from product data."""
    name = json_ld.get("name", "").lower()
    category_lower = (category or "").lower()

    women_keywords = ["women", "woman", "female", "womens"]
    men_keywords = ["men", "man", "male"]

    for kw in women_keywords:
        if kw in name or kw in category_lower:
            return "female"
    for kw in men_keywords:
        if kw in name or kw in category_lower:
            return "male"

    # Equalite is primarily a menswear/unisex brand
    return "unisex"


def extract_sizes(json_ld: Dict[str, Any]) -> Optional[str]:
    """Extract available sizes from product offers."""
    offers = json_ld.get("offers", [])
    if isinstance(offers, dict):
        offers = [offers]

    sizes = []
    for offer in offers:
        name = offer.get("name", "")
        # Extract size from variant name (e.g., "OFF-WHITE / XXS" -> "XXS")
        if " / " in name:
            size_part = name.split(" / ", 1)[1].strip()
        else:
            size_part = name.strip()

        if size_part and size_part not in sizes:
            sizes.append(size_part)

    if sizes:
        return ", ".join(sizes)
    return None


def extract_description(json_ld: Dict[str, Any]) -> Optional[str]:
    """Extract product description."""
    desc = json_ld.get("description")
    if desc:
        desc = desc.strip()
        desc = re.sub(r'\s+', ' ', desc)
        return desc
    return None


def generate_product_id(product_url: str) -> str:
    """Generate a unique product ID from the URL."""
    parsed = urlparse(product_url)
    handle = parsed.path.rstrip("/").split("/")[-1]
    return f"equalite-{handle}"


def scrape_product(url: str) -> Optional[Dict[str, Any]]:
    """Scrape a single product page and return structured product data."""
    logger.debug("Scraping product: %s", url)

    html = fetch_product_page(url)
    if not html:
        logger.error("Failed to fetch product page: %s", url)
        return None

    json_ld = parse_json_ld(html)
    if not json_ld:
        logger.error("No JSON-LD product data found for: %s", url)
        return None

    meta = extract_meta_tags(html)

    # Extract all product fields
    name = json_ld.get("name", "").strip()
    if not name:
        logger.warning("No product name found for: %s", url)
        return None

    # Category
    raw_category = json_ld.get("category")
    category = parse_category(raw_category) if raw_category else None

    # Price
    price_str, sale_str = extract_price_info(json_ld, meta)

    # Images
    main_image, additional_images = extract_images(json_ld, html)

    if not main_image:
        logger.warning("No main image found for: %s", url)
        return None

    # Description
    description = extract_description(json_ld)

    # Gender
    gender = extract_gender(json_ld, category)

    # Sizes
    size = extract_sizes(json_ld)

    # Build metadata as comprehensive JSON
    metadata = {
        "name": name,
        "description": description,
        "category": category,
        "gender": gender,
        "sizes": size,
        "price": price_str,
        "sale": sale_str,
        "brand": BRAND_NAME,
        "product_url": url,
        "image_url": main_image,
        "additional_images": additional_images,
        "variants": json_ld.get("offers", []),
        "sku": json_ld.get("sku"),
        "gtin": json_ld.get("gtin"),
        "aggregate_rating": json_ld.get("aggregateRating"),
    }

    # Build the full product record
    product = {
        "id": generate_product_id(url),
        "source": SOURCE_NAME,
        "product_url": url,
        "affiliate_url": None,
        "image_url": main_image,
        "brand": BRAND_NAME,
        "title": name,
        "description": description,
        "category": category,
        "gender": gender,
        "size": size,
        "second_hand": False,
        "country": "NL",
        "tags": [category] if category else None,
        "price": price_str,
        "sale": sale_str,
        "additional_images": additional_images,
        "metadata": json.dumps(metadata, ensure_ascii=False),
        "other": None,
    }

    return product
