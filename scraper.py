#!/usr/bin/env python3
"""Equalite Product Scraper - Main Entry Point.

Scrapes all products from Equalite.nl, generates SigLIP embeddings,
and imports everything into Supabase.

Usage:
    python scraper.py                     # Full scrape
    python scraper.py --skip-embeddings   # Scrape without embeddings
    python scraper.py --resume            # Resume from progress file
    python scraper.py --limit 10          # Scrape only 10 products
    python scraper.py --skip-existing     # Skip products already in DB
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from dotenv import load_dotenv
from tqdm import tqdm

from src.collector import scrape_all_product_urls
from src.database import DatabaseManager
from src.embeddings import (
    generate_image_embedding,
    generate_text_embedding,
)
from src.product import scrape_product

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("scraper")

# Suppress verbose library logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("supabase").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

# Progress file for resuming
PROGRESS_FILE = ".scraper_progress.json"

BATCH_SIZE = 5  # Number of products to process before saving progress


def load_progress() -> Dict[str, Any]:
    """Load progress from file for resume capability."""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            logger.warning("Failed to load progress file, starting fresh")
    return {
        "processed_urls": [],
        "skipped_urls": [],
        "failed_urls": [],
        "total_scraped": 0,
        "total_embedded": 0,
        "total_uploaded": 0,
        "timestamp": None,
    }


def save_progress(progress: Dict[str, Any]):
    """Save progress to file."""
    progress["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def build_info_text(product: Dict[str, Any]) -> str:
    """Build comprehensive info text for text embedding.
    
    Includes all available product information: title, description, category,
    gender, price, sale, brand, tags, size, country, and metadata summary.
    """
    parts = []
    if product.get("title"):
        parts.append(f"Title: {product['title']}")
    if product.get("description"):
        parts.append(f"Description: {product['description']}")
    if product.get("category"):
        parts.append(f"Category: {product['category']}")
    if product.get("gender"):
        parts.append(f"Gender: {product['gender']}")
    if product.get("price"):
        parts.append(f"Price: {product['price']}")
    if product.get("sale"):
        parts.append(f"Sale: {product['sale']}")
    if product.get("brand"):
        parts.append(f"Brand: {product['brand']}")
    if product.get("tags"):
        parts.append(f"Tags: {', '.join(product['tags'])}")
    if product.get("size"):
        parts.append(f"Sizes: {product['size']}")
    if product.get("country"):
        parts.append(f"Country: {product['country']}")
    if product.get("metadata"):
        # Include key metadata fields (exclude the full variants array which is long)
        try:
            md = json.loads(product["metadata"])
            summary_parts = []
            if md.get("sku"):
                summary_parts.append(f"SKU: {md['sku']}")
            if md.get("gtin"):
                summary_parts.append(f"GTIN: {md['gtin']}")
            if md.get("aggregate_rating"):
                rating = md['aggregate_rating']
                rating_str = f"Rating: {rating.get('ratingValue', '?')}/5 ({rating.get('reviewCount', '0')} reviews)"
                summary_parts.append(rating_str)
            if summary_parts:
                parts.append(" | ".join(summary_parts))
        except (json.JSONDecodeError, TypeError):
            pass

    return " | ".join(parts)


def process_product(
    url: str,
    db: Optional[DatabaseManager],
    skip_embeddings: bool,
) -> Optional[Dict[str, Any]]:
    """Scrape a single product and optionally generate embeddings."""
    product = scrape_product(url)
    if not product:
        return None

    if not skip_embeddings:
        # Generate image embedding
        image_emb = generate_image_embedding(product["image_url"])
        if image_emb:
            product["image_embedding"] = image_emb

        # Generate text embedding from comprehensive info
        info_text = build_info_text(product)
        if info_text:
            text_emb = generate_text_embedding(info_text)
            if text_emb:
                product["info_embedding"] = text_emb

    return product


def run_scrape(
    limit: Optional[int] = None,
    skip_embeddings: bool = False,
    resume: bool = False,
    skip_existing: bool = False,
):
    """Run the full scraping pipeline."""
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        logger.error(
            "SUPABASE_URL and SUPABASE_KEY must be set in .env file"
        )
        sys.exit(1)

    # Initialize database manager
    db = DatabaseManager(supabase_url, supabase_key)

    # Load progress if resuming
    progress = load_progress() if resume else {
        "processed_urls": [],
        "skipped_urls": [],
        "failed_urls": [],
        "total_scraped": 0,
        "total_embedded": 0,
        "total_uploaded": 0,
        "timestamp": None,
    }

    processed_set = set(progress.get("processed_urls", []))
    skipped_set = set(progress.get("skipped_urls", []))
    failed_set = set(progress.get("failed_urls", []))

    # Get existing product IDs if skip_existing
    existing_ids: Set[str] = set()
    if skip_existing:
        logger.info("Fetching existing product IDs from database...")
        existing_ids = db.get_existing_ids()
        logger.info("Found %d existing products", len(existing_ids))

    # Step 1: Collect all product URLs
    logger.info("=" * 60)
    logger.info("STEP 1: Collecting product URLs from collection pages")
    logger.info("=" * 60)

    all_urls = scrape_all_product_urls()
    if not all_urls:
        logger.error("No product URLs found. Exiting.")
        sys.exit(1)

    logger.info("Found %d total product URLs", len(all_urls))

    # Filter out already processed URLs
    urls_to_process = [
        u for u in all_urls
        if u not in processed_set
        and u not in skipped_set
        and u not in failed_set
    ]

    if not urls_to_process:
        logger.info("All URLs have already been processed!")
        return

    # Apply limit
    if limit and limit > 0:
        urls_to_process = urls_to_process[:limit]

    logger.info(
        "Processing %d new URLs (already processed: %d, skipped: %d, failed: %d)",
        len(urls_to_process),
        len(processed_set),
        len(skipped_set),
        len(failed_set),
    )

    # Step 2: Process each product
    logger.info("=" * 60)
    logger.info("STEP 2: Scraping products and generating embeddings")
    logger.info("=" * 60)

    pbar = tqdm(urls_to_process, desc="Processing products", unit="product")
    batch_products: List[Dict[str, Any]] = []

    for idx, url in enumerate(pbar):
        # Check if product already exists in DB
        product_id = None
        if skip_existing:
            # Generate product ID the same way as in product.py
            from urllib.parse import urlparse
            parsed = urlparse(url)
            handle = parsed.path.rstrip("/").split("/")[-1]
            product_id = f"equalite-{handle}"
            if product_id in existing_ids:
                pbar.set_description(f"Skipping existing: {handle}")
                progress["skipped_urls"].append(url)
                continue

        pbar.set_description(f"Processing: {url.split('/')[-1][:40]}")

        try:
            product = process_product(url, db, skip_embeddings)
        except Exception as e:
            logger.error("Unexpected error processing %s: %s", url, e)
            progress["failed_urls"].append(url)
            save_progress(progress)
            continue

        if product is None:
            progress["failed_urls"].append(url)
            save_progress(progress)
            continue

        # If skip_existing, double-check by product ID
        if skip_existing and product_id and product_id in existing_ids:
            progress["skipped_urls"].append(url)
            continue

        progress["total_scraped"] += 1
        if not skip_embeddings and product.get("image_embedding"):
            progress["total_embedded"] += 1

        batch_products.append(product)
        progress["processed_urls"].append(url)

        # Batch upload to database
        if len(batch_products) >= BATCH_SIZE:
            upload_batch(batch_products, db, progress)
            batch_products = []

        # Save progress periodically
        if (idx + 1) % BATCH_SIZE == 0:
            save_progress(progress)

    # Upload remaining products
    if batch_products:
        upload_batch(batch_products, db, progress)

    # Final progress save
    save_progress(progress)

    # Print summary
    print()
    logger.info("=" * 60)
    logger.info("SCRAPING COMPLETE")
    logger.info("=" * 60)
    logger.info("Total URLs found:     %d", len(all_urls))
    logger.info("Products scraped:     %d", progress["total_scraped"])
    logger.info("Products embedded:    %d", progress["total_embedded"])
    logger.info("Products uploaded:    %d", progress["total_uploaded"])
    logger.info("Failed URLs:          %d", len(progress.get("failed_urls", [])))
    logger.info("Skipped (existing):   %d", len(progress.get("skipped_urls", [])))
    logger.info("=" * 60)


def upload_batch(
    batch_products: List[Dict[str, Any]],
    db: DatabaseManager,
    progress: Dict[str, Any],
):
    """Upload a batch of products to Supabase."""
    try:
        success, failed = db.batch_upsert(batch_products)
        progress["total_uploaded"] += success
        if failed > 0:
            logger.warning(
                "Batch upload: %d succeeded, %d failed",
                success,
                failed,
            )
    except Exception as e:
        logger.error("Batch upload error: %s", e)
        progress["failed_urls"].extend(
            [p["product_url"] for p in batch_products]
        )


def main():
    parser = argparse.ArgumentParser(
        description="Equalite Product Scraper - scrape all products and embed with SigLIP",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding generation (faster, for testing)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from saved progress",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of products to process (for testing)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip products that already exist in the database",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose debug logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger("scraper").setLevel(logging.DEBUG)
        logging.getLogger("src").setLevel(logging.DEBUG)

    logger.info(
        "Starting Equalite Scraper (embeddings: %s, resume: %s, limit: %s, skip-existing: %s)",
        "disabled" if args.skip_embeddings else "enabled",
        "yes" if args.resume else "no",
        args.limit if args.limit else "all",
        "yes" if args.skip_existing else "no",
    )

    # Warn about model loading time
    if not args.skip_embeddings:
        logger.info(
            "Note: First run will download the SigLIP model (~1GB). "
            "This may take several minutes."
        )

    start_time = time.time()
    run_scrape(
        limit=args.limit,
        skip_embeddings=args.skip_embeddings,
        resume=args.resume,
        skip_existing=args.skip_existing,
    )
    elapsed = time.time() - start_time
    logger.info("Total execution time: %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)


if __name__ == "__main__":
    main()
