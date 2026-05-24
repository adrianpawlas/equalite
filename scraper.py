#!/usr/bin/env python3
"""Equalite Product Scraper — Main Entry Point.

Scrapes all products from Equalite.nl, compares against existing DB records,
generates SigLIP embeddings only when needed, batch upserts, removes stale
products, and prints a detailed run summary.

Usage:
    python scraper.py                          # Full smart scrape
    python scraper.py --skip-embeddings        # Skip embeddings entirely
    python scraper.py --resume                 # Resume from progress file
    python scraper.py --limit 10               # Scrape only 10 products
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

from dotenv import load_dotenv
from tqdm import tqdm

from src.collector import scrape_all_product_urls
from src.database import DatabaseManager, ProductStatus
from src.embeddings import (
    generate_image_embedding,
    generate_text_embedding,
)
from src.product import scrape_product

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("scraper")

# Suppress verbose library logging
for lib in ["httpx", "urllib3", "supabase", "PIL", "transformers"]:
    logging.getLogger(lib).setLevel(logging.WARNING)

SOURCE_NAME = "scraper-equalite"
PROGRESS_FILE = ".scraper_progress.json"
TRACKING_FILE = ".scraper_tracking.json"

# Batch / delays
DB_BATCH_SIZE = 50          # products per database batch upsert
PROGRESS_SAVE_INTERVAL = 50 # save progress every N products
EMBEDDING_DELAY = 0.5       # seconds between embedding generations


# ------------------------------------------------------------------ #
# Tracking file (stale product detection)
# ------------------------------------------------------------------ #

def load_tracking() -> Dict[str, Any]:
    """Load the tracking file that records which products were seen each run.

    Schema:
    {
        "seen": {"<product_url>": "<last_seen_iso_timestamp>", ...},
        "missed": {"<product_url>": <consecutive_miss_count>, ...}
    }
    """
    if os.path.exists(TRACKING_FILE):
        try:
            with open(TRACKING_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            logger.warning("Failed to load tracking file, starting fresh")
    return {"seen": {}, "missed": {}}


def save_tracking(tracking: Dict[str, Any]):
    """Persist the tracking file."""
    with open(TRACKING_FILE, "w") as f:
        json.dump(tracking, f, indent=2)


def update_tracking(
    tracking: Dict[str, Any],
    seen_urls: Set[str],
    all_source_urls_in_db: Set[str],
) -> Tuple[List[str], List[str]]:
    """Update tracking data based on which URLs were seen this run.

    Returns ``(freshly_missed, to_delete)``:
    * ``freshly_missed`` — URLs that were NOT seen this run (missed count increased).
    * ``to_delete`` — URLs whose missed count reached >= 2 and should be deleted.
    """
    now = datetime.now(timezone.utc).isoformat()
    seen = tracking.get("seen", {})
    missed = tracking.get("missed", {})

    # Mark all seen URLs as seen (reset missed count)
    for url in seen_urls:
        seen[url] = now
        missed.pop(url, None)

    # Find which previously-seen products were NOT seen this run
    freshly_missed_entries: List[Tuple[str, int]] = []
    to_delete: List[str] = []

    for url in list(seen.keys()):
        if url not in seen_urls:
            # Product was seen before but not this run
            current_miss = missed.get(url, 0) + 1
            missed[url] = current_miss
            freshly_missed_entries.append((url, current_miss))

    # Mark products for deletion after 2 consecutive misses
    for url in list(missed.keys()):
        if missed[url] >= 2 and url in all_source_urls_in_db:
            to_delete.append(url)

    # Clean up — remove tracking entries for deleted products
    for url in to_delete:
        seen.pop(url, None)
        missed.pop(url, None)

    tracking["seen"] = seen
    tracking["missed"] = missed
    save_tracking(tracking)

    freshly_missed = [u for u, _ in freshly_missed_entries]
    return freshly_missed, to_delete


# ------------------------------------------------------------------ #
# Progress file (resume support)
# ------------------------------------------------------------------ #

def load_progress() -> Dict[str, Any]:
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            logger.warning("Failed to load progress file, starting fresh")
    return {
        "processed_urls": [],
        "failed_urls": [],
        "total_new": 0,
        "total_updated": 0,
        "total_unchanged": 0,
        "total_deleted": 0,
        "timestamp": None,
    }


def save_progress(progress: Dict[str, Any]):
    progress["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


# ------------------------------------------------------------------ #
# Embedding helpers
# ------------------------------------------------------------------ #

def build_info_text(product: Dict[str, Any]) -> str:
    """Build a comprehensive info string for text embedding."""
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
        try:
            md = json.loads(product["metadata"])
            summary = []
            if md.get("sku"):
                summary.append(f"SKU: {md['sku']}")
            if md.get("gtin"):
                summary.append(f"GTIN: {md['gtin']}")
            if md.get("aggregate_rating"):
                r = md["aggregate_rating"]
                summary.append(f"Rating: {r.get('ratingValue', '?')}/5 ({r.get('reviewCount', '0')} reviews)")
            if summary:
                parts.append(" | ".join(summary))
        except (json.JSONDecodeError, TypeError):
            pass
    return " | ".join(parts)


def generate_embeddings_for_product(
    product: Dict[str, Any],
    needs_image_embed: bool,
    needs_text_embed: bool,
) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """Generate requested embeddings for a product with staggered delay."""
    time.sleep(EMBEDDING_DELAY)

    image_emb = None
    info_emb = None

    if needs_image_embed and product.get("image_url"):
        image_emb = generate_image_embedding(product["image_url"])
        time.sleep(EMBEDDING_DELAY)

    if needs_text_embed:
        info_text = build_info_text(product)
        if info_text:
            info_emb = generate_text_embedding(info_text)

    return image_emb, info_emb


# ------------------------------------------------------------------ #
# Main pipeline
# ------------------------------------------------------------------ #

def run_scrape(
    limit: Optional[int] = None,
    skip_embeddings: bool = False,
    resume: bool = False,
):
    """Run the full smart scraping pipeline."""

    # -- Init database --------------------------------------------------
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        logger.error("SUPABASE_URL and SUPABASE_KEY must be set in .env file")
        sys.exit(1)

    db = DatabaseManager(supabase_url, supabase_key)

    # -- Load state -----------------------------------------------------
    tracking = load_tracking()
    progress = load_progress() if resume else {
        "processed_urls": [],
        "failed_urls": [],
        "total_new": 0,
        "total_updated": 0,
        "total_unchanged": 0,
        "total_deleted": 0,
        "timestamp": None,
    }
    processed_set = set(progress.get("processed_urls", []))
    failed_set = set(progress.get("failed_urls", []))

    # -- Step 1: Collect all product URLs --------------------------------
    logger.info("=" * 60)
    logger.info("STEP 1: Collecting product URLs from collection pages")
    logger.info("=" * 60)
    all_urls = scrape_all_product_urls()
    if not all_urls:
        logger.error("No product URLs found. Exiting.")
        sys.exit(1)
    logger.info("Found %d total product URLs", len(all_urls))

    # Filter out already-processed / failed URLs
    urls_to_process = [
        u for u in all_urls
        if u not in processed_set and u not in failed_set
    ]
    if not urls_to_process:
        logger.info("All URLs have already been processed!")
        return

    if limit and limit > 0:
        urls_to_process = urls_to_process[:limit]

    logger.info("Processing %d URLs", len(urls_to_process))

    # -- Step 2: Fetch existing products from DB -------------------------
    logger.info("=" * 60)
    logger.info("STEP 2: Fetching existing products from database")
    logger.info("=" * 60)
    existing_products = db.fetch_existing_products(SOURCE_NAME)
    all_source_urls_in_db: Set[str] = set(existing_products.keys())

    # Pre-classify each URL before scraping
    classification: Dict[str, Tuple[str, bool]] = {}
    for url in urls_to_process:
        existing = existing_products.get(url)
        if existing is None:
            classification[url] = (ProductStatus.NEW, True)
        else:
            # We'll properly classify after scraping (we need scraped data)
            # For now, mark as "unknown" — we classify after scraping
            classification[url] = ("pending", False)

    # -- Step 3: Process each product ------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 3: Scraping, comparing, and embedding products")
    logger.info("=" * 60)

    pbar = tqdm(urls_to_process, desc="Processing", unit="product")

    # Collectors
    newly_scraped: Dict[str, Dict[str, Any]] = {}
    updated_scraped: Dict[str, Dict[str, Any]] = {}
    unchanged_urls: List[str] = []
    seen_this_run: Set[str] = set()
    failed_urls_run: List[str] = []
    batch_for_db: List[Dict[str, Any]] = []

    new_count = 0
    updated_count = 0
    unchanged_count = 0

    for idx, url in enumerate(pbar):
        short = url.split("/")[-1][:45]
        seen_this_run.add(url)

        # ---- Scrape ----
        try:
            product = scrape_product(url)
        except Exception as e:
            logger.error("Error scraping %s: %s", url, e)
            failed_urls_run.append(url)
            continue

        if product is None:
            failed_urls_run.append(url)
            continue

        # ---- Compare with existing DB record ----
        existing = existing_products.get(url)
        status, image_changed = db.classify_product(product, existing)

        if status == ProductStatus.UNCHANGED:
            unchanged_count += 1
            unchanged_urls.append(url)
            pbar.set_description(f"Unchanged: {short}")
            continue  # Skip entirely — no DB update, no embeddings

        # ---- Generate embeddings (if needed) ----
        if not skip_embeddings:
            needs_image = status == ProductStatus.NEW or image_changed
            needs_text = True  # always generate text embedding for new/changed
            img_emb, txt_emb = generate_embeddings_for_product(
                product, needs_image, needs_text,
            )
            if img_emb is not None:
                product["image_embedding"] = img_emb
            if txt_emb is not None:
                product["info_embedding"] = txt_emb
        else:
            img_emb, txt_emb = None, None

        # ---- Collect for batch upsert ----
        if status == ProductStatus.NEW:
            new_count += 1
            pbar.set_description(f"New: {short}")
        else:
            updated_count += 1
            pbar.set_description(f"Updated: {short}")

        batch_for_db.append(product)

        # ---- Periodic batch upsert ----
        if len(batch_for_db) >= DB_BATCH_SIZE:
            success, fail, fail_urls = db.batch_upsert(batch_for_db)
            if fail_urls:
                failed_urls_run.extend(fail_urls)
            batch_for_db = []

        # ---- Periodic progress save ----
        if (idx + 1) % PROGRESS_SAVE_INTERVAL == 0:
            save_progress({
                "processed_urls": list(seen_this_run - set(failed_urls_run)),
                "failed_urls": failed_urls_run,
                "total_new": new_count,
                "total_updated": updated_count,
                "total_unchanged": unchanged_count,
                "total_deleted": progress["total_deleted"],
                "timestamp": None,
            })

    pbar.close()

    # -- Flush remaining batch ------------------------------------------
    if batch_for_db:
        success, fail, fail_urls = db.batch_upsert(batch_for_db)
        if fail_urls:
            failed_urls_run.extend(fail_urls)

    # -- Step 4: Handle stale products ----------------------------------
    logger.info("=" * 60)
    logger.info("STEP 4: Checking for stale products")
    logger.info("=" * 60)

    freshly_missed, to_delete = update_tracking(
        tracking, seen_this_run, all_source_urls_in_db,
    )

    deleted_count = 0
    if to_delete:
        logger.info(
            "Deleting %d stale products (missed >= 2 runs): %s",
            len(to_delete),
            to_delete,
        )
        del_success, del_fail = db.delete_products_by_urls(to_delete)
        deleted_count = del_success
    else:
        logger.info("No stale products to delete")

    if freshly_missed:
        logger.info(
            "%d products missed this run (will be deleted if missed again): %s",
            len(freshly_missed),
            freshly_missed,
        )

    # -- Save final progress ---------------------------------------------
    save_progress({
        "processed_urls": list(seen_this_run - set(failed_urls_run)),
        "failed_urls": failed_urls_run,
        "total_new": new_count,
        "total_updated": updated_count,
        "total_unchanged": unchanged_count,
        "total_deleted": deleted_count,
        "timestamp": None,
    })

    # -- Step 5: Run summary ---------------------------------------------
    print()
    logger.info("=" * 60)
    logger.info("  RUN SUMMARY")
    logger.info("=" * 60)
    logger.info("  %-30s %d", "URLs discovered", len(all_urls))
    logger.info("  %-30s %d", "New products added", new_count)
    logger.info("  %-30s %d", "Products updated", updated_count)
    logger.info("  %-30s %d", "Unchanged (skipped)", unchanged_count)
    logger.info("  %-30s %d", "Stale products deleted", deleted_count)
    logger.info("  %-30s %d", "Failed", len(failed_urls_run))
    logger.info("  %-30s %d", "Products missed (1st run)",
                len(freshly_missed))
    logger.info("=" * 60)

    # Log failed URLs if any
    if failed_urls_run:
        logger.warning("Failed URLs (%d):", len(failed_urls_run))
        for fu in failed_urls_run:
            logger.warning("  - %s", fu)


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Equalité Product Scraper — smart pipeline with upsert, "
                    "stale detection, and batch operations",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding generation entirely",
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
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose debug logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger("scraper").setLevel(logging.DEBUG)
        logging.getLogger("src").setLevel(logging.DEBUG)

    logger.info(
        "Starting Equalité Scraper (embeddings: %s, resume: %s, limit: %s)",
        "disabled" if args.skip_embeddings else "enabled",
        "yes" if args.resume else "no",
        args.limit if args.limit else "all",
    )

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
    )
    elapsed = time.time() - start_time
    logger.info(
        "Total execution time: %.1f seconds (%.1f minutes)",
        elapsed, elapsed / 60,
    )


if __name__ == "__main__":
    main()
