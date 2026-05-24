"""Supabase database operations for product data.

Handles smart upsert logic, stale product detection, batch operations,
and comprehensive product comparison.
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from supabase import Client, create_client

logger = logging.getLogger(__name__)

TABLE_NAME = "products"
SOURCE_NAME = "scraper-equalite"

# Comparison fields — if any of these differ, the product is "changed"
COMPARISON_FIELDS: List[str] = [
    "title",
    "price",
    "sale",
    "image_url",
    "description",
    "category",
    "gender",
    "size",
    "tags",
    "additional_images",
]


class ProductStatus:
    """Status of a product after comparison with the database."""
    NEW = "new"
    CHANGED = "changed"
    UNCHANGED = "unchanged"


class DatabaseManager:
    """Manages Supabase database operations for product data."""

    def __init__(self, supabase_url: str, supabase_key: str):
        self.client: Client = create_client(supabase_url, supabase_key)
        logger.info("Supabase client initialized")

    # ------------------------------------------------------------------ #
    # Fetching
    # ------------------------------------------------------------------ #

    def fetch_existing_products(self, source: str = SOURCE_NAME) -> Dict[str, Dict[str, Any]]:
        """Fetch all existing products for a given source.

        Returns a dict keyed by ``product_url`` for fast lookup.
        """
        products_by_url: Dict[str, Dict[str, Any]] = {}
        try:
            offset = 0
            page_size = 1000

            while True:
                result = (
                    self.client.table(TABLE_NAME)
                    .select("*")
                    .eq("source", source)
                    .range(offset, offset + page_size - 1)
                    .execute()
                )

                data = result.data if hasattr(result, "data") else []
                if not data:
                    break

                for row in data:
                    url = row.get("product_url")
                    if url:
                        products_by_url[url] = row

                if len(data) < page_size:
                    break
                offset += page_size

        except Exception as e:
            logger.warning("Error fetching existing products: %s", e)

        logger.info("Fetched %d existing products for source '%s'", len(products_by_url), source)
        return products_by_url

    def get_existing_ids(self) -> Set[str]:
        """Get all existing product IDs from the database (legacy)."""
        existing_ids: Set[str] = set()
        try:
            offset = 0
            page_size = 1000
            while True:
                result = (
                    self.client.table(TABLE_NAME)
                    .select("id")
                    .range(offset, offset + page_size - 1)
                    .execute()
                )
                data = result.data if hasattr(result, "data") else []
                if not data:
                    break
                for row in data:
                    existing_ids.add(row["id"])
                if len(data) < page_size:
                    break
                offset += page_size
        except Exception as e:
            logger.warning("Error fetching existing IDs: %s", e)
        return existing_ids

    # ------------------------------------------------------------------ #
    # Comparison
    # ------------------------------------------------------------------ #

    @staticmethod
    def classify_product(
        scraped: Dict[str, Any],
        existing: Optional[Dict[str, Any]],
    ) -> Tuple[str, bool]:
        """Classify a product compared to its existing database record.

        Returns ``(status, image_changed)`` where:

        * ``status`` is one of ``ProductStatus.NEW`` / ``CHANGED`` / ``UNCHANGED``.
        * ``image_changed`` is ``True`` only when the image URL differs (triggers
          re-embedding).
        """
        if existing is None:
            return ProductStatus.NEW, True

        image_changed = False
        for field in COMPARISON_FIELDS:
            scraped_val = scraped.get(field)
            existing_val = existing.get(field)

            # Normalise both to string (or None) for comparison
            s = _normalise(scraped_val)
            e = _normalise(existing_val)

            if s != e:
                if field == "image_url":
                    image_changed = True
                return ProductStatus.CHANGED, image_changed

        return ProductStatus.UNCHANGED, False

    # ------------------------------------------------------------------ #
    # Batch upsert
    # ------------------------------------------------------------------ #

    def batch_upsert(
        self,
        products: List[Dict[str, Any]],
        batch_size: int = 50,
        max_retries: int = 3,
    ) -> Tuple[int, int, List[str]]:
        """Upsert multiple products in batches.

        Args:
            products: List of product records to upsert.
            batch_size: Number of products per batch (default 50).
            max_retries: Retry attempts per batch on failure.

        Returns:
            Tuple of ``(success_count, fail_count, failed_product_urls)``.
        """
        success = 0
        failed_urls: List[str] = []

        records = [
            self._build_record(p) for p in products
        ]

        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            batch_urls = [r.get("product_url", "?") for r in batch]

            for attempt in range(1, max_retries + 1):
                try:
                    self.client.table(TABLE_NAME).upsert(
                        batch,
                        on_conflict="id",
                        ignore_duplicates=False,
                    ).execute()
                    success += len(batch)
                    logger.debug(
                        "Batch upserted %d products (attempt %d/%d) — running: %d/%d",
                        len(batch),
                        attempt,
                        max_retries,
                        success,
                        success + len(failed_urls),
                    )
                    break  # success → exit retry loop
                except Exception as e:
                    logger.warning(
                        "Batch upsert attempt %d/%d failed for %d products: %s",
                        attempt,
                        max_retries,
                        len(batch),
                        e,
                    )
                    if attempt < max_retries:
                        time.sleep(2 ** attempt)
                    else:
                        failed_urls.extend(batch_urls)
                        logger.error(
                            "Failed to upsert %d products after %d attempts: %s",
                            len(batch),
                            max_retries,
                            batch_urls,
                        )

        return success, len(failed_urls), failed_urls

    # ------------------------------------------------------------------ #
    # Deletion (stale products)
    # ------------------------------------------------------------------ #

    def delete_products_by_urls(
        self,
        product_urls: List[str],
        source: str = SOURCE_NAME,
        max_retries: int = 3,
    ) -> Tuple[int, int]:
        """Delete products by ``product_url`` and ``source``.

        Returns ``(success_count, fail_count)``.
        """
        success = 0
        failed = 0

        for url in product_urls:
            for attempt in range(1, max_retries + 1):
                try:
                    self.client.table(TABLE_NAME).delete().eq(
                        "product_url", url
                    ).eq("source", source).execute()
                    success += 1
                    logger.info("Deleted stale product: %s", url)
                    break
                except Exception as e:
                    logger.warning(
                        "Delete attempt %d/%d failed for %s: %s",
                        attempt,
                        max_retries,
                        url,
                        e,
                    )
                    if attempt < max_retries:
                        time.sleep(2 ** attempt)
                    else:
                        logger.error("Failed to delete product: %s", url)
                        failed += 1

        return success, failed

    # ------------------------------------------------------------------ #
    # Record builder
    # ------------------------------------------------------------------ #

    def _build_record(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """Build a database record from a product dict (with embeddings if present)."""
        record: Dict[str, Any] = {
            "id": product["id"],
            "source": product.get("source"),
            "product_url": product.get("product_url"),
            "affiliate_url": product.get("affiliate_url"),
            "image_url": product.get("image_url"),
            "brand": product.get("brand"),
            "title": product.get("title"),
            "description": product.get("description"),
            "category": product.get("category"),
            "gender": product.get("gender"),
            "size": product.get("size"),
            "second_hand": product.get("second_hand", False),
            "country": product.get("country"),
            "tags": product.get("tags"),
            "price": product.get("price"),
            "sale": product.get("sale"),
            "additional_images": product.get("additional_images"),
            "metadata": product.get("metadata"),
            "other": product.get("other"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Embeddings — only include if they were actually generated
        if product.get("image_embedding") is not None:
            record["image_embedding"] = product["image_embedding"]
        if product.get("info_embedding") is not None:
            record["info_embedding"] = product["info_embedding"]

        return record


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _normalise(value: Any) -> Optional[str]:
    """Normalise a value for comparison (string/None)."""
    if value is None:
        return None
    if isinstance(value, list):
        return json.dumps(sorted(str(v) for v in value), ensure_ascii=False)
    return str(value).strip()
