"""Supabase database operations for product data."""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from supabase import Client, create_client

logger = logging.getLogger(__name__)

TABLE_NAME = "products"


class DatabaseManager:
    """Manages Supabase database operations for product data."""

    def __init__(self, supabase_url: str, supabase_key: str):
        self.client: Client = create_client(supabase_url, supabase_key)
        logger.info("Supabase client initialized")

    def upsert_product(
        self,
        product: Dict[str, Any],
        image_embedding: Optional[List[float]] = None,
        info_embedding: Optional[List[float]] = None,
    ) -> bool:
        """Upsert a single product to the database with optional embeddings."""
        record = self._build_record(product, image_embedding, info_embedding)

        try:
            result = self.client.table(TABLE_NAME).upsert(
                record,
                on_conflict="id",
                ignore_duplicates=False,
            ).execute()

            return True
        except Exception as e:
            logger.error(
                "Exception upserting product %s: %s",
                product["id"],
                e,
            )
            return False

    def batch_upsert(
        self,
        products: List[Dict[str, Any]],
        batch_size: int = 10,
    ) -> Tuple[int, int]:
        """Upsert multiple products in batches. Returns (success_count, fail_count)."""
        success = 0
        failed = 0

        for i in range(0, len(products), batch_size):
            batch = products[i : i + batch_size]
            records = [
                self._build_record(
                    p,
                    p.get("image_embedding"),
                    p.get("info_embedding"),
                )
                for p in batch
            ]

            try:
                self.client.table(TABLE_NAME).upsert(
                    records,
                    on_conflict="id",
                    ignore_duplicates=False,
                ).execute()
                success += len(records)
                logger.debug(
                    "Batch upserted %d products (running: %d/%d)",
                    len(records),
                    success,
                    success + failed,
                )
            except Exception as e:
                logger.error(
                    "Batch upsert error for %d products: %s",
                    len(records),
                    e,
                )
                failed += len(records)

        return success, failed

    def get_existing_ids(self) -> Set[str]:
        """Get all existing product IDs from the database."""
        existing_ids: Set[str] = set()
        try:
            offset = 0
            page_size = 1000

            while True:
                result = self.client.table(TABLE_NAME).select("id").range(
                    offset, offset + page_size - 1
                ).execute()

                data = result.data if hasattr(result, 'data') else []
                if data:
                    for row in data:
                        existing_ids.add(row["id"])
                    if len(data) < page_size:
                        break
                    offset += page_size
                else:
                    break
        except Exception as e:
            logger.warning("Error fetching existing IDs: %s", e)

        logger.info("Found %d existing products in database", len(existing_ids))
        return existing_ids

    def _build_record(
        self,
        product: Dict[str, Any],
        image_embedding: Optional[List[float]] = None,
        info_embedding: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Build a database record from a product dict."""
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

        if image_embedding is not None:
            record["image_embedding"] = image_embedding
        if info_embedding is not None:
            record["info_embedding"] = info_embedding

        return record
