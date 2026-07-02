"""core/capabilities/web_asset_handler.py — Image Search, Download & Validation
================================================================================
General-purpose web asset handler. Searches, downloads, validates, and
manages images and other web assets.

NOT hardcoded for any specific content. The LLM + planner decides what
to search for; this layer handles the HOW.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus, urlparse

from core.container import ServiceContainer
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.network_gateway import get_network_gateway

logger = logging.getLogger("Aura.WebAssetHandler")


class WebAssetHandler:
    """Search, download, validate, and manage web assets.

    Usage:
        handler = get_web_asset_handler()
        results = await handler.search_images("mountain landscape")
        path = await handler.download_image(results[0]["url"], "~/Documents/Aura/images")
        validation = await handler.validate_image(path)
    """

    DEFAULT_SAVE_DIR = "~/.aura/data/assets"
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

    def __init__(self) -> None:
        self._download_count = 0
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        ServiceContainer.register_instance("web_asset_handler", self, required=False)
        self._started = True
        logger.info("WebAssetHandler ONLINE")

    async def search_images(
        self, query: str, count: int = 5, min_width: int = 800, min_height: int = 600
    ) -> List[Dict[str, str]]:
        """Search for images using DuckDuckGo image search.

        Returns list of {url, title, source, thumbnail} dicts.
        """
        try:
            results = await self._ddg_image_search(query, count * 2)

            # Filter by minimum dimensions if info available
            filtered = []
            for r in results:
                # Accept all initially — we'll validate after download
                filtered.append(r)
                if len(filtered) >= count:
                    break

            logger.info("Image search '%s': %d results", query[:30], len(filtered))
            return filtered

        except (OSError, RuntimeError) as e:
            record_degradation("web_asset.search", e)
            return []

    async def _ddg_image_search(self, query: str, count: int) -> List[Dict[str, str]]:
        """Scrape DuckDuckGo image search results."""
        url = f"https://duckduckgo.com/?q={quote_plus(query)}&iax=images&ia=images"
        try:
            response = await get_network_gateway().request_async(
                "GET",
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
                },
                timeout=10,
                read_only=True,
                source="web_asset_handler.ddg_image_search",
            )
            if not response.get("ok"):
                raise RuntimeError(str(response.get("error") or response.get("status_code")))
            html = bytes(response.get("content", b"")).decode("utf-8", errors="replace")

            # Extract vqd token for API call
            vqd_match = re.search(r"vqd=['\"]([^'\"]+)", html)
            if not vqd_match:
                return await self._fallback_image_search(query, count)

            vqd = vqd_match.group(1)

            # Call DuckDuckGo image API
            api_url = (
                f"https://duckduckgo.com/i.js?l=us-en&o=json&q={quote_plus(query)}"
                f"&vqd={vqd}&f=size:Large&p=1"
            )
            response = await get_network_gateway().request_async(
                "GET",
                api_url,
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Referer": "https://duckduckgo.com/",
                },
                timeout=10,
                read_only=True,
                source="web_asset_handler.ddg_image_api",
            )
            if not response.get("ok"):
                raise RuntimeError(str(response.get("error") or response.get("status_code")))
            data = bytes(response.get("content", b"")).decode("utf-8", errors="replace")

            parsed = json.loads(data)
            results = []
            for item in parsed.get("results", [])[:count]:
                img_url = item.get("image", "")
                if img_url and img_url.startswith("http"):
                    results.append({
                        "url": img_url,
                        "title": item.get("title", ""),
                        "source": item.get("source", urlparse(img_url).netloc),
                        "thumbnail": item.get("thumbnail", ""),
                        "width": str(item.get("width", "")),
                        "height": str(item.get("height", "")),
                    })

            return results

        except (json.JSONDecodeError, OSError, RuntimeError, TypeError, ValueError) as e:
            record_degradation("web_asset.ddg_image_search", e)
            logger.debug("DDG image API failed: %s", e)
            return await self._fallback_image_search(query, count)

    async def _fallback_image_search(self, query: str, count: int) -> List[Dict[str, str]]:
        """Fallback: use Unsplash Source for free images."""
        results = []
        for i in range(min(count, 5)):
            results.append({
                "url": f"https://source.unsplash.com/1920x1080/?{quote_plus(query)}&sig={i}",
                "title": f"{query} (Unsplash #{i+1})",
                "source": "unsplash.com",
            })
        return results

    async def download_image(
        self, url: str, save_dir: str = "", filename: str = ""
    ) -> str:
        """Download an image, validate it, and save locally.

        Returns the local file path, or empty string on failure.
        """
        if not save_dir:
            save_dir = os.path.expanduser(self.DEFAULT_SAVE_DIR)

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        try:
            response = await get_network_gateway().request_async(
                "GET",
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
                },
                timeout=30,
                read_only=True,
                source="web_asset_handler.download_image",
            )
            if not response.get("ok"):
                raise RuntimeError(str(response.get("error") or response.get("status_code")))

            # Check content type
            headers = response.get("headers", {})
            content_type = str(headers.get("Content-Type", ""))
            if not content_type.startswith("image/"):
                logger.warning("Not an image: %s (type=%s)", url[:60], content_type)
                # Try anyway — some servers don't set correct type

            # Read data with size limit
            data = bytes(response.get("content", b""))
            if len(data) > self.MAX_FILE_SIZE:
                logger.warning("Image too large: %s (%d bytes)", url[:60], len(data))
                return ""

            if len(data) < 100:
                logger.warning("Image too small: %s (%d bytes)", url[:60], len(data))
                return ""

            # Validate image header
            is_valid, fmt = self._validate_image_header(data)
            if not is_valid:
                logger.warning("Invalid image data from %s", url[:60])
                return ""

            # Determine filename
            if not filename:
                # Generate from URL or hash
                parsed = urlparse(url)
                url_path = parsed.path.split("/")[-1] if parsed.path else ""
                if url_path and "." in url_path:
                    filename = url_path
                else:
                    ext = {"jpeg": ".jpg", "png": ".png", "webp": ".webp", "gif": ".gif"}.get(fmt, ".jpg")
                    filename = f"image_{hashlib.sha256(data).hexdigest()[:12]}{ext}"

            # Sanitize filename
            filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)[:200]

            # Save
            file_path = save_path / filename
            if file_path.exists():
                # Version it
                stem = file_path.stem
                suffix = file_path.suffix
                for v in range(2, 20):
                    candidate = save_path / f"{stem}_v{v}{suffix}"
                    if not candidate.exists():
                        file_path = candidate
                        break

            await get_file_write_gateway().write_bytes_async(
                file_path,
                data,
                source="web_asset_handler.download_image",
            )
            self._download_count += 1
            logger.info("Downloaded image: %s (%d bytes, %s)", file_path.name, len(data), fmt)
            return str(file_path)

        except (OSError, RuntimeError, TypeError, ValueError) as e:
            record_degradation("web_asset.download", e)
            logger.debug("Image download failed from %s: %s", url[:60], e)
            return ""

    async def validate_image(self, path: str) -> Dict[str, Any]:
        """Validate an image file."""
        p = Path(path)
        if not p.exists():
            return {"valid": False, "error": "File not found"}

        data = p.read_bytes()
        is_valid, fmt = self._validate_image_header(data)

        result: Dict[str, Any] = {
            "valid": is_valid,
            "format": fmt,
            "size_bytes": len(data),
            "path": str(p),
        }

        # Try to get dimensions
        try:
            from PIL import Image
            img = Image.open(str(p))
            result["width"] = img.width
            result["height"] = img.height
            result["mode"] = img.mode
        except ImportError:
            result["dimensions_available"] = False
        except (OSError, RuntimeError, ValueError) as exc:
            record_degradation("web_asset.validate_image_dimensions", exc)
            result["dimensions_available"] = False

        return result

    @staticmethod
    def _validate_image_header(data: bytes) -> tuple[bool, str]:
        """Check image magic bytes."""
        if len(data) < 8:
            return False, "unknown"
        if data[:2] == b"\xff\xd8":
            return True, "jpeg"
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return True, "png"
        if data[8:12] == b"WEBP":
            return True, "webp"
        if data[:4] == b"GIF8":
            return True, "gif"
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return True, "webp"
        # BMP
        if data[:2] == b"BM":
            return True, "bmp"
        return False, "unknown"

    def get_status(self) -> Dict[str, Any]:
        return {"downloads": self._download_count}


_instance: Optional[WebAssetHandler] = None


def get_web_asset_handler() -> WebAssetHandler:
    global _instance
    if _instance is None:
        _instance = WebAssetHandler()
    return _instance


__all__ = ["WebAssetHandler", "get_web_asset_handler"]
