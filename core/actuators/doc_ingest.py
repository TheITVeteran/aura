"""core/actuators/doc_ingest.py
============================
Document Ingest Actuator.
Parses text, PDF, HTML, and image files and loads them into Aura's semantic memory.
"""

import os
import re
import logging
from typing import Any, List
from core.actuators.actuator_registry import BaseActuator, ActuatorResult
from core.container import ServiceContainer

logger = logging.getLogger("Aura.DocumentIngest")


class DocumentIngestActuator(BaseActuator):
    """Actuator that parses external documents and indexes them in semantic memory."""

    requires_authority = True

    @property
    def name(self) -> str:
        return "document_ingest"

    @property
    def description(self) -> str:
        return "Extracts text from PDF, HTML, or text files and indexes them in the semantic memory."

    def validate_params(self, params: dict[str, Any]) -> bool:
        if not isinstance(params, dict) or "path" not in params:
            return False
        path = params["path"]
        if not isinstance(path, str) or not path.strip():
            return False
        return True

    def execute(self, params: dict[str, Any]) -> ActuatorResult:
        if not params.get("_aura_authorized"):
            return ActuatorResult(False, "Document ingestion requires ActuatorRegistry/AuthorityGateway authorization.", {})
        if not self.validate_params(params):
            return ActuatorResult(False, "Invalid path parameter.", {})

        path = os.path.abspath(params["path"])
        if not os.path.exists(path):
            return ActuatorResult(False, f"Target file '{path}' does not exist.", {})

        ext = os.path.splitext(path)[1].lower()
        extracted_text = ""

        try:
            if ext in (".html", ".htm"):
                extracted_text = self._parse_html(path)
            elif ext == ".pdf":
                extracted_text = self._parse_pdf(path)
            elif ext in (".png", ".jpg", ".jpeg"):
                extracted_text = self._parse_image(path)
            else:
                # Default to text file parsing
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    extracted_text = f.read()
        except (OSError, UnicodeError, RuntimeError, ImportError, AttributeError, TypeError, ValueError) as e:
            return ActuatorResult(False, f"Failed to parse document: {e}", {})

        if not extracted_text or not extracted_text.strip():
            return ActuatorResult(False, "Parsed document yielded no text content.", {})

        # Chunk the content
        chunks = self._chunk_text(extracted_text)
        
        memory_facade = ServiceContainer.get("memory_facade", default=None)
        if not memory_facade:
            return ActuatorResult(False, "MemoryFacade unavailable. Cannot index content.", {})

        indexed_count = 0
        try:
            from core.memory.memory_write_gateway import get_memory_write_gateway
            from core.runtime.gateways import MemoryWriteRequest
            from core.actuators.web_actuators import run_async_in_sync

            gateway = get_memory_write_gateway()
            # Index chunks
            for i, chunk in enumerate(chunks[:20]): # Limit to top 20 chunks to avoid saturation
                text = f"Document Ingestion: {os.path.basename(path)}\n\n{chunk}"
                metadata = {
                        "source": "document_ingestion",
                        "provenance_source": "document_ingestion",
                        "file_path": path,
                        "file_name": os.path.basename(path),
                        "chunk_index": i,
                        "importance": float(params.get("importance", 0.5))
                    }
                run_async_in_sync(
                    gateway.write(
                        MemoryWriteRequest(
                            content=text,
                            cause="document_ingest_actuator",
                            metadata={
                                **metadata,
                                "family": "episodic",
                            },
                            receipt_id=str(params.get("_capability_token_id") or ""),
                        )
                    )
                )
                maybe_result = memory_facade.add_memory(text=text, metadata=metadata)
                if hasattr(maybe_result, "__await__"):
                    run_async_in_sync(maybe_result)
                indexed_count += 1
        except (PermissionError, RuntimeError, OSError, AttributeError, TypeError, ValueError) as e:
            return ActuatorResult(False, f"Failed during indexing: {e}", {})

        msg = f"Successfully parsed '{os.path.basename(path)}' and indexed {indexed_count} chunks."
        return ActuatorResult(
            success=True,
            message=msg,
            updates={
                "file_path": path,
                "file_size_bytes": os.path.getsize(path),
                "chunks_indexed": indexed_count,
                "total_chunks": len(chunks)
            }
        )

    def _parse_html(self, path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()
        # Strip HTML tags
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _parse_pdf(self, path: str) -> str:
        try:
            import pypdf
            reader = pypdf.PdfReader(path)
            pages_text = []
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    pages_text.append(t)
            return "\n".join(pages_text)
        except ImportError:
            # Simple fallback: extract ascii characters/strings from binary
            with open(path, "rb") as f:
                content = f.read()
            # Find printable ASCII strings of length >= 4
            ascii_strings = re.findall(b"[ -~]{4,}", content)
            return " ".join(s.decode("ascii", errors="ignore") for s in ascii_strings)

    def _parse_image(self, path: str) -> str:
        try:
            from PIL import Image
            import pytesseract
            img = Image.open(path)
            return pytesseract.image_to_string(img)
        except ImportError:
            raise RuntimeError(
                f"OCR unavailable for image file {os.path.basename(path)}: PIL and pytesseract are required"
            )

    def _chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
        words = text.split()
        chunks = []
        step = chunk_size - overlap
        for i in range(0, len(words), step):
            chunk_words = words[i:i + chunk_size]
            chunks.append(" ".join(chunk_words))
            if i + chunk_size >= len(words):
                break
        return chunks
