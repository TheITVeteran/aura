# autonomy_engine/storage/vector_adapter.py
import logging

logger = logging.getLogger("Storage.VectorAdapter")

VECTOR_STORE_RECOVERABLE_ERRORS = (
    AttributeError,
    TypeError,
    ValueError,
    RuntimeError,
    OSError,
)


class VectorStoreAdapter:
    """
    Adapter layer for different vector store APIs.
    Implementations should pass the underlying store object.
    """

    def __init__(self, store):
        self.store = store

    def add(self, _id: str, embedding, text: str, metadata: dict = None):
        metadata = metadata or {}
        try:
            # Try common Chroma-like signature first
            if hasattr(self.store, "add") and callable(self.store.add):
                try:
                    return self.store.add(
                        documents=[text],
                        metadatas=[metadata],
                        ids=[_id],
                        embeddings=[embedding],
                    )
                except TypeError:
                    # Fallback: store.add(id, embedding, text, metadata)
                    try:
                        return self.store.add(_id, embedding, text, metadata)
                    except TypeError:  # Non-critical, fallback handled
                        # Give up to allow other fallback code below
                        pass  # Intentional silent fallback

            # FAISS-like or custom upsert
            if hasattr(self.store, "upsert"):
                # upsert signature varies; many accept tuples
                try:
                    return self.store.upsert([(_id, embedding, metadata, text)])
                except TypeError:
                    return self.store.upsert([(_id, embedding)])
            # If none matched, log and return None to allow graceful degradation
            logger.error("Unsupported vector store interface - no compatible add method found")
            return None
        except VECTOR_STORE_RECOVERABLE_ERRORS:
            logger.exception("VectorStoreAdapter.add failed")
            return None  # Graceful degradation instead of raising

    def query(self, query_embedding, top_k=5):
        if hasattr(self.store, "query"):
            try:
                return self.store.query(query_embedding, top_k=top_k)
            except VECTOR_STORE_RECOVERABLE_ERRORS as e:
                logger.error(f"Vector store query failed: {e}")
                return []
        if hasattr(self.store, "search"):
            try:
                return self.store.search(query_embedding, top_k)
            except VECTOR_STORE_RECOVERABLE_ERRORS as e:
                logger.error(f"Vector store search failed: {e}")
                return []
        logger.error("Unsupported vector store query API")
        return []  # Return empty results instead of crashing
