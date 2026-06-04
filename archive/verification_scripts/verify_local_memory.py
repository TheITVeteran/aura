import asyncio
import logging
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory_vector import VectorMemory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Verify.Memory")

async def verify_memory():
    logger.info("🔍 VERIFYING LOCAL MEMORY INTEGRATION...")
    
    # Initialize Memory with local_first=True
    memory = VectorMemory(path="data/test_memory.json", local_first=True)
    
    # Test text
    test_text = "Aura is a sovereign autonomous agent running on local hardware."
    
    logger.info(f"Adding memory: '{test_text}'")
    success = memory.add(test_text, metadata={"test": True})
    
    if success:
        logger.info("✅ Memory addition SUCCESSFUL (Local Embeddings).")
    else:
        logger.error("❌ Memory addition FAILED.")
        return

    # Test search
    logger.info("Searching for: 'sovereign agent'")
    results = memory.search("sovereign agent", k=1)
    
    if results and test_text in results[0]["text"]:
        logger.info(f"✅ Search SUCCESSFUL. Found: '{results[0]['text']}'")
        logger.info(f"   Similarity: {results[0]['similarity']:.4f}")
    else:
        logger.error("❌ Search FAILED or returned no results.")

    # Cleanup
    if os.path.exists("data/test_memory.json"):
        os.remove("data/test_memory.json")
        logger.info("Cleaned up test data.")

if __name__ == "__main__":
    asyncio.run(verify_memory())
