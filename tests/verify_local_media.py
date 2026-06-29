import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from skills._local_media_generation import LocalMediaGenerationSkill


async def test_local_generation():
    print("--- Testing Local Media Generation ---")
    
    skill = LocalMediaGenerationSkill()
    
    prompt = "A futuristic city with flying cars, cyberpunk style, intense colors."
    print(f"Generating image for: '{prompt}'")
    
    result = await skill.execute({"objective": prompt}, {})

    assert result["ok"], result
    image_path = Path(result["path"])
    assert await asyncio.to_thread(image_path.exists), result
    content = await asyncio.to_thread(image_path.read_bytes)
    assert content.startswith(b"\x89PNG\r\n\x1a\n")
    print("✓ Generation successful!")
    print(f"  Mode: {result.get('generation_mode')}")
    print(f"  URL: {result['url']}")
    print(f"  Path: {result['path']}")

if __name__ == "__main__":
    asyncio.run(test_local_generation())
