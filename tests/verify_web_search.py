import asyncio
import logging
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.capabilities.phantom_browser import PhantomBrowser

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("VerifyWebSearch")

WEB_SEARCH_VERIFY_ERRORS = (
    AttributeError,
    ImportError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)

async def test_browser():
    logger.info("Starting browser verification...")
    browser = PhantomBrowser(visible=False)
    ok = True
    
    try:
        # 1. Navigation
        logger.info("Testing navigation to Wikipedia...")
        success = await browser.browse("https://en.wikipedia.org/wiki/Main_Page")
        if not success:
            logger.error("Navigation failed!")
            return False
            
        # 2. Reading Content
        logger.info("Testing content extraction...")
        content = await browser.read_content()
        logger.info(f"Extracted content length: {len(content)}")
        if len(content) < 100:
            logger.error("Content extraction seems thin!")
            ok = False
        else:
            logger.info("Content extraction successful.")

        # 3. Getting Links
        logger.info("Testing link extraction...")
        links = await browser.get_links()
        logger.info(f"Found {len(links)} links.")
        if not links:
            logger.error("No links found!")
            ok = False
        
        # 4. Scrolling
        logger.info("Testing scrolling...")
        await browser.scroll(direction="down", amount=1000)
        logger.info("Scroll command executed.")

        # 5. Clicking
        logger.info("Testing clicking 'Random article'...")
        # Find the random article link. In Wikipedia it's usually in the sidebar.
        # But let's try get_by_text since we have it.
        click_success = await browser.click(selector=None, text_match="Random article")
        if click_success:
            logger.info("Successfully clicked 'Random article'.")
            await asyncio.sleep(2) # Wait for page load
            new_title = await browser.page.title()
            logger.info(f"New page title: {new_title}")
        else:
            logger.warning("Failed to click 'Random article'.")
            ok = False

    except WEB_SEARCH_VERIFY_ERRORS as e:
        logger.error(f"Verification encountered an error: {e}")
        ok = False
    finally:
        await browser.close()
        logger.info("Browser closed.")
    return ok

if __name__ == "__main__":
    sys.exit(0 if asyncio.run(test_browser()) else 1)
