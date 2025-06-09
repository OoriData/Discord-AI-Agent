# source_handlers.py
'''
Actual handling of tools & data resources via B4A & MCP
'''
import time
from typing import Optional, Any

import structlog
import httpx

# Attempt to import feedparser, needed for @rss type
try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False
    feedparser = None  # Define feedparser as None if not available

# from b4a_config import B4AConfig, B4AConfigError, MCPConfig, MCPConfigError, RSSConfig

logger = structlog.get_logger(__name__)

# Simple in-memory cache for RSS feeds (URL -> (timestamp, feed_data))
# Managed within the cog instance that uses this function
# _rss_feed_cache = {}
# _rss_cache_ttl = 300  # Cache feeds for 5 minutes (300 seconds)

async def fetch_and_parse_rss(
    url: str,
    cache: dict[str, tuple[float, Any]],  # Pass cache dict from cog
    cache_ttl: int  # Pass TTL from cog
) -> tuple[Any | None, str | None]:
    '''
    Fetches and parses an RSS feed URL with simple time-based caching.

    Args:
        url: The URL of the RSS feed.
        cache: The dictionary used for caching.
        cache_ttl: The time-to-live for cache entries in seconds.

    Returns:
        A tuple containing (parsed_feed, error_message).
        If successful, parsed_feed is the feedparser object, error_message is None.
        If failed, parsed_feed is None, error_message contains the error details.
    '''
    if not FEEDPARSER_AVAILABLE:
        return None, 'RSS processing requires the `feedparser` library, which is not installed.'

    current_time = time.time()

    # Check cache
    if url in cache:
        timestamp, cached_feed = cache[url]
        if current_time - timestamp < cache_ttl:
            logger.debug('Returning cached RSS feed', url=url)
            return cached_feed, None  # Return cached data, no error

    # Fetch feed with httpx
    logger.debug('Fetching RSS feed', url=url)
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
            response = await client.get(url)
            response.raise_for_status()  # Raise HTTPStatusError for bad responses (4xx or 5xx)
    except httpx.TimeoutException:
        logger.warning('Timeout fetching RSS feed', url=url)
        return None, f'Timeout trying to fetch the RSS feed from {url}.'
    except httpx.RequestError as e:
        logger.warning('Error fetching RSS feed', url=url, error=str(e))
        return None, f'Network error fetching RSS feed from {url}: {e.__class__.__name__}.'
    except Exception as e:
        logger.exception('Unexpected error during RSS feed fetch', url=url)
        return None, f'An unexpected error occurred while fetching {url}.'

    # Parse feed
    try:
        # Use feedparser.parse on the response content (bytes)
        # feedparser handles encoding detection reasonably well
        parsed_feed = feedparser.parse(response.content)

        # Check for bozo flag (indicates potential parsing issues)
        if parsed_feed.bozo:
            bozo_exception = parsed_feed.get('bozo_exception', 'Unknown parsing issue')
            # Log warning but proceed, sometimes feeds are slightly malformed but usable
            logger.warning('RSS feed potentially malformed (bozo flag set)', url=url, exception=str(bozo_exception))
            # You could choose to return an error here if strict parsing is required:
            # return None, f'Failed to properly parse RSS feed from {url}: {bozo_exception}'

        # Basic check if entries exist (or if it's a valid feed structure)
        if not hasattr(parsed_feed, 'entries'):
             logger.warning('Parsed RSS feed missing "entries" attribute', url=url)
             return None, f'Could not find any entries in the feed from {url}. It might be empty or not a valid feed.'

        logger.debug('Parsed RSS feed successfully', url=url, entries_found=len(parsed_feed.entries))

        # Update cache
        cache[url] = (current_time, parsed_feed)
        return parsed_feed, None  # Return parsed data, no error

    except Exception as e:
        logger.exception('Unexpected error during RSS feed parsing', url=url)
        return None, f'An unexpected error occurred while parsing the feed from {url}.'
