# source_handlers
'''
Actual handling of tools & data resources via B4A & MCP
'''
# import os
# from typing import List, Dict, Any, Optional, Tuple, Union
from contextlib import asynccontextmanager

import structlog
import httpx

# Attempt to import feedparser, needed for @rss type
try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False

from b4a_config import B4AConfig, B4AConfigError, MCPConfig, MCPConfigError, RSSConfig

logger = structlog.get_logger(__name__)


class RssSource:
    def __init__(self, config: RSSConfig):
        if not FEEDPARSER_AVAILABLE:
            raise ImportError('feedparser is not available. Please install it (`pip install feedparser`) to use RSS sources.')
        self.url = config.url

    async def _initialize(self):
        self.resource_ready = True

    @classmethod
    async def async_init(cls):
        instance = cls()
        await instance._initialize()
        return instance

    @classmethod
    @asynccontextmanager
    async def acquire(cls, *args, **kwargs):
        resource = await cls.async_init(*args, **kwargs)
        try:
            yield resource
        finally:
            # Any cleanup tasks (e.g., closing a connection)
            pass  # None for now

    async def retrieve(self):
        '''Retrieve the RSS feed data'''
        # Pull feed with httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(self.url)
            if response.status_code != 200:
                raise ValueError(f'Failed to retrieve RSS feed from {self.url}')
            logger.debug('Retrieved RSS feed', url=self.url)
        self.feed = feedparser.parse(response.text)  # Cache the feedparser object
        if not self.feed.bozo:
            raise ValueError(f'Failed to parse RSS feed from {self.url}')
        logger.debug('Parsed RSS feed', url=self.url)
