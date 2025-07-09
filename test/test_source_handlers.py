# SPDX-FileCopyrightText: 2025-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
'''
Test suite for source_handlers module - RSS fetching and parsing.
'''

import pytest
import httpx
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any

from discord_aiagent.source_handlers import fetch_and_parse_rss


class TestFetchAndParseRSS:
    '''Test the fetch_and_parse_rss function.'''

    @pytest.mark.asyncio
    async def test_fetch_and_parse_rss_success(self, mock_httpx_response, mock_feedparser_result):
        '''Test successful RSS fetching and parsing.'''
        url = 'https://www.reddit.com/r/LocalLLaMA.rss'
        cache = {}
        cache_ttl = 300
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.get.return_value = mock_httpx_response
            mock_client_class.return_value = mock_client
            
            with patch('discord_aiagent.source_handlers.feedparser') as mock_feedparser:
                mock_feedparser.parse.return_value = mock_feedparser_result
                
                result, error = await fetch_and_parse_rss(url, cache, cache_ttl)
                
                assert result == mock_feedparser_result
                assert error is None
                assert url in cache
                assert len(cache[url]) == 2  # (timestamp, feed_data)

    @pytest.mark.asyncio
    async def test_fetch_and_parse_rss_cache_hit(self, mock_feedparser_result):
        '''Test RSS fetching with cache hit.'''
        url = 'https://www.reddit.com/r/LocalLLaMA.rss'
        cache = {url: (1234567890.0, mock_feedparser_result)}
        cache_ttl = 300
        
        # Mock the current time to ensure cache is valid
        with patch('discord_aiagent.source_handlers.time.time', return_value=1234567890.0 + 100):
            result, error = await fetch_and_parse_rss(url, cache, cache_ttl)
            
            assert result is mock_feedparser_result
            assert error is None

    @pytest.mark.asyncio
    async def test_fetch_and_parse_rss_cache_expired(self, mock_httpx_response, mock_feedparser_result):
        '''Test RSS fetching with expired cache.'''
        url = 'https://www.reddit.com/r/LocalLLaMA.rss'
        cache = {url: (0.0, mock_feedparser_result)}  # Very old timestamp
        cache_ttl = 300
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.get.return_value = mock_httpx_response
            mock_client_class.return_value = mock_client
            
            with patch('discord_aiagent.source_handlers.feedparser') as mock_feedparser:
                mock_feedparser.parse.return_value = mock_feedparser_result
                
                result, error = await fetch_and_parse_rss(url, cache, cache_ttl)
                
                assert result == mock_feedparser_result
                assert error is None

    @pytest.mark.asyncio
    async def test_fetch_and_parse_rss_timeout_error(self):
        '''Test RSS fetching with timeout error.'''
        url = 'https://www.reddit.com/r/LocalLLaMA.rss'
        cache = {}
        cache_ttl = 300
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.get.side_effect = httpx.TimeoutException('Request timed out')
            mock_client_class.return_value = mock_client
            
            result, error = await fetch_and_parse_rss(url, cache, cache_ttl)
            
            assert result is None
            assert 'Timeout' in error

    @pytest.mark.asyncio
    async def test_fetch_and_parse_rss_request_error(self):
        '''Test RSS fetching with request error.'''
        url = 'https://www.reddit.com/r/LocalLLaMA.rss'
        cache = {}
        cache_ttl = 300
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.get.side_effect = httpx.RequestError('Network error')
            mock_client_class.return_value = mock_client
            
            result, error = await fetch_and_parse_rss(url, cache, cache_ttl)
            
            assert result is None
            assert 'Network error' in error

    @pytest.mark.asyncio
    async def test_fetch_and_parse_rss_http_error(self, mock_httpx_response):
        '''Test RSS fetching with HTTP error.'''
        url = 'https://www.reddit.com/r/LocalLLaMA.rss'
        cache = {}
        cache_ttl = 300
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.get.return_value = mock_httpx_response
            mock_httpx_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                '404 Not Found', request=Mock(), response=mock_httpx_response
            )
            mock_client_class.return_value = mock_client
            
            result, error = await fetch_and_parse_rss(url, cache, cache_ttl)
            
            assert result is None
            assert 'unexpected error' in error

    @pytest.mark.asyncio
    async def test_fetch_and_parse_rss_bozo_feed(self, mock_httpx_response):
        '''Test RSS fetching with malformed feed (bozo flag).'''
        url = 'https://www.reddit.com/r/LocalLLaMA.rss'
        cache = {}
        cache_ttl = 300
        
        mock_feed = Mock()
        mock_feed.entries = [Mock()]
        mock_feed.bozo = True
        mock_feed.bozo_exception = 'Malformed XML'
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.get.return_value = mock_httpx_response
            mock_client_class.return_value = mock_client
            
            with patch('discord_aiagent.source_handlers.feedparser') as mock_feedparser:
                mock_feedparser.parse.return_value = mock_feed
                
                result, error = await fetch_and_parse_rss(url, cache, cache_ttl)
                
                # Should still return the feed despite bozo flag
                assert result == mock_feed
                assert error is None

    @pytest.mark.asyncio
    async def test_fetch_and_parse_rss_no_entries(self, mock_httpx_response):
        '''Test RSS fetching with feed that has no entries.'''
        url = 'https://www.reddit.com/r/LocalLLaMA.rss'
        cache = {}
        cache_ttl = 300
        
        class FeedWithEmptyEntries:
            entries = []
            bozo = False
        mock_feed = FeedWithEmptyEntries()
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.get.return_value = mock_httpx_response
            mock_client_class.return_value = mock_client
            
            with patch('discord_aiagent.source_handlers.feedparser') as mock_feedparser:
                mock_feedparser.parse.return_value = mock_feed
                
                result, error = await fetch_and_parse_rss(url, cache, cache_ttl)
                
                assert result is None
                assert 'Could not find any entries' in error

    @pytest.mark.asyncio
    async def test_fetch_and_parse_rss_missing_entries_attribute(self, mock_httpx_response):
        '''Test RSS fetching with feed missing entries attribute.'''
        url = 'https://www.reddit.com/r/LocalLLaMA.rss'
        cache = {}
        cache_ttl = 300
        
        class FeedWithoutEntries:
            bozo = False
        mock_feed = FeedWithoutEntries()
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.get.return_value = mock_httpx_response
            mock_client_class.return_value = mock_client
            
            with patch('discord_aiagent.source_handlers.feedparser') as mock_feedparser:
                mock_feedparser.parse.return_value = mock_feed
                
                result, error = await fetch_and_parse_rss(url, cache, cache_ttl)
                
                assert result is None
                assert 'An unexpected error occurred while parsing the feed from' in error

    @pytest.mark.asyncio
    async def test_fetch_and_parse_rss_parsing_error(self, mock_httpx_response):
        '''Test RSS fetching with parsing error.'''
        url = 'https://www.reddit.com/r/LocalLLaMA.rss'
        cache = {}
        cache_ttl = 300
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.get.return_value = mock_httpx_response
            mock_client_class.return_value = mock_client
            
            with patch('discord_aiagent.source_handlers.feedparser') as mock_feedparser:
                mock_feedparser.parse.side_effect = Exception('Parsing failed')
                
                result, error = await fetch_and_parse_rss(url, cache, cache_ttl)
                
                assert result is None
                assert 'unexpected error' in error

    @pytest.mark.asyncio
    async def test_fetch_and_parse_rss_feedparser_not_available(self):
        '''Test RSS fetching when feedparser is not available.'''
        url = 'https://www.reddit.com/r/LocalLLaMA.rss'
        cache = {}
        cache_ttl = 300
        
        with patch('discord_aiagent.source_handlers.FEEDPARSER_AVAILABLE', False):
            result, error = await fetch_and_parse_rss(url, cache, cache_ttl)
            
            assert result is None
            assert 'feedparser' in error


@pytest.mark.integration
class TestIntegration:
    '''Integration tests for RSS fetching.'''

    @pytest.mark.asyncio
    async def test_full_rss_fetch_pipeline(self, sample_rss_xml):
        '''Test the complete RSS fetching pipeline.'''
        url = 'https://www.reddit.com/r/LocalLLaMA.rss'
        cache = {}
        cache_ttl = 300
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            
            # Create a mock response with the sample XML
            mock_response = Mock()
            mock_response.content = sample_rss_xml.encode('utf-8')
            mock_response.raise_for_status.return_value = None
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            with patch('discord_aiagent.source_handlers.feedparser') as mock_feedparser:
                # Create a mock feedparser result
                mock_feed = Mock()
                mock_feed.entries = [Mock(), Mock(), Mock()]
                mock_feed.bozo = False
                mock_feedparser.parse.return_value = mock_feed
                
                result, error = await fetch_and_parse_rss(url, cache, cache_ttl)
                
                assert result == mock_feed
                assert error is None
                assert url in cache 