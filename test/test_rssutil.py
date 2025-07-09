# SPDX-FileCopyrightText: 2025-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
'''
Test suite for rssutil module - RSS feed querying and search functionality.
'''

import pytest
import time
import email.utils
from unittest.mock import Mock, patch, AsyncMock
from typing import List

from discord_aiagent.rssutil import RSSAgent


class MockRSSConfig:
    '''Mock RSS configuration for testing.'''
    
    def __init__(self, name: str, url: str, cosine_similarity_threshold: float = 0.65, top_k: int = 5):
        self.name = name
        self.url = url
        self.cosine_similarity_threshold = cosine_similarity_threshold
        self.top_k = top_k


class MockRSSEntry:
    '''Mock RSS entry for testing.'''
    
    def __init__(self, title: str, summary: str, link: str = 'http://example.com', published: str = None):
        self.title = title
        self.summary = summary
        self.link = link
        self.published = published
        self.published_parsed = None
        if published:
            try:
                parsed_date = email.utils.parsedate_tz(published)
                if parsed_date:
                    self.published_parsed = parsed_date
            except:
                pass

    def get(self, key: str, default: str = ''):
        return getattr(self, key, default)


class MockParsedFeed:
    '''Mock parsed RSS feed for testing.'''
    
    def __init__(self, entries: List[MockRSSEntry], bozo: bool = False, bozo_exception: str = None):
        self.entries = entries
        self.bozo = bozo
        self.bozo_exception = bozo_exception


@pytest.fixture
def sample_rss_entries():
    '''Sample RSS entries for testing.'''
    return [
        MockRSSEntry(
            title='<h1>LocalLLaMA: Open Source AI Models</h1>',
            summary='<p>Discussion about <strong>open source</strong> large language models. Learn about <em>Llama</em>, <code>Mistral</code>, and other <a href="#">local AI models</a> for running on your own hardware.</p>',
            link='https://www.reddit.com/r/LocalLLaMA/comments/example1/',
            published='Mon, 01 Jan 2024 12:00:00 GMT'
        ),
        MockRSSEntry(
            title='<h1>Fine-tuning LLMs for Specific Tasks</h1>',
            summary='<p>Guide to <b>fine-tuning</b> language models for specific domains. We\'ll explore <i>LoRA</i>, <code>QLoRA</code>, and <a href="#">parameter efficient training</a> techniques.</p>',
            link='https://www.reddit.com/r/LocalLLaMA/comments/example2/',
            published='Mon, 01 Jan 2024 13:00:00 GMT'
        ),
        MockRSSEntry(
            title='<h1>Hardware Requirements for Local LLMs</h1>',
            summary='<p>Understanding <strong>hardware requirements</strong> for running large language models locally. Topics include <em>GPU memory</em>, <code>quantization</code>, and <a href="#">optimization strategies</a>.</p>',
            link='https://www.reddit.com/r/LocalLLaMA/comments/example3/',
            published='Mon, 01 Jan 2024 14:00:00 GMT'
        )
    ]


@pytest.fixture
def mock_b4a_data():
    '''Mock B4A data configuration for testing.'''
    mock_data = Mock()
    mock_data.rss_sources = [
        MockRSSConfig('local_llama', 'https://www.reddit.com/r/LocalLLaMA.rss', 0.65, 5),
        MockRSSConfig('ai_news', 'https://example.com/ai-news.rss', 0.7, 3),
    ]
    return mock_data


@pytest.fixture
def rss_agent(mock_b4a_data):
    '''RSS agent fixture for testing.'''
    rss_feed_cache = {}
    rss_cache_ttl = 300
    return RSSAgent(mock_b4a_data, rss_feed_cache, rss_cache_ttl)


class TestRSSAgent:
    '''Test the RSSAgent class.'''

    def test_init(self, mock_b4a_data):
        '''Test RSSAgent initialization.'''
        rss_feed_cache = {}
        rss_cache_ttl = 300
        agent = RSSAgent(mock_b4a_data, rss_feed_cache, rss_cache_ttl)
        
        assert agent.b4a_data == mock_b4a_data
        assert agent._rss_feed_cache == rss_feed_cache
        assert agent._rss_cache_ttl == rss_cache_ttl
        assert agent._search_engine is not None

    @pytest.mark.asyncio
    async def test_handle_tool_unknown_tool(self, rss_agent):
        '''Test handling of unknown tool.'''
        result = await rss_agent.handle_tool('unknown_tool', {})
        
        assert 'error' in result
        assert 'Unknown tool name' in result['error']

    @pytest.mark.asyncio
    async def test_execute_rss_query_missing_feed_name(self, rss_agent):
        '''Test RSS query execution with missing feed name.'''
        result = await rss_agent.execute_rss_query({})
        
        assert 'error' in result
        assert 'Missing required parameter: feed_name' in result['error']

    @pytest.mark.asyncio
    async def test_execute_rss_query_feed_not_found(self, rss_agent):
        '''Test RSS query execution with non-existent feed.'''
        result = await rss_agent.execute_rss_query({'feed_name': 'nonexistent_feed'})
        
        assert 'error' in result
        assert 'not found' in result['error']

    @pytest.mark.asyncio
    async def test_execute_rss_query_invalid_limit(self, rss_agent):
        '''Test RSS query execution with invalid limit.'''
        result = await rss_agent.execute_rss_query({
            'feed_name': 'local_llama',
            'limit': 'invalid'
        })
        
        # Should default to 5
        assert 'error' not in result or 'limit' not in result['error']

    @pytest.mark.asyncio
    async def test_execute_rss_query_success_no_query(self, rss_agent, sample_rss_entries):
        '''Test successful RSS query execution without search query.'''
        mock_feed = MockParsedFeed(sample_rss_entries)
        
        with patch('discord_aiagent.rssutil.fetch_and_parse_rss', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = (mock_feed, None)
            
            result = await rss_agent.execute_rss_query({
                'feed_name': 'local_llama',
                'limit': 3
            })
            
            assert 'result' in result
            assert 'Found 3 entries from feed `local_llama`' in result['result']
            # Check for the actual title from the mock entry
            assert 'LocalLLaMA: Open Source AI Models' in result['result']

    @pytest.mark.asyncio
    async def test_execute_rss_query_with_search(self, rss_agent, sample_rss_entries):
        '''Test RSS query execution with search query.'''
        mock_feed = MockParsedFeed(sample_rss_entries)
        
        with patch('discord_aiagent.rssutil.fetch_and_parse_rss', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = (mock_feed, None)
            
            with patch.object(rss_agent._search_engine, 'vector_search') as mock_vector_search:
                mock_vector_search.return_value = []
                
                with patch.object(rss_agent._search_engine, 'keyword_search') as mock_keyword_search:
                    mock_keyword_search.return_value = [
                        Mock(entry=sample_rss_entries[0], score=1.0, best_chunk_text='test', best_chunk_index=0)
                    ]
                    
                    result = await rss_agent.execute_rss_query({
                        'feed_name': 'local_llama',
                        'query': 'open source',
                        'limit': 3
                    })
                    
                    assert 'result' in result
                    assert 'matching query `open source`' in result['result']

    @pytest.mark.asyncio
    async def test_execute_rss_query_fetch_error(self, rss_agent):
        '''Test RSS query execution when fetch fails.'''
        with patch('discord_aiagent.rssutil.fetch_and_parse_rss', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = (None, 'Network error')
            
            result = await rss_agent.execute_rss_query({
                'feed_name': 'local_llama'
            })
            
            assert 'error' in result
            assert 'Error fetching feed `local_llama`' in result['error']

    @pytest.mark.asyncio
    async def test_execute_rss_query_no_entries(self, rss_agent):
        '''Test RSS query execution when feed has no entries.'''
        mock_feed = MockParsedFeed([])
        
        with patch('discord_aiagent.rssutil.fetch_and_parse_rss', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = (mock_feed, None)
            
            result = await rss_agent.execute_rss_query({
                'feed_name': 'local_llama'
            })
            
            assert 'error' in result
            assert 'Failed to get valid entries from feed' in result['error']

    def test_filter_entries_by_timestamp_no_timestamp(self, rss_agent, sample_rss_entries):
        '''Test filtering entries when no timestamp is provided.'''
        result = rss_agent._filter_entries_by_timestamp(sample_rss_entries, None)
        
        assert result == sample_rss_entries

    def test_filter_entries_by_timestamp_valid_timestamp(self, rss_agent, sample_rss_entries):
        '''Test filtering entries with valid timestamp.'''
        # Set a timestamp that would filter out some entries
        parsed_date = email.utils.parsedate_tz('Mon, 01 Jan 2024 13:30:00 GMT')
        since_timestamp = email.utils.mktime_tz(parsed_date)
        
        result = rss_agent._filter_entries_by_timestamp(sample_rss_entries, since_timestamp)
        
        # Should filter out entries published before 13:30
        assert len(result) < len(sample_rss_entries)

    def test_filter_entries_by_timestamp_invalid_timestamp(self, rss_agent, sample_rss_entries):
        '''Test filtering entries with invalid timestamp.'''
        result = rss_agent._filter_entries_by_timestamp(sample_rss_entries, 'invalid_timestamp')
        
        assert result is None

    def test_perform_search_vector_search_success(self, rss_agent, sample_rss_entries):
        '''Test search performance with successful vector search.'''
        with patch.object(rss_agent._search_engine, 'vector_search') as mock_vector_search:
            mock_vector_search.return_value = [
                Mock(entry=sample_rss_entries[0], score=0.8, best_chunk_text='test', best_chunk_index=0)
            ]
            
            results = rss_agent._perform_search(sample_rss_entries, 'test query', 0.65, 5)
            
            assert len(results) == 1
            mock_vector_search.assert_called_once()

    def test_perform_search_vector_search_fallback(self, rss_agent, sample_rss_entries):
        '''Test search performance with vector search fallback to keyword search.'''
        with patch.object(rss_agent._search_engine, 'vector_search') as mock_vector_search:
            mock_vector_search.return_value = []
            
            with patch.object(rss_agent._search_engine, 'keyword_search') as mock_keyword_search:
                mock_keyword_search.return_value = [
                    Mock(entry=sample_rss_entries[0], score=1.0, best_chunk_text='test', best_chunk_index=0)
                ]
                
                results = rss_agent._perform_search(sample_rss_entries, 'test query', 0.65, 5)
                
                assert len(results) == 1
                mock_vector_search.assert_called_once()
                mock_keyword_search.assert_called_once()

    def test_format_search_results_with_search_data(self, rss_agent, sample_rss_entries):
        '''Test formatting search results with search metadata.'''
        search_results = [
            Mock(
                entry=sample_rss_entries[0],
                score=0.85,
                best_chunk_text='Best matching chunk text',
                best_chunk_index=0
            )
        ]
        
        result_items = rss_agent._format_search_results([sample_rss_entries[0]], search_results, 'test query')
        
        assert len(result_items) == 1
        assert 'Score: 0.850' in result_items[0]
        assert 'Best Match: Best matching chunk text' in result_items[0]

    def test_format_search_results_without_search_data(self, rss_agent, sample_rss_entries):
        '''Test formatting search results without search metadata.'''
        result_items = rss_agent._format_search_results([sample_rss_entries[0]], [], None)
        
        assert len(result_items) == 1
        assert 'Title: <h1>LocalLLaMA: Open Source AI Models</h1>' in result_items[0]
        assert 'Score:' not in result_items[0]
        assert 'Best Match:' not in result_items[0]


@pytest.mark.integration
class TestIntegration:
    '''Integration tests for RSS functionality.'''

    @pytest.mark.asyncio
    async def test_full_rss_pipeline(self, mock_b4a_data, sample_rss_entries):
        '''Test the full RSS pipeline from configuration to results.'''
        rss_feed_cache = {}
        rss_cache_ttl = 300
        agent = RSSAgent(mock_b4a_data, rss_feed_cache, rss_cache_ttl)
        
        mock_feed = MockParsedFeed(sample_rss_entries)
        
        with patch('discord_aiagent.rssutil.fetch_and_parse_rss', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = (mock_feed, None)
            
            result = await agent.execute_rss_query({
                'feed_name': 'local_llama',
                'query': 'open source',
                'limit': 3
            })
            
            assert 'result' in result
            assert 'Found' in result['result']
            assert 'local_llama' in result['result']

    @pytest.mark.asyncio
    async def test_rss_caching_behavior(self, mock_b4a_data, sample_rss_entries):
        '''Test RSS caching behavior.'''
        rss_feed_cache = {}
        rss_cache_ttl = 300
        agent = RSSAgent(mock_b4a_data, rss_feed_cache, rss_cache_ttl)
        
        mock_feed = MockParsedFeed(sample_rss_entries)
        
        # Mock the HTTP client to avoid actual network calls while preserving cache logic
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            
            # Create a mock response with sample RSS XML
            mock_response = Mock()
            mock_response.content = b'<rss><channel><item><title>Test</title></item></channel></rss>'
            mock_response.raise_for_status.return_value = None
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            # Mock feedparser to return our test feed
            with patch('discord_aiagent.source_handlers.feedparser') as mock_feedparser:
                mock_feedparser.parse.return_value = mock_feed
                
                # First call should fetch
                await agent.execute_rss_query({'feed_name': 'local_llama'})
                assert mock_client.get.call_count == 1
                
                # Second call should use cache
                await agent.execute_rss_query({'feed_name': 'local_llama'})
                assert mock_client.get.call_count == 1  # Should not increase

    def test_timestamp_filtering_integration(self, mock_b4a_data, sample_rss_entries):
        '''Test timestamp filtering integration.'''
        rss_feed_cache = {}
        rss_cache_ttl = 300
        agent = RSSAgent(mock_b4a_data, rss_feed_cache, rss_cache_ttl)
        
        # Test with timestamp that should filter some entries
        parsed_date = email.utils.parsedate_tz('Mon, 01 Jan 2024 13:30:00 GMT')
        since_timestamp = email.utils.mktime_tz(parsed_date)
        
        filtered_entries = agent._filter_entries_by_timestamp(sample_rss_entries, since_timestamp)
        
        # Should have fewer entries after filtering
        assert len(filtered_entries) < len(sample_rss_entries)
        
        # All remaining entries should be from after the timestamp
        for entry in filtered_entries:
            if entry.published_parsed:
                entry_time = email.utils.mktime_tz(entry.published_parsed)
                assert entry_time >= since_timestamp


@pytest.mark.parametrize('feed_name,expected_config', [
    ('local_llama', MockRSSConfig('local_llama', 'https://www.reddit.com/r/LocalLLaMA.rss', 0.65, 5)),
    ('ai_news', MockRSSConfig('ai_news', 'https://example.com/ai-news.rss', 0.7, 3)),
])
def test_rss_config_lookup(mock_b4a_data, feed_name, expected_config):
    '''Test RSS configuration lookup by name.'''
    rss_feed_cache = {}
    rss_cache_ttl = 300
    agent = RSSAgent(mock_b4a_data, rss_feed_cache, rss_cache_ttl)
    
    found_config = next((conf for conf in agent.b4a_data.rss_sources if conf.name == feed_name), None)
    
    assert found_config is not None
    assert found_config.name == expected_config.name
    assert found_config.url == expected_config.url
    assert found_config.cosine_similarity_threshold == expected_config.cosine_similarity_threshold
    assert found_config.top_k == expected_config.top_k 