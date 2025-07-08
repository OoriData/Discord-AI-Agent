# SPDX-FileCopyrightText: 2025-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
'''
Shared test fixtures and configuration for pytest.
'''

import pytest
import structlog
from unittest.mock import Mock
from typing import Dict, Any, List


@pytest.fixture(scope='session', autouse=True)
def configure_logging():
    '''Configure structlog for testing.'''
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt='iso'),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


@pytest.fixture
def mock_rss_feed_cache():
    '''Mock RSS feed cache for testing.'''
    return {}


@pytest.fixture
def mock_rss_cache_ttl():
    '''Mock RSS cache TTL for testing.'''
    return 300


@pytest.fixture
def sample_rss_xml():
    '''Sample RSS XML content based on LocalLLaMA subreddit.'''
    return '''<?xml version='1.0' encoding='UTF-8'?>
<rss version='2.0' xmlns:content='http://purl.org/rss/1.0/modules/content/'>
  <channel>
    <title>LocalLLaMA</title>
    <link>https://www.reddit.com/r/LocalLLaMA/</link>
    <description>Discussion about open source large language models</description>
    <item>
      <title><![CDATA[LocalLLaMA: Open Source AI Models]]></title>
      <link>https://www.reddit.com/r/LocalLLaMA/comments/example1/</link>
      <guid>https://www.reddit.com/r/LocalLLaMA/comments/example1/</guid>
      <pubDate>Mon, 01 Jan 2024 12:00:00 GMT</pubDate>
      <description><![CDATA[<p>Discussion about <strong>open source</strong> large language models. Learn about <em>Llama</em>, <code>Mistral</code>, and other <a href='#'>local AI models</a> for running on your own hardware.</p>]]></description>
    </item>
    <item>
      <title><![CDATA[Fine-tuning LLMs for Specific Tasks]]></title>
      <link>https://www.reddit.com/r/LocalLLaMA/comments/example2/</link>
      <guid>https://www.reddit.com/r/LocalLLaMA/comments/example2/</guid>
      <pubDate>Mon, 01 Jan 2024 13:00:00 GMT</pubDate>
      <description><![CDATA[<p>Guide to <b>fine-tuning</b> language models for specific domains. We'll explore <i>LoRA</i>, <code>QLoRA</code>, and <a href='#'>parameter efficient training</a> techniques.</p>]]></description>
    </item>
    <item>
      <title><![CDATA[Hardware Requirements for Local LLMs]]></title>
      <link>https://www.reddit.com/r/LocalLLaMA/comments/example3/</link>
      <guid>https://www.reddit.com/r/LocalLLaMA/comments/example3/</guid>
      <pubDate>Mon, 01 Jan 2024 14:00:00 GMT</pubDate>
      <description><![CDATA[<p>Understanding <strong>hardware requirements</strong> for running large language models locally. Topics include <em>GPU memory</em>, <code>quantization</code>, and <a href='#'>optimization strategies</a>.</p>]]></description>
    </item>
  </channel>
</rss>'''


@pytest.fixture
def mock_httpx_response():
    '''Mock httpx response for testing.'''
    mock_response = Mock()
    mock_response.content = b'<rss><channel><item><title>Test</title></item></channel></rss>'
    mock_response.raise_for_status.return_value = None
    return mock_response


@pytest.fixture
def mock_feedparser_result():
    '''Mock feedparser result for testing.'''
    mock_feed = Mock()
    mock_feed.entries = [
        Mock(
            title='Test Entry 1',
            summary='Test summary 1',
            link='https://example.com/1',
            published='Mon, 01 Jan 2024 12:00:00 GMT',
            published_parsed=None
        ),
        Mock(
            title='Test Entry 2', 
            summary='Test summary 2',
            link='https://example.com/2',
            published='Mon, 01 Jan 2024 13:00:00 GMT',
            published_parsed=None
        )
    ]
    mock_feed.bozo = False
    mock_feed.bozo_exception = None
    return mock_feed


@pytest.fixture
def mock_sentence_transformer():
    '''Mock SentenceTransformer for testing.'''
    mock_model = Mock()
    mock_model.encode.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    return mock_model


@pytest.fixture
def mock_chonkie_chunker():
    '''Mock Chonkie chunker for testing.'''
    mock_chunker = Mock()
    mock_chunk1 = Mock()
    mock_chunk1.text = 'This is sentence one.'
    mock_chunk2 = Mock()
    mock_chunk2.text = 'This is sentence two.'
    mock_chunker.chunk.return_value = [mock_chunk1, mock_chunk2]
    return mock_chunker


@pytest.fixture
def mock_html2text_converter():
    '''Mock html2text converter for testing.'''
    mock_converter = Mock()
    mock_converter.handle.return_value = '# Test Title\n\nThis is **bold** and *italic* text.'
    return mock_converter 