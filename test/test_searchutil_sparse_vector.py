# SPDX-FileCopyrightText: 2025-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
'''
Test suite for searchutil module - vector search, chunking, and HTML conversion.
'''

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import List, Any

from discord_aiagent.searchutil import (
    VectorSearchEngine, 
    SearchChunk, 
    SearchResult, 
    get_vector_search_engine
)


class MockRSSEntry:
    '''Mock RSS entry for testing.'''
    def __init__(self, title: str, summary: str, link: str = 'http://example.com'):
        self.title = title
        self.summary = summary
        self.link = link
    
    def get(self, key: str, default: str = ''):
        return getattr(self, key, default)


@pytest.fixture
def sample_rss_entries():
    '''Sample RSS entries with HTML content for testing.'''
    return [
        MockRSSEntry(
            title='<h1>LocalLLaMA: Open Source AI Models</h1>',
            summary='<p>Discussion about <strong>open source</strong> large language models. Learn about <em>Llama</em>, <code>Mistral</code>, and other <a href="#">local AI models</a> for running on your own hardware.</p>'
        ),
        MockRSSEntry(
            title='<h2>Fine-tuning LLMs for Specific Tasks</h2>',
            summary='<p>Guide to <b>fine-tuning</b> language models for specific domains. We\'ll explore <i>LoRA</i>, <code>QLoRA</code>, and <a href="#">parameter efficient training</a> techniques.</p>'
        ),
        MockRSSEntry(
            title='<h3>Hardware Requirements for Local LLMs</h3>',
            summary='<p>Understanding <strong>hardware requirements</strong> for running large language models locally. Topics include <em>GPU memory</em>, <code>quantization</code>, and <a href="#">optimization strategies</a>.</p>'
        ),
        MockRSSEntry(
            title='<h4>Model Comparison: Llama vs Mistral vs Others</h4>',
            summary='<p>Comprehensive <strong>comparison</strong> of different open source models. Performance benchmarks, <em>resource usage</em>, <code>inference speed</code>, and <a href="#">quality assessments</a>.</p>'
        ),
        MockRSSEntry(
            title='<h5>Deployment Strategies for Local LLMs</h5>',
            summary='<p>Best practices for <strong>deploying</strong> local language models in production. Docker containers, <em>API servers</em>, <code>load balancing</code>, and <a href="#">monitoring solutions</a>.</p>'
        )
    ]


@pytest.fixture
def search_engine():
    '''Create a VectorSearchEngine instance for testing.'''
    return VectorSearchEngine()


class TestVectorSearchEngine:
    '''Test the VectorSearchEngine class.'''

    def test_init(self):
        '''Test VectorSearchEngine initialization.'''
        engine = VectorSearchEngine()
        assert engine._embedding_model is None
        assert engine._chunker is None
        assert engine._html_converter is None

    def test_html_to_markdown_with_html2text(self, search_engine):
        '''Test HTML to Markdown conversion with html2text available.'''
        html_input = '<h1>Title</h1><p>This is <strong>bold</strong> and <em>italic</em> text.</p>'
        
        with patch('discord_aiagent.searchutil.html2text') as mock_html2text:
            mock_converter = Mock()
            mock_converter.handle.return_value = '# Title\n\nThis is **bold** and _italic_ text.\n'
            mock_html2text.HTML2Text.return_value = mock_converter
            
            result = search_engine.html_to_markdown(html_input)
            
            assert result == '# Title\n\nThis is **bold** and _italic_ text.\n'
            mock_converter.handle.assert_called_once_with(html_input)

    def test_html_to_markdown_without_html2text(self, search_engine):
        '''Test HTML to Markdown conversion fallback when html2text is not available.'''
        html_input = '<h1>Title</h1><p>This is <strong>bold</strong> text.</p>'
        
        with patch('discord_aiagent.searchutil.html2text', None):
            result = search_engine.html_to_markdown(html_input)
            
            assert result == 'TitleThis is bold text.'

    def test_html_to_markdown_empty_input(self, search_engine):
        '''Test HTML to Markdown conversion with empty input.'''
        result = search_engine.html_to_markdown('')
        assert result == ''
        
        result = search_engine.html_to_markdown(None)
        assert result == ''

    def test_chunk_text_with_chonkie(self, search_engine):
        '''Test text chunking with Chonkie available.'''
        text = 'This is sentence one. This is sentence two. This is sentence three.'
        
        with patch('discord_aiagent.searchutil.SentenceChunker') as mock_chunker_class:
            mock_chunker = Mock()
            mock_chunk1 = Mock()
            mock_chunk1.text = 'This is sentence one. This is sentence two. This is sentence three.'
            mock_chunker.chunk.return_value = [mock_chunk1]
            mock_chunker_class.return_value = mock_chunker
            
            result = search_engine.chunk_text(text)
            
            assert result == ['This is sentence one. This is sentence two. This is sentence three.']

    def test_chunk_text_without_chonkie(self, search_engine):
        '''Test text chunking fallback when Chonkie is not available.'''
        text = 'This is sentence one. This is sentence two. This is sentence three.'
        
        with patch('discord_aiagent.searchutil.SentenceChunker', None):
            result = search_engine.chunk_text(text)
            
            assert len(result) == 3
            assert 'sentence one' in result[0]
            assert 'sentence two' in result[1]
            assert 'sentence three' in result[2]

    def test_chunk_text_empty_input(self, search_engine):
        '''Test text chunking with empty input.'''
        result = search_engine.chunk_text('')
        assert result == []
        
        result = search_engine.chunk_text(None)
        assert result == []

    def test_prepare_entry_chunks(self, search_engine, sample_rss_entries):
        '''Test preparing chunks from RSS entries.'''
        with patch.object(search_engine, 'html_to_markdown') as mock_html_to_md:
            mock_html_to_md.side_effect = lambda x: x.replace('<h1>', '# ').replace('<p>', '').replace('</p>', '')
            
            with patch.object(search_engine, 'chunk_text') as mock_chunk:
                mock_chunk.side_effect = lambda x: [x] if x.strip() else []
                
                chunks = search_engine.prepare_entry_chunks(sample_rss_entries)
                
                assert len(chunks) == len(sample_rss_entries)
                assert all(isinstance(chunk, SearchChunk) for chunk in chunks)
                assert chunks[0].entry_index == 0
                assert chunks[0].entry == sample_rss_entries[0]

    def test_vector_search_with_sentence_transformer(self, search_engine, sample_rss_entries):
        '''Test vector search with SentenceTransformer available.'''
        query = 'open source AI models'
        
        with patch('discord_aiagent.searchutil.SentenceTransformer') as mock_transformer_class:
            mock_model = Mock()
            mock_model.encode.side_effect = [
                np.array([[0.1, 0.2, 0.3]]),  # Query embedding
                np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])  # Chunk embeddings
            ]
            mock_transformer_class.return_value = mock_model
            
            with patch.object(search_engine, 'prepare_entry_chunks') as mock_prepare:
                mock_chunk1 = SearchChunk('LocalLLaMA: Open Source AI Models', 0, 0, sample_rss_entries[0])
                mock_chunk2 = SearchChunk('Fine-tuning LLMs for Specific Tasks', 1, 0, sample_rss_entries[1])
                mock_chunk3 = SearchChunk('Hardware Requirements for Local LLMs', 2, 0, sample_rss_entries[2])
                mock_prepare.return_value = [mock_chunk1, mock_chunk2, mock_chunk3]
                
                results = search_engine.vector_search(sample_rss_entries, query, cosine_threshold=0.5, top_k=3)
                
                assert len(results) > 0
                assert all(isinstance(result, SearchResult) for result in results)

    def test_vector_search_without_sentence_transformer(self, search_engine, sample_rss_entries):
        '''Test vector search when SentenceTransformer is not available.'''
        query = 'open source AI models'
        
        # Mock the _get_embedding_model method to return None
        with patch.object(search_engine, '_get_embedding_model', return_value=None):
            results = search_engine.vector_search(sample_rss_entries, query)
            
            assert results == []

    def test_vector_search_empty_input(self, search_engine):
        '''Test vector search with empty input.'''
        results = search_engine.vector_search([], 'query')
        assert results == []
        
        results = search_engine.vector_search([MockRSSEntry('title', 'summary')], '')
        assert results == []

    def test_keyword_search(self, search_engine, sample_rss_entries):
        '''Test keyword search functionality.'''
        query = 'open source'
        
        with patch.object(search_engine, 'html_to_markdown') as mock_html_to_md:
            mock_html_to_md.side_effect = lambda x: x.replace('<h1>', '# ').replace('<p>', '').replace('</p>', '')
            
            with patch.object(search_engine, 'chunk_text') as mock_chunk:
                mock_chunk.side_effect = lambda x: [x] if x.strip() else []
                
                results = search_engine.keyword_search(sample_rss_entries, query)
                
                assert len(results) > 0
                assert all(isinstance(result, SearchResult) for result in results)
                assert all(result.score == 1.0 for result in results)

    def test_keyword_search_no_matches(self, search_engine, sample_rss_entries):
        '''Test keyword search with no matches.'''
        query = 'nonexistent term'
        
        results = search_engine.keyword_search(sample_rss_entries, query)
        
        assert results == []

    def test_keyword_search_empty_input(self, search_engine):
        '''Test keyword search with empty input.'''
        results = search_engine.keyword_search([], 'query')
        assert results == []
        
        results = search_engine.keyword_search([MockRSSEntry('title', 'summary')], '')
        assert results == []


class TestGlobalFunctions:
    '''Test global functions in searchutil module.'''

    def test_get_vector_search_engine_singleton(self):
        '''Test that get_vector_search_engine returns a singleton.'''
        engine1 = get_vector_search_engine()
        engine2 = get_vector_search_engine()
        
        assert engine1 is engine2
        assert isinstance(engine1, VectorSearchEngine)


class TestSearchChunk:
    '''Test the SearchChunk dataclass.'''

    def test_search_chunk_creation(self, sample_rss_entries):
        '''Test SearchChunk creation and attributes.'''
        entry = sample_rss_entries[0]
        chunk = SearchChunk(
            text='Test chunk text',
            entry_index=0,
            chunk_index=1,
            entry=entry
        )
        
        assert chunk.text == 'Test chunk text'
        assert chunk.entry_index == 0
        assert chunk.chunk_index == 1
        assert chunk.entry == entry


class TestSearchResult:
    '''Test the SearchResult dataclass.'''

    def test_search_result_creation(self, sample_rss_entries):
        '''Test SearchResult creation and attributes.'''
        entry = sample_rss_entries[0]
        result = SearchResult(
            entry=entry,
            score=0.85,
            best_chunk_text='Best matching chunk',
            best_chunk_index=2
        )
        
        assert result.entry == entry
        assert result.score == 0.85
        assert result.best_chunk_text == 'Best matching chunk'
        assert result.best_chunk_index == 2


@pytest.mark.integration
class TestIntegration:
    '''Integration tests for the search functionality.'''

    def test_full_search_pipeline(self, sample_rss_entries):
        '''Test the full search pipeline from RSS entries to results.'''
        search_engine = get_vector_search_engine()
        query = 'local language models'
        
        # Test vector search first
        vector_results = search_engine.vector_search(sample_rss_entries, query, cosine_threshold=0.3, top_k=3)
        
        # If vector search fails, test keyword search
        if not vector_results:
            keyword_results = search_engine.keyword_search(sample_rss_entries, query)
            assert len(keyword_results) > 0
            assert all(isinstance(result, SearchResult) for result in keyword_results)
        else:
            assert len(vector_results) > 0
            assert all(isinstance(result, SearchResult) for result in vector_results)

    def test_html_conversion_and_chunking_pipeline(self, sample_rss_entries):
        '''Test the full pipeline of HTML conversion and chunking.'''
        search_engine = get_vector_search_engine()
        
        # Test HTML to Markdown conversion
        for entry in sample_rss_entries:
            title_md = search_engine.html_to_markdown(entry.title)
            summary_md = search_engine.html_to_markdown(entry.summary)
            
            assert title_md != entry.title  # Should be converted
            assert summary_md != entry.summary  # Should be converted
            
            # Test chunking of converted text
            full_text = f'{title_md}\n\n{summary_md}'
            chunks = search_engine.chunk_text(full_text)
            
            assert len(chunks) > 0
            assert all(isinstance(chunk, str) for chunk in chunks)
            assert all(len(chunk.strip()) > 0 for chunk in chunks) 