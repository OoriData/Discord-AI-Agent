# SPDX-FileCopyrightText: 2025-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# discord_aiagent.rssutil
import time
import email.utils
from typing import Any
import structlog

from discord_aiagent.source_handlers import fetch_and_parse_rss
from discord_aiagent.searchutil import get_vector_search_engine, SearchResult

logger = structlog.get_logger(__name__)

class RSSAgent:
    def __init__(self, b4a_data, rss_feed_cache, rss_cache_ttl):
        self.b4a_data = b4a_data
        self._rss_feed_cache = rss_feed_cache
        self._rss_cache_ttl = rss_cache_ttl
        self._search_engine = get_vector_search_engine()

    async def handle_tool(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, str]:
        '''Handles the execution of the specified tool.'''
        if tool_name == 'query_rss_feed':
            return await self.execute_rss_query(tool_input)
        else:
            return {'error': f'Unknown tool name: {tool_name}'}

    async def execute_rss_query(self, tool_input: dict[str, Any]) -> dict[str, str]:
        '''Handles the execution of the 'query_rss_feed' tool.'''
        feed_name = tool_input.get('feed_name')
        query = tool_input.get('query')  # Optional query string
        limit = tool_input.get('limit', 5)  # Default limit
        since_timestamp = tool_input.get('since_timestamp')  # Optional timestamp filter

        if not feed_name:
            return {'error': 'Missing required parameter: feed_name'}
        if not isinstance(limit, int) or limit < 1:
            limit = 5  # Default to 5 if invalid limit provided

        # Find the RSSConfig
        rss_conf = next((conf for conf in self.b4a_data.rss_sources if conf.name == feed_name), None)
        if not rss_conf:
            return {'error': f'Configured RSS feed named `{feed_name}` not found.'}

        # Get vector search config from feed config (with sensible defaults)
        cosine_threshold = float(getattr(rss_conf, 'cosine_similarity_threshold', 0.65))
        top_k = int(getattr(rss_conf, 'top_k', limit))
        logger.debug('RSS vector search config', feed=feed_name, cosine_threshold=cosine_threshold, top_k=top_k)

        # Fetch and parse using the handler function
        logger.debug('Fetching and parsing RSS feed', url=rss_conf.url)
        parsed_feed, error = await fetch_and_parse_rss(rss_conf.url, self._rss_feed_cache, self._rss_cache_ttl)

        if error:
            logger.error('Error fetching RSS feed', feed=feed_name, error=error)
            return {'error': f'Error fetching feed `{feed_name}`: {error}'}
        if not parsed_feed or not hasattr(parsed_feed, 'entries'):
            logger.error('No valid entries in RSS feed', feed=feed_name)
            return {'error': f'Failed to get valid entries from feed `{feed_name}`.'}
        if not parsed_feed.entries:
            logger.error('RSS feed has no entries', feed=feed_name)
            return {'error': f'Failed to get valid entries from feed `{feed_name}`.'}

        logger.debug('Fetched RSS entries', feed=feed_name, entry_count=len(parsed_feed.entries))
        
        # Debug: Log some sample entries to see what's in the feed
        if len(parsed_feed.entries) > 0:
            sample_entries = parsed_feed.entries[:3]  # Show first 3 entries
            for i, entry in enumerate(sample_entries):
                title = entry.get('title', 'No Title')
                summary = entry.get('summary', 'No Summary')[:100]  # First 100 chars
                logger.debug(f'Sample RSS entry {i+1}', feed=feed_name, title=title, summary=summary)
        else:
            logger.warning('RSS feed has no entries', feed=feed_name, url=rss_conf.url)

        # Filter entries by timestamp if provided
        matching_entries = self._filter_entries_by_timestamp(parsed_feed.entries, since_timestamp)
        if matching_entries is None:
            return {'error': f'Invalid since_timestamp value: {since_timestamp}. Must be a valid Unix timestamp.'}

        # Perform search if query provided
        search_results = []
        if query:
            search_results = self._perform_search(matching_entries, query, cosine_threshold, top_k)
            matching_entries = [result.entry for result in search_results]
        else:
            # No query, just use all entries
            matching_entries = matching_entries[:limit]

        # Limit the results (in case top_k < limit)
        limited_entries = matching_entries[:limit]
        logger.debug('Final limited entries', count=len(limited_entries))

        if not limited_entries:
            result_text = f'No entries found in feed `{feed_name}`'
            if since_timestamp is not None:
                result_text += f' since {since_timestamp}'
            if query: 
                result_text += f' matching query `{query}`'
            logger.info('No matching RSS entries found', feed=feed_name, query=query)
            return {'error': f'Failed to get valid entries from feed `{feed_name}`.'}

        # Format the results
        result_items = self._format_search_results(limited_entries, search_results, query)
        
        final_result = f'Found {len(limited_entries)} entries from feed `{feed_name}`'
        if since_timestamp is not None:
            final_result += f' since {since_timestamp}'
        if query: 
            final_result += f' matching query `{query}`'
        final_result += ':\n\n' + '\n\n'.join(result_items)
        logger.info('Returning RSS query results', feed=feed_name, count=len(limited_entries))
        return {'result': final_result}

    def _filter_entries_by_timestamp(self, entries, since_timestamp):
        """Filter entries by timestamp if provided."""
        if since_timestamp is None:
            return entries
            
        try:
            since_time = float(since_timestamp)
            filtered_entries = []
            for entry in entries:
                published_time = None
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    published_time = email.utils.mktime_tz(entry.published_parsed)
                elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                    published_time = email.utils.mktime_tz(entry.updated_parsed)
                elif hasattr(entry, 'published') and entry.published:
                    try:
                        parsed_date = email.utils.parsedate_tz(entry.published)
                        if parsed_date:
                            published_time = email.utils.mktime_tz(parsed_date)
                    except:
                        pass
                if published_time is None or published_time >= since_time:
                    filtered_entries.append(entry)
            logger.debug('Entries after timestamp filter', count=len(filtered_entries))
            return filtered_entries
        except (ValueError, TypeError) as e:
            logger.error('Invalid since_timestamp value', value=since_timestamp, error=str(e))
            return None

    def _perform_search(self, entries, query, cosine_threshold, top_k):
        """Perform vector search with fallback to keyword search."""
        # Try vector search first
        search_results = self._search_engine.vector_search(
            entries, query, cosine_threshold, top_k
        )
        
        if search_results:
            logger.debug('Vector search successful', query=query, results=len(search_results))
            return search_results
        
        # Fallback to keyword search
        logger.debug('Vector search failed or no results, falling back to keyword search', query=query)
        search_results = self._search_engine.keyword_search(entries, query)
        return search_results

    def _format_search_results(self, entries, search_results, query):
        """Format the search results for output."""
        result_items = []
        
        # Create a mapping from entry to search result for easy lookup
        result_map = {result.entry: result for result in search_results}
        
        for entry in entries:
            title = entry.get('title', 'No Title')
            link = entry.get('link', 'No Link')
            summary = entry.get('summary', 'No Summary')
            summary_short = (summary[:200] + '...') if len(summary) > 200 else summary
            
            # Check if this entry has search result metadata
            search_result = result_map.get(entry)
            if search_result and query:
                # Include search score and best matching chunk
                score_text = f" (Score: {search_result.score:.3f})"
                chunk_text = f"\n  Best Match: {search_result.best_chunk_text[:150]}..."
                result_items.append(f'- Title: {title}{score_text}\n  Link: {link}\n  Summary: {summary_short}{chunk_text}')
            else:
                # Standard format for non-search results
                result_items.append(f'- Title: {title}\n  Link: {link}\n  Summary: {summary_short}')
        
        return result_items
