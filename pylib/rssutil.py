# SPDX-FileCopyrightText: 2025-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# discord_aiagent.rssutil
from typing import Any

from source_handlers import fetch_and_parse_rss  # , FEEDPARSER_AVAILABLE


class RSSAgent:
    def __init__(self, b4a_data, rss_feed_cache, rss_cache_ttl):
        self.b4a_data = b4a_data
        self._rss_feed_cache = rss_feed_cache
        self._rss_cache_ttl = rss_cache_ttl

    async def handle_tool(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, str]:
        '''Handles the execution of the specified tool.'''
        if tool_name == 'query_rss_feed':
            return await self.execute_rss_query(tool_input)
        else:
            return {'error': f'Unknown tool name: {tool_name}'}

    async def execute_rss_query(self, tool_input: dict[str, Any]) -> dict[str, str]:
        '''Handles the execution of the 'query_rss_feed' tool.'''
        feed_name = tool_input.get('feed_name')
        query = tool_input.get('query') # Optional query string
        limit = tool_input.get('limit', 5) # Default limit

        if not feed_name:
            return {'error': 'Missing required parameter: feed_name'}
        if not isinstance(limit, int) or limit < 1:
            limit = 5 # Default to 5 if invalid limit provided

        # Find the RSSConfig
        rss_conf = next((conf for conf in self.b4a_data.rss_sources if conf.name == feed_name), None)
        if not rss_conf:
            return {'error': f'Configured RSS feed named `{feed_name}` not found.'}

        # Fetch and parse using the handler function
        parsed_feed, error = await fetch_and_parse_rss(rss_conf.url, self._rss_feed_cache, self._rss_cache_ttl)

        if error:
            return {'error': f'Error fetching feed `{feed_name}`: {error}'}
        if not parsed_feed or not hasattr(parsed_feed, 'entries'):
                return {'error': f'Failed to get valid entries from feed `{feed_name}`.'}

        # Filter entries if query is provided
        matching_entries = []
        if query:
            query_lower = query.lower()
            for entry in parsed_feed.entries:
                title = entry.get('title', '').lower()
                summary = entry.get('summary', '').lower()
                # Basic keyword check in title or summary
                if query_lower in title or query_lower in summary:
                    matching_entries.append(entry)
        else:
            # No query, use all entries (up to the limit)
            matching_entries = parsed_feed.entries

        # Limit the results
        limited_entries = matching_entries[:limit]

        if not limited_entries:
            result_text = f'No entries found in feed `{feed_name}`'
            if query: result_text += f' matching query `{query}`'
            return {'result': result_text}

        # Format the results
        result_items = []
        for entry in limited_entries:
            title = entry.get('title', 'No Title')
            link = entry.get('link', 'No Link')
            summary = entry.get('summary', 'No Summary')
            # Simple formatting, truncate summary
            summary_short = (summary[:200] + '...') if len(summary) > 200 else summary
            result_items.append(f'- Title: {title}\n  Link: {link}\n  Summary: {summary_short}')

        final_result = f'Found {len(limited_entries)} entries from feed `{feed_name}`'
        if query: final_result += f' matching query `{query}`'
        final_result += ':\n\n' + '\n\n'.join(result_items)

        return {'result': final_result}
