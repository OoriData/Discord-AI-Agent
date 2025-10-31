# SPDX-FileCopyrightText: 2025-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
'''
History provider abstraction for Discord AI Agent.
Provides unified interface for in-memory and PGVector history storage.
'''
import uuid
from abc import ABC, abstractmethod
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class HistoryProvider(ABC):
    '''Abstract base class for message history providers.'''

    @abstractmethod
    async def get_messages(self, channel_id: int, limit: int = 20) -> list[dict[str, Any]]:
        '''
        Get message history for a channel.
        Returns list of message dicts with 'role' and 'content' keys.
        '''
        pass

    @abstractmethod
    async def add_message(
        self,
        channel_id: int,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None
    ) -> None:
        '''
        Add a message to history.

        Args:
            channel_id: Discord channel ID
            role: Message role ('user', 'assistant', 'tool')
            content: Message content
            metadata: Optional metadata dict
        '''
        pass

    @abstractmethod
    async def clear(self, channel_id: int) -> None:
        '''Clear history for a channel.'''
        pass


class MemoryHistoryProvider(HistoryProvider):
    '''In-memory message history provider.'''

    def __init__(self, max_history_length: int = 20):
        self.message_history: dict[int, list[dict[str, Any]]] = {}
        self.max_history_length = max_history_length

    async def get_messages(self, channel_id: int, limit: int = 20) -> list[dict[str, Any]]:
        '''Get message history from in-memory cache.'''
        if channel_id not in self.message_history:
            self.message_history[channel_id] = []
            logger.info(f'Initialized in-memory message history for channel {channel_id}')
            return []

        # Trim history if it exceeds max_history_length
        current_len = len(self.message_history[channel_id])
        if current_len > self.max_history_length:
            amount_to_remove = current_len - self.max_history_length
            self.message_history[channel_id] = self.message_history[channel_id][amount_to_remove:]
            logger.debug(
                f'Trimmed in-memory message history for channel {channel_id}, '
                f'removed {amount_to_remove} messages.'
            )

        # Return last N messages (respecting limit param)
        history = self.message_history[channel_id]
        if len(history) > limit:
            return history[-limit:]
        return history

    async def add_message(
        self,
        channel_id: int,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None
    ) -> None:
        '''Add message to in-memory cache.'''
        if channel_id not in self.message_history:
            self.message_history[channel_id] = []

        message_dict: dict[str, Any] = {'role': role, 'content': content}

        # Store tool_calls if present in metadata
        if metadata and 'tool_calls' in metadata:
            message_dict['tool_calls'] = metadata['tool_calls']

        self.message_history[channel_id].append(message_dict)
        logger.debug(f'Added {role} message to in-memory history for channel {channel_id}')

    async def clear(self, channel_id: int) -> None:
        '''Clear in-memory history for a channel.'''
        if channel_id in self.message_history:
            del self.message_history[channel_id]
            logger.info(f'Cleared in-memory history for channel {channel_id}')


class PGVectorHistoryProvider(HistoryProvider):
    '''PGVector-backed message history provider.'''

    def __init__(self, pgvector_db, namespace_uuid: uuid.UUID):
        '''
        Initialize PGVector history provider.

        Args:
            pgvector_db: MessageDB instance from ogbujipt.embedding.pgvector
            namespace_uuid: UUID namespace for generating history keys
        '''
        self.pgvector_db = pgvector_db
        self.namespace_uuid = namespace_uuid

    def _get_history_key(self, channel_id: int) -> uuid.UUID:
        '''Generate deterministic UUIDv5 from channel ID.'''
        return uuid.uuid5(self.namespace_uuid, str(channel_id))

    async def get_messages(self, channel_id: int, limit: int = 20) -> list[dict[str, Any]]:
        '''Get message history from PGVector.'''
        history_key = self._get_history_key(channel_id)
        logger.debug(f'Retrieving history from PGVector for key: {history_key}')

        try:
            # Retrieve messages from PGVector
            pg_messages_gen = await self.pgvector_db.get_messages(history_key=history_key, limit=limit)
            # Convert generator to list and reverse it to get chronological order
            pg_messages = list(pg_messages_gen)
            pg_messages.reverse()  # Reverse to chronological

            # Format for LLM (ensure 'role' and 'content' keys)
            # Filter out any 'system' role messages, as the system message is now dynamic
            formatted_history = [
                {'role': msg['role'], 'content': msg['content']}
                for msg in pg_messages if msg.get('role') != 'system'
            ]

            # Restore tool_calls from metadata if present
            for i, msg in enumerate(formatted_history):
                if pg_messages[i].get('metadata', {}).get('tool_calls'):
                    formatted_history[i]['tool_calls'] = pg_messages[i]['metadata']['tool_calls']

            logger.debug(f'Retrieved {len(formatted_history)} messages from PGVector for key {history_key}')
            return formatted_history

        except Exception as e:
            logger.exception(f'Error retrieving history from PGVector for key {history_key}')
            raise

    async def add_message(
        self,
        channel_id: int,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None
    ) -> None:
        '''Add message to PGVector.'''
        history_key = self._get_history_key(channel_id)
        logger.debug(f'Storing {role} message to PGVector for key: {history_key}')

        try:
            # Ensure content is not None
            content_to_insert = content if content is not None else ''

            # Prepare metadata
            if metadata is None:
                metadata = {}

            await self.pgvector_db.insert(
                history_key=history_key,
                role=role,
                content=content_to_insert,
                metadata=metadata
            )
        except Exception as e:
            logger.exception(f'Failed to store {role} message to PGVector for key {history_key}')
            raise

    async def clear(self, channel_id: int) -> None:
        '''Clear PGVector history for a channel.'''
        history_key = self._get_history_key(channel_id)
        logger.info(f'Clearing PGVector history for key {history_key}')
        # Note: MessageDB doesn't have a clear method, so this is a no-op
        # In practice, old messages naturally age out based on the limit parameter
        logger.warning('PGVector history clearing not implemented - messages will age out naturally')
