# SPDX-FileCopyrightText: 2025-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# discord_aiagent

import os
import urllib.parse
from typing import Any

import structlog
from sentence_transformers import SentenceTransformer

logger = structlog.get_logger()


def assemble_pgvector_config(config: dict[str, Any] | None = None):
    '''
    Assemble PGVector configuration from config file and environment variables.

    Args:
        config: Main config dict. If None, checks for legacy AIBOT_PGVECTOR_HISTORY_ENABLED env var.

    Returns:
        PGVector config dict if enabled, None otherwise.
    '''
    # Check if PGVector history is enabled via config or legacy env var
    pgvector_enabled = False
    if config and 'pgvector_history' in config:
        pgvector_history_config = config.get('pgvector_history', {})
        pgvector_enabled = pgvector_history_config.get('enabled', False)
        logger.info('PGVector history enabled via config file.')
    else:
        # Fallback to legacy environment variable check
        pgvector_enabled = os.environ.get('AIBOT_PGVECTOR_HISTORY_ENABLED', 'false').lower() == 'true'
        if pgvector_enabled:
            logger.warning('PGVector history enabled via deprecated AIBOT_PGVECTOR_HISTORY_ENABLED env var. Consider using [pgvector_history] section in config file instead.')

    if not pgvector_enabled:
        return None

    # Validate required environment variables are present
    conn_str = os.environ.get('AIBOT_PG_CONNECT_STRING')
    # If using connection string, we need at least that
    if conn_str:
        required_vars = []  # Connection string alone is sufficient
    else:
        # If using individual components, we need these
        required_vars = ['AIBOT_PG_DB_NAME', 'AIBOT_PG_USER', 'AIBOT_PG_USER_PASSWORD']

    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    if missing_vars:
        logger.warning(
            f'PGVector history enabled, but missing required environment variables: {missing_vars}. '
            'PGVector initialization may fail. Please set these variables or provide AIBOT_PG_CONNECT_STRING.'
        )
        # Don't disable here - let it fail during initialization with a clearer error

    # Use resolve_value for potential env var in model name?
    embedding_model_name = os.environ.get('AIBOT_PG_EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
    logger.info(f'Loading sentence transformer model: {embedding_model_name}')
    embedding_model = SentenceTransformer(embedding_model_name)
    logger.info('Sentence transformer model loaded.')

    pgvector_config = {
        'enabled': True,
        'embedding_model_name': embedding_model_name,
        'embedding_model': embedding_model,
        'conn_str': conn_str,
        'su_conn_str': os.environ.get('AIBOT_PG_SUPERUSER_CONNECT_STRING'),
        'table_name': os.environ.get('AIBOT_PG_TABLE_NAME', 'discord_chat_history'),
        'db_name': os.environ.get('AIBOT_PG_DB_NAME'),
        'host': os.environ.get('AIBOT_PG_HOST', 'localhost'),
        'port': int(os.environ.get('AIBOT_PG_PORT', 5432)),
        'user': os.environ.get('AIBOT_PG_USER'),
        'password': os.environ.get('AIBOT_PG_USER_PASSWORD'),
        'user_tenant': os.environ.get('AIBOT_PG_USER_TENANT', os.environ.get('AIBOT_PG_USER')),
    }
    pgvector_config['user_tenant'] = os.environ.get('AIBOT_PG_USER_TENANT', pgvector_config['user'])
    if pgvector_config['conn_str']:
        # e.g. common with Supabase
        if pgvector_config['conn_str'] == 'COMPOSE':
            logger.info('Constructing a connection string for PGVector from components')
            pgvector_config['conn_str'] = (
                f'postgres://{urllib.parse.quote(pgvector_config['user_tenant'])}:'
                f'{urllib.parse.quote(pgvector_config['password'])}'
                # f'@{pgvector_config['host']}:{pgvector_config['port']}/{pgvector_config['db_name']}'
                f'@{pgvector_config['host']}:{pgvector_config['port']}/postgres'
            )

    return pgvector_config
