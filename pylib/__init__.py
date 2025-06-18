# SPDX-FileCopyrightText: 2025-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# discord_aiagent

import os
import urllib.parse

import structlog
from sentence_transformers import SentenceTransformer

logger = structlog.get_logger()


def assemble_pgvector_config():
    pgvector_enabled = os.environ.get('AIBOT_PGVECTOR_HISTORY_ENABLED', 'false').lower() == 'true'
    if not pgvector_enabled:
        return None

    logger.info('PGVector history is ENABLED via environment variable.')
    # Use resolve_value for potential env var in model name?
    embedding_model_name = os.environ.get('AIBOT_PG_EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
    logger.info(f'Loading sentence transformer model: {embedding_model_name}')
    embedding_model = SentenceTransformer(embedding_model_name)
    logger.info('Sentence transformer model loaded.')

    # XXX: Needs to take into account case of AIBOT_PG_CONNECT_STRING == 'COMPOSE'
    # # Validate required PGVector env vars are present
    # required_pg_vars = ['AIBOT_PG_DB_NAME', 'AIBOT_PG_USER', 'AIBOT_PG_PASSWORD']
    # missing_vars = [var for var in required_pg_vars if not os.environ.get(var)]
    # if missing_vars:
    #     logger.error(f'PGVector enabled, but missing required environment variables: {missing_vars}. Disabling.')
    #     pgvector_config['enabled'] = False  # Disable if config is incomplete
    #     pgvector_enabled = False

    pgvector_config = {
        'enabled': True,
        'embedding_model_name': embedding_model_name,
        'embedding_model': embedding_model,
        'conn_str': os.environ.get('AIBOT_PG_CONNECT_STRING'),
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
