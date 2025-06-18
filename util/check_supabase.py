# Just make sure we can connect to supabase with the provided connection string
# op run --env-file .env -- python util/check_supabase.py
import os
import asyncio
import asyncpg

from sentence_transformers import SentenceTransformer
from ogbujipt.embedding.pgvector import MessageDB
from discord_aiagent import assemble_pgvector_config

pgvector_config = assemble_pgvector_config()

async def get_db():
    db = await MessageDB.from_conn_string(pgvector_config['su_conn_str'], pgvector_config['embedding_model'], pgvector_config['table_name'])
    return db

async def main():
    # Test connect as superuser
    # print(pgvector_config['su_conn_str'])
    conn = await asyncpg.connect(pgvector_config['su_conn_str'])
    # Test connect as app user
    # print(pgvector_config['conn_str'])
    conn = await asyncpg.connect(pgvector_config['conn_str'])
    await asyncio.sleep(3)
    # Test for OgbujiPT
    db = await get_db()

asyncio.run(main())
