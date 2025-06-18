# Run only once
# op run --env-file .env -- python util/supabase_setup.py
import os
import asyncio
import asyncpg

from sentence_transformers import SentenceTransformer
from ogbujipt.embedding.pgvector import MessageDB
from discord_aiagent import assemble_pgvector_config

pgvector_config = assemble_pgvector_config()

async def create_app_user():
    # Connect as superuser (e.g., 'postgres')
    conn = await asyncpg.connect(pgvector_config['su_conn_str'])
    try:
        # Create the application user (idempotent: IF NOT EXISTS)
        await conn.execute('''
            DO $$ BEGIN 
            IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = '{username}') THEN 
            CREATE ROLE {username} WITH LOGIN PASSWORD '{password}';
            END IF;
            END $$;
        '''.format(username=pgvector_config['username'], password=pgvector_config['password']))
        # Grant privileges on your table (replace 'discord_chat_history' if needed)
        await conn.execute(
            f'GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO {pgvector_config["username"]};'
        )
        # (Optional) Grant usage on the schema if needed
        await conn.execute(
            f'GRANT USAGE ON SCHEMA public TO {pgvector_config["username"]};'
        )
        print(f'App user {pgvector_config["username"]} created and privileges granted.')
    finally:
        await conn.close()

async def db_setup():
    db = await MessageDB.from_conn_string(pgvector_config['su_conn_str'], pgvector_config['embedding_model'], pgvector_config['table_name'])
    await db.create_table()
    return db

db = asyncio.run(db_setup())
asyncio.run(create_app_user())
