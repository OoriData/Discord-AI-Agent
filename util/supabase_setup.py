# Run only once
# op run --env-file .env -- python util/supabase_setup.py
import os
import asyncio
import asyncpg

from sentence_transformers import SentenceTransformer
from ogbujipt.embedding.pgvector import MessageDB

emodel = SentenceTransformer('all-MiniLM-L6-v2')

su_conn_str = os.environ.get('AIBOT_PG_SUPERUSER_CONNECT_STRING')
user_conn_str = os.environ.get('AIBOT_PG_USER_CONNECT_STRING')
username = os.environ.get('AIBOT_PG_USER')
password = os.environ.get('AIBOT_PG_USER_PASSWORD')
tname = os.environ.get('AIBOT_PG_TABLE_NAME', 'discord_chat_history')

async def create_app_user():
    # Connect as superuser (e.g., 'postgres')
    conn = await asyncpg.connect(su_conn_str)
    try:
        # Create the application user (idempotent: IF NOT EXISTS)
        await conn.execute('''
            DO $$ BEGIN 
            IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = '{username}') THEN 
            CREATE ROLE {username} WITH LOGIN PASSWORD '{password}';
            END IF;
            END $$;
        '''.format(username=username, password=password))
        # Grant privileges on your table (replace 'discord_chat_history' if needed)
        await conn.execute(
            f'GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO {username};'
        )
        # (Optional) Grant usage on the schema if needed
        await conn.execute(
            f'GRANT USAGE ON SCHEMA public TO {username};'
        )
        print(f'App user {username} created and privileges granted.')
    finally:
        await conn.close()

async def db_setup():
    db = await MessageDB.from_conn_string(su_conn_str, emodel, tname)
    await db.create_table()
    return db

db = asyncio.run(db_setup())
asyncio.run(create_app_user())
