# Just make sure we can connect to supabase with the provided connection string
# op run --env-file .env -- python util/check_supabase.py
import os
import asyncio
import asyncpg

from sentence_transformers import SentenceTransformer
from ogbujipt.embedding.pgvector import MessageDB

emodel = SentenceTransformer('all-MiniLM-L6-v2')

su_conn_str = os.environ.get('AIBOT_PG_SUPERUSER_CONNECT_STRING')
user_conn_str = os.environ.get('AIBOT_PG_USER_CONNECT_STRING')
print(user_conn_str)
username = os.environ.get('AIBOT_PG_USER')
password = os.environ.get('AIBOT_PG_USER_PASSWORD')
tname = os.environ.get('AIBOT_PG_TABLE_NAME', 'discord_chat_history')

async def get_db():
    db = await MessageDB.from_conn_string(su_conn_str, emodel, tname)
    return db

async def main():
    # Test connect as superuser
    conn = await asyncpg.connect(su_conn_str)
    # Test connect as app user
    conn = await asyncpg.connect(user_conn_str)
    # Test for OgbujiPT
    db = await get_db()

asyncio.run(main())
