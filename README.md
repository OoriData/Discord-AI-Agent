**Discord AI Agent**

Discord bot for supporting AI/LLM chat applications powered by the Model Context Protocol (MCP), allowing for numerous integrations

For general MCP resources, see Arkestra:cookbook/mcp/README.md

# Install

```sh
uv pip install -U .
```

# Configuration

```sh
cp config/example.main.toml config/main.toml
```

Then edit `main.toml` as needed. It specifies your LLM endpoint and model context resources such as MCP servers.

Follow the same pattern with `config/*news*.example.b4a.toml` depending on what context tools you plan to use, e.g.

```sh
cp config/toys.example.b4a.toml config/toys.b4a.toml
```

# Running

If you included `toys.b4a.toml` in your `main.toml` you'll need to have that MCP server running. In a separate terminal:

```sh
cd demo_server 
uv pip install -Ur requirements.txt --constraint=constraints.txt
uvicorn toy_mcp_server:create_app --factory --host 127.0.0.1 --port 8902
```

To use a different port, make sure you also update the `b4a.toml` file

Make sure you set up any other MCP or other resources you've specified in your B4A. Now you can run the bot.

```sh
# Assumes you've exported DISCORD_TOKEN="YOUR_TOKEN"
python mcp_discord_bot.py --discord-token $DISCORD_TOKEN --config-path config
```

Structlog/rich tracebacks can be elaborate, so there is a `--classic-tracebacks` option to tame them

For very copious logging you can add `--loglevel DEBUG`

Note: you can use the environment rather than `--discord-token` & `--config-path`

```sh
export AIBOT_DISCORD_TOKEN="YOUR_TOKEN"
export AIBOT_DISCORD_CONFIG_PATH="./config"
python mcp_discord_bot.py # Reads from env vars
```

# Quick start

Launch servers & bot, then DM the bot

Use command `/context_info`

Use command `/context_details`


# Sample queries

## From the toy MCP servers

* Ask the magic 8-ball if I should deploy to production on Friday
* Consult the magic 8-ball about my chances of winning the lottery
* Magic 8-ball, tell me: will it rain tomorrow?

## From RSS (Based on r/LocalLlama)

* Any recent observations about Qwen 3 among Local LLM enthusiasts today?

## PGVector chat history

To enable persistent chat history storage using a PostgreSQL database with the PGVector extension, Ensure you have a running PostgreSQL database server with the [pgvector extension](https://github.com/pgvector/pgvector) available, then set up the environment variables accordingly. See below for notes on DB setup.

### Additional Python dependencies:

```sh
uv pip install ogbujipt pgvector asyncpg sentence-transformers
```

### Enable the Feature

Set the `AIBOT_PGVECTOR_HISTORY_ENABLED` environment variable to `true`:

You can either use a connection string or elaborated credentials.

### Configure with DB connection string:

Make sure it's in the `AIBOT_PG_USER_CONNECT_STRING` environment variable. If this variable exists, it will supersede any other PG env vars.

### Configure with DB connection components

Other environment variables:

* **Required:**
    * `AIBOT_PG_DB_NAME`: Name of the database to use
    * `AIBOT_PG_USER`: Role name to use. It's a single-role authentication system
    * `AIBOT_PG_PASSWORD`: Role's password
* **Optional (Defaults Shown):**
    * `AIBOT_PG_HOST`: Database host (default: `localhost`)
    * `AIBOT_PG_PORT`: Database port (default: `5432`)
    * `AIBOT_PG_TABLE_NAME`: Table name for storing history (default: `discord_chat_history`)
    * `AIBOT_PG_EMBEDDING_MODEL`: Sentence Transformer model to use for embeddings (default: `all-MiniLM-L6-v2`)

3.  **Database Setup:** [cite: 2]. You can use Docker for a quick setup (similar to the demo notebook [cite: 3]):

```sh
# Example using default user/pass/db - change values as needed!
docker run --name pg_chat_history -d -p 5432:5432 \
    -e POSTGRES_DB=YOUR_PGVECTOR_DB_NAME \
    -e POSTGRES_USER=YOUR_PGVECTOR_USER \
    -e POSTGRES_PASSWORD=YOUR_PGVECTOR_PASSWORD \
    pgvector/pgvector
```

Replace `YOUR_PGVECTOR_DB_NAME`, `YOUR_PGVECTOR_USER`, and `YOUR_PGVECTOR_PASSWORD` with the values you set in the environment variables.

The bot will automatically connect to the database, create the table (if it doesn't exist), and start storing/retrieving chat history
from PGVector. If any required variables are missing or the connection fails, it will fall back to the default in-memory history.


# Implementation notes

* [discord.py](https://github.com/Rapptz/discord.py)
* [mcp-sse-client](https://github.com/zanetworker/mcp-sse-client-python) MCP client library

# Using mlx_lm.server

```sh
uv pip install mlx-omni-server
mlx-omni-server --port 1234
# uv pip install mlx mlx_lm
# mlx_lm.server --model mlx-community/Llama-3.2-3B-Instruct-4bit --port 1234
```

Note: with mlx-omni-server [ran into `RuntimeError: Failed to generate completion: generate_step() got an unexpected keyword argument 'user'`](https://github.com/madroidmaq/mlx-omni-server/issues/37)

Fixed with this patch:

```diff
diff --git a/chat/mlx/mlx_model.py b/chat/mlx/mlx_model.py
index da7aef5..094ae9c 100644
--- a/chat/mlx/mlx_model.py
+++ b/chat/mlx/mlx_model.py
@@ -45,6 +45,9 @@ class MLXModel(BaseTextModel):
 
     def _get_generation_params(self, request: ChatCompletionRequest) -> Dict[str, Any]:
         params = request.get_extra_params()
+        # Exclude user. See #37
+        if "user" in params:
+            del params["user"]
         known_params = {
             "top_k",
             "min_tokens_to_keep",
```

There are [many local MLX models](https://huggingface.co/mlx-community) from which you can pick

# Checking MCP servers

For SSE servers, you can check with curl, e.g.

```sh
curl -N http://localhost:8901/sse
```

# Troubleshooting PGVector

You can try a simple connection as follows, to make sure there is no exception:

```py
import os
import asyncio
from sentence_transformers import SentenceTransformer
from ogbujipt.embedding.pgvector import MessageDB
emodel_name = os.environ.get('AIBOT_PG_EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
emodel = SentenceTransformer(emodel_name)
su_conn_str = os.environ.get('AIBOT_PG_SUPERUSER_CONNECT_STRING')
tname = os.environ.get('AIBOT_PG_TABLE_NAME', 'discord_chat_history')
db = asyncio.run(await MessageDB.from_conn_string(su_conn_str, emodel, tname))
```

## Note for Supabase

You can use the "Connect" icon at the top to get connection string info

Unless you buy an IPV4 add-on you need to use the session pooler version of the connection string,
or you'll get `nodename nor servname provided, or not known`. Ref: https://github.com/orgs/supabase/discussions/33534

If so, don't forget to include the tenant ID (e.g. `[USER].hjdsfghjfbdhsk`; the part after the dot) or you'll get InternalServerError: Tenant or user not found

It's probably a good idea to have an app-level user, in order to assert least privilege.

You can just use `util/supabase_setup.py`, which you should run only once.

```sh
op run --env-file .env -- python util/supabase_setup.py
```

Note: [Supavisor does not support prepared statements in transaction mode. It does via direct connections & in session mode.](https://supabase.com/docs/guides/troubleshooting/disabling-prepared-statements-qL8lEL)

To make sure asyncpg doesn't cause probs with this

> Disable automatic use of prepared statements by passing `statement_cache_size=0` to `asyncpg.connect()` and `asyncpg.create_pool()` (and, obviously, avoid the use of `Connection.prepare()`

## Public schema warning

Tables are in the `public` schema in Supabase, are automatically exposed via Supabase's auto-generated REST API. Without RLS enabled, any table in the public schema is accessible through the API to anyone who has this anon key, which is exposed in client-side code.

If you do have your Supabase table in the `public` schema, we recommend you enable RLS on it, without creating any policies. That will blocks all public API access to the table, while the configured service role can bypass RLS. For example:

```sql
ALTER TABLE public.discord_chat_history ENABLE ROW LEVEL SECURITY;
```
