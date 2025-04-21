**Discord-AIBot**

Discord bot for supporting AI/LLM chat applications powered by the Model Context Protocol (MCP), allowing for numerous integrations

For general MCP resources, see Arkestra:cookbook/mcp/README.md

# Configuration

```sh
cp config/example.main.toml config/main.toml
```

Then edit `main.toml` as needed. It specifies your LLM endpoint and model context resources such as MCP servers.

# Running

If you included `toys.b4a.toml` in your `main.toml` you'll need to have that MCP server running. In a separate terminal `cd demo_server` then run

```sh
uv pip install -Ur requirements.txt
uvicorn toy_mcp_server:create_app --factory --host 127.0.0.1 --port 8902
```

Make sure you set up any other MCP or other resources you've specified in your B4A. Now you can run the bot.

```sh
# Assumes you've exported DISCORD_TOKEN="YOUR_TOKEN" 
python mcp_discord_bot.py --discord-token $DISCORD_TOKEN --config-path config/mymain.toml
```

Structlog/rich tracebacks can be elaborate, so there is a `--classic-tracebacks` option to tame them

For very copious logging you can add `--loglevel DEBUG`

Note: you can use the environment rather than `--discord-token` & `--config-path`

```sh
export AIBOT_DISCORD_TOKEN="YOUR_TOKEN"
export AIBOT_DISCORD_CONFIG_PATH="./config.toml"
python mcp_discord_bot.py # Reads from env vars
```

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

Note: with mlx-omni-server ran into `RuntimeError: Failed to generate completion: generate_step() got an unexpected keyword argument 'user'`

There are [many local MLX models](https://huggingface.co/mlx-community) from which you can pick

# Checking MCP servers

For SSE servers, you can check with curl, e.g.

```sh
curl -N http://localhost:8901/sse
```
