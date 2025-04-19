**Discord-AIBot**

Discord bot for supporting AI/LLM chat applications powered by the Model Context Protocol (MCP), allowing for numerous integrations

For general MCP resources, see Arkestra:cookbook/mcp/README.md

# Configuration

```sh
cp example.config.toml myconfig.toml
```

Then edit `myconfig.toml` as needed. It specifies your LLM endpoint and MCP servers.

# Running

```sh
python mcp_discord_bot.py --discord-token $DISCORD_TOKEN --config-path myconfig.toml
```

Structlog/rich tracebacks can be elaborate, so there is a `--classic-tracebacks` option to tame them

Note: you can use the environment rather than `--discord-token` & `--config-path`

```sh
export MCP_DISCORD_DISCORD_TOKEN="YOUR_TOKEN"
export MCP_DISCORD_CONFIG_PATH="./config.toml"
python mcp_discord_bot.py # Reads from env vars
```

There's an `MCP_DEBUG=1` variable from upstream, but I'm not entirely sure what it shows.

# Checking MCP servers

For SSE servers, you can check with curl, e.g.

```sh
curl -N http://localhost:8901/sse
```
