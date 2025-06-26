MCP Demo Servers

This directory contains demo MCP servers that work with the refactored Discord AI Agent client using the official MCP SDK with Streamable HTTP transport.

# Servers

## 1. Toy MCP Server

### FastAPI Version (`toy_mcp_server_fastapi.py`)
A simple server with basic tools for testing:
- `add(a: int, b: int)`: Adds two integers
- `magic_8_ball(question: str)`: Returns a random Magic 8-Ball answer
- `long_task(delay: float)`: Simulates a long-running task
- `error_tool(message: str)`: Intentionally raises an error for testing

**To run:**
```bash
cd demo_server
uvicorn toy_mcp_server_fastapi:create_app --factory --host 127.0.0.1 --port 8902
```

### Flask Version (`toy_mcp_server.py`) ⭐ **Recommended**
Same functionality as FastAPI version but with proper task group initialization:
- Uses FastMCP's `run_http_async` method for proper task group initialization
- No route registration issues
- Reliable async functionality

**To run:**
```bash
cd demo_server
python toy_mcp_server.py
# Runs on port 8901 by default
```

**Configuration for Discord AI Agent:**
```toml
# config/toy.example.b4a.toml
[[context_source]]
name = "toy_server"
description = "Toy MCP server with basic tools for testing"
type = "@mcp"
connection_method = "streamable_http"
url = "http://localhost:8901"  # Flask version port
```

## 2. Local Files MCP Server

### FastAPI Version (`local_files_mcp_server_fastapi.py`)
A server that provides file system operations:
- `get_file(path: str)`: Retrieves file content
- `list_files(path: str)`: Lists files and directories

**To run:**
```bash
cd demo_server
python local_files_mcp_server_fastapi.py --directory ./my_files --port 8910
```

### Flask Version (`local_files_mcp_server.py`) ⭐ **Recommended**
Same functionality as FastAPI version but with proper task group initialization:
- Uses FastMCP's `run_http_async` method for proper task group initialization
- No route registration issues
- Reliable async functionality

**To run:**
```bash
cd demo_server
python local_files_mcp_server.py --directory ./my_files --port 8911
```

**Configuration for Discord AI Agent:**
```toml
# config/local_files.example.b4a.toml
[[context_source]]
name = "local_files"
description = "Local files MCP server for file system operations"
type = "@mcp"
connection_method = "streamable_http"
url = "http://localhost:8911"  # Flask version port
```

# Why Flask Versions?

The Flask-based servers were created to address persistent issues with FastMCP's streamable-http transport when integrated with FastAPI:

1. **Proper Task Group Initialization**: Uses FastMCP's `run_http_async` method which properly initializes the task group
2. **No Route Registration Issues**: Avoids the complex ASGI integration that was causing route registration problems
3. **Simplified Architecture**: Direct use of FastMCP's built-in HTTP server capabilities
4. **Async Support**: Maintains full async functionality through proper task group management

# Testing

You can test the servers using the test script in the main directory:

```bash
# Start the Flask-based toy server (recommended)
cd demo_server
python toy_mcp_server.py

# In another terminal, run the test
cd ..
python test_mcp_servers.py
```

# Requirements

- Python 3.12+
- Official MCP SDK (`mcp>=1.9.4`)
- FastMCP for server implementation
- Flask for Flask-based servers
- FastAPI and Uvicorn for FastAPI-based servers
- Tenacity for retry logic in the client

# Troubleshooting

1. **Connection refused**: Make sure the server is running on the correct port
2. **Tool not found**: Check that the server is properly initialized and tools are registered
3. **Timeout errors**: Increase timeout values in the connection parameters if needed
4. **Import errors**: Ensure all dependencies are installed with `uv pip install -r requirements.txt`
5. **Task group errors**: Use Flask-based servers with `run_http_async` for proper initialization 