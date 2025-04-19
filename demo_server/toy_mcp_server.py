# toy_mcp_server.py
#!/usr/bin/env python
'''
A simple toy MCP server for testing MCP clients.
Provides basic tools like add, echo, and a simulated long task.

# How to run:
uvicorn demo_server.toy_mcp_server:create_app --factory --host 127.0.0.1 --port 8902
'''
from __future__ import annotations

import asyncio
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from mcp.server.fastmcp import FastMCP
import mcp.server.fastmcp
from sse_server_vendored import VendoredSseServerTransport
mcp.server.fastmcp.SseServerTransport = VendoredSseServerTransport
# Vendored from MCP SDK: mcp.client.sse

import structlog
logger = structlog.get_logger(__name__)

# Had expected the FastMCP object to have .name, .version etc. but apparently not
TOY_SERVER_NAME = 'Toy MCP Server'
TOY_SERVER_VERSION = '0.0.1'
TOY_SERVER_DESCRIPTION = 'A simple MCP server with basic tools for testing'


# 1. Initialize FastMCP
#    - name, version, description are used for MCP discovery
#    - context can hold shared state, not needed for this simple example
#    - auto_mount=False means we'll mount the SSE app manually later
mcp = FastMCP(
    name=TOY_SERVER_NAME,
    version=TOY_SERVER_VERSION,
    description=TOY_SERVER_DESCRIPTION,
    context={}, # No shared context needed for these tools
    auto_mount=False
)

# 2. Define Simple MCP Tools
#    - Use the @mcp.tool() decorator
#    - Functions should be async
#    - Type hints are recommended

@mcp.tool()
async def add(a: int, b: int) -> int:
    '''Adds two integers together.'''
    logger.info('Executing `add` tool', a=a, b=b)
    result = a + b
    logger.info('Result of `add`', result=result)
    return result

@mcp.tool()
async def echo(text: str, repeat: int = 1) -> str:
    '''Echoes the provided text a specified number of times.'''
    logger.info('Executing `echo` tool', text=text, repeat=repeat)
    if repeat < 1:
        raise ValueError('Repeat count must be at least 1')
    result = ' '.join([text] * repeat)
    logger.info('Result of `echo`', result=result)
    return result

@mcp.tool()
async def long_task(delay: float = 2.0) -> str:
    '''Simulates a task that takes some time to complete.'''
    logger.info('Executing `long_task` tool', delay=delay)
    if delay < 0:
        raise ValueError('Delay cannot be negative')
    await asyncio.sleep(delay)
    result = f'Long task finished after {delay:.2f} seconds.'
    logger.info('Result of `long_task`', result=result)
    return result

@mcp.tool()
async def error_tool(message: str = 'This is a simulated error') -> None:
    '''Intentionally raises a ValueError for testing error handling.'''
    logger.info('Executing `error_tool`', message=message)
    raise ValueError(message)

# 3. Define Resource Handlers (Optional for simple tools, but good practice)
#    You could add resource handlers here if needed, similar to your gdrive example.
#    For this toy server, we'll skip them to keep it minimal.

# @mcp.resource('toy://items/{cursor}/{page_size}')
# async def list_toy_items(cursor: str | None = None, page_size: int = 10):
#     # Implementation to list dummy resources
#     pass

# @mcp.resource('toy://item/{item_id}')
# async def read_toy_item(item_id: str):
#     # Implementation to read a dummy resource
#     pass

# 4. Create the FastAPI App Factory
def create_app() -> FastAPI:
    '''Factory function for Uvicorn to create the FastAPI app.'''
    logger.info('Creating Toy FastAPI app instance via factory.')

    # Create base FastAPI app
    app = FastAPI(
        title=TOY_SERVER_NAME,
        version=TOY_SERVER_VERSION,
        description=TOY_SERVER_DESCRIPTION,
    )

    # Mount the FastMCP SSE application onto the root path
    # This makes the MCP endpoints (/mcp, /mcp/sse, /mcp/tools, etc.) available
    app.mount('/', mcp.sse_app())

    # Add CORS middleware (useful for testing from web clients)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],  # Allow all origins for simple testing
        allow_credentials=True,
        allow_methods=['*'],  # Allow all methods
        allow_headers=['*'],  # Allow all headers
    )

    # Add a simple health check endpoint (optional but good practice)
    @app.get('/health')
    async def health_check():
        return {'status': 'ok', 'service': TOY_SERVER_NAME, 'version': TOY_SERVER_VERSION}

    @app.on_event('startup')
    async def startup_event():
        logger.info('Toy MCP Server starting up.')
        logger.info(f'MCP Name: {TOY_SERVER_NAME}')
        logger.info(f'MCP Version: {TOY_SERVER_VERSION}')
        # Removed the problematic loop here. Tool discovery happens via MCP protocol.
        logger.info('MCP tools are available via the standard MCP discovery endpoints.')

    logger.info('Toy FastAPI app created with MCP routes mounted.')
    return app

# 5. Add a main block for direct execution (optional)
if __name__ == '__main__':
    # This allows running the server directly using: python toy_server.py
    # It's often better to use Uvicorn directly for production/testing,
    # but this can be convenient for quick checks.
    print('Starting server directly using uvicorn.run()...')
    print('For standard execution, use:')
    print('uvicorn toy_server:create_app --factory --host 127.0.0.1 --port 8901')
    uvicorn.run(
        'toy_server:create_app', # Point to the factory function string
        host='127.0.0.1',
        port=8901,
        factory=True, # Tell uvicorn to use the factory
        reload=True   # Enable auto-reload for development
        )
