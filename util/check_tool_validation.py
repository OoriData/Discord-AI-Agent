#!/usr/bin/env python3
# util/check_tool_validation.py
'''
Utility to probe remote MCP tools via FastMCP, specialized for the tools in toy_mcp_server.py
'''

from fastmcp import FastMCP, Client
from fastmcp.tools import Tool

async def test_tool_validation():
    '''Test the tool validation logic from mcp_cog.py'''
    
    # Create a FastMCP instance and add some tools (similar to toy_mcp_server.py)
    mcp = FastMCP(
        name='test',
        version='1.0',
        instructions='test server'
    )
    
    @mcp.tool()
    def add(a: int, b: int) -> int:
        '''Adds two integers together.'''
        return a + b
    
    @mcp.tool()
    def magic_8_ball(question: str = 'Your question') -> str:
        '''Consult the Magic 8-Ball for a glimpse into the future!'''
        return 'It is certain.'
    
    # Start the FastMCP server
    import asyncio
    import uvicorn
    from fastapi import FastAPI
    
    # Create a FastAPI app and mount the FastMCP server
    app = FastAPI()
    app.mount('/mcp', mcp.streamable_http_app)
    
    # Start the server in the background
    config = uvicorn.Config(app, host='127.0.0.1', port=8903, log_level='error')
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())
    
    # Wait a moment for the server to start
    await asyncio.sleep(1)
    
    # Connect to the server using the Client (same as the Discord bot)
    client = Client('http://127.0.0.1:8901/mcp')
    
    # Get the tools using list_tools (same as the Discord bot)
    async with client:
        tools = await client.list_tools()
    
    # Stop the server
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass
    
    print(f'Found {len(tools)} tools')
    
    # Test the validation logic from mcp_cog.py
    for i, tool in enumerate(tools):
        print(f'\nTool {i}:')
        print(f'  Type: {type(tool)}')
        print(f'  Is Tool instance: {isinstance(tool, Tool)}')
        print(f'  Has name attribute: {hasattr(tool, 'name')}')
        print(f'  Name: {getattr(tool, 'name', 'NO_NAME')}')
        print(f'  Title: {getattr(tool, 'title', 'NO_TITLE')}')
        print(f'  Description: {getattr(tool, 'description', 'NO_DESC')}')
        
        # Apply the validation logic from mcp_cog.py
        if not (isinstance(tool, Tool) or hasattr(tool, 'name')):
            print('  ❌ Skipping non-Tool object')
            continue
        
        if not hasattr(tool, 'name') or not tool.name:
            print('  ❌ Skipping nameless Tool')
            continue
        
        print('  ✅ Tool passed validation')

if __name__ == '__main__':
    import asyncio
    asyncio.run(test_tool_validation())
