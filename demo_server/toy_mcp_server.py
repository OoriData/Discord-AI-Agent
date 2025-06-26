# toy_mcp_server.py
'''
A clean, direct FastMCP server for testing MCP clients.
This version uses FastMCP's built-in HTTP transport without web framework complications.
'''

import asyncio
import random
from fastmcp import FastMCP

import structlog
logger = structlog.get_logger(__name__)

# Simple debug logging setup
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="%H:%M:%S"),
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

# Set the root logger to DEBUG level
import logging
logging.basicConfig(level=logging.DEBUG)

# Server metadata
TOY_SERVER_NAME = 'Toy MCP Server'
TOY_SERVER_VERSION = '0.0.4'
TOY_SERVER_DESCRIPTION = 'A clean FastMCP server with basic tools for testing'
TOY_SERVER_PORT = 8901

# Initialize FastMCP
mcp = FastMCP(
    name=TOY_SERVER_NAME,
    version=TOY_SERVER_VERSION,
    instructions=TOY_SERVER_DESCRIPTION,
)

@mcp.tool()
async def add(a: int, b: int) -> int:
    '''Adds two integers together.'''
    logger.info('Executing `add` tool', a=a, b=b)
    result = a + b
    logger.info('Result of `add`', result=result)
    return result

@mcp.tool()
async def magic_8_ball(question: str = 'Your question') -> str:
    '''Consult the Magic 8-Ball for a glimpse into the future! (Question is ignored).'''
    logger.info('Executing `magic_8_ball` tool', question=question)
    answers = [
        'It is certain.', 'It is decidedly so.', 'Without a doubt.', 'Yes – definitely.',
        'You may rely on it.', 'As I see it, yes.', 'Most likely.', 'Outlook good.',
        'Yes.', 'Signs point to yes.', 'Reply hazy, try again.', 'Ask again later.',
        'Better not tell you now.', 'Cannot predict now.', 'Concentrate and ask again.',
        'Don\'t count on it.', 'My reply is no.', 'My sources say no.', 'Outlook not so good.',
        'Very doubtful.'
    ]
    result = random.choice(answers)
    logger.info('Result of `magic_8_ball`', answer=result)
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

async def main():
    '''Main async function to run the MCP server.'''
    logger.info('Starting FastMCP server...')
    logger.info(f'MCP Name: {TOY_SERVER_NAME}')
    logger.info(f'MCP Version: {TOY_SERVER_VERSION}')
    logger.info(f'Server will be available at: http://127.0.0.1:{TOY_SERVER_PORT}')
    
    # Run FastMCP with streamable HTTP transport
    await mcp.run_async(
        transport='streamable-http',
        host='127.0.0.1',
        port=TOY_SERVER_PORT
    )

if __name__ == '__main__':
    # Run the async main function
    asyncio.run(main()) 