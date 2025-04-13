'''
Main launcher for MCP Discord bot. Loads config and the main cog
'''
# import os
# import asyncio
import tomllib
import discord
from discord.ext import commands
import click

from mcp_cog import MCPCog

import structlog
logger = structlog.get_logger()


def load_config(ctx, param, value):
    '''
    Callback for processing --config option
    '''
    if not value:
        return None

    with open(value, 'rb') as fp:
        config = tomllib.load(fp)

    # Validate and set defaults for config
    assert 'mcp' in config, 'Config missing required table (section) mcp'
    assert 'llm_endpoint' in config, 'Config missing required table (section) llm_endpoint'
    config['llm_endpoint'].setdefault('base_url', 'http://localhost:1234/v1')
    config['llm_endpoint'].setdefault('api_key', 'lm-studio')

    return config

@click.command()
@click.option('--discord-token', required=True, help='Discord app token')
@click.option('--config', '-c', required=True, type=click.Path(dir_okay=False), callback=load_config,
              help='TOML Config file')
def cli(discord_token, config):
    '''Launch the MCP Discord bot.'''
    # Set up bot with intents
    intents = discord.Intents.default()
    intents.message_content = True
    intents.members = True

    bot = commands.Bot(command_prefix=['!'], intents=intents, description='MCP Discord Bot')

    mcp_cog = MCPCog(bot, config)

    async def setup_bot():
        await bot.add_cog(mcp_cog)
        logger.info('MCP Cog loaded successfully')

    @bot.event
    async def setup_hook():
        await setup_bot()
        await bot.tree.sync()
        logger.info('App commands synced successfully')

    bot.run(discord_token)


if __name__ == '__main__':
    cli(auto_envvar_prefix='MCP_DISCORD')
