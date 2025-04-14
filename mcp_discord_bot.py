# mcp_discord_bot.py
'''
Main launcher for MCP Discord bot. Loads config and the main cog
'''
import os
import tomllib
import logging
import sys

import fire
import discord
from discord.ext import commands
import rich.traceback # For customized tracebacks
import structlog

from mcp_cog import MCPCog

# Initialize structlog early, but configuration will happen in main
logger = structlog.get_logger()


def setup_logging(classic_tracebacks: bool = False):
    '''Configures structlog processors and rendering.'''

    # This function will be passed to ConsoleRenderer's exception_formatter
    def classic_rich_exception_formatter(event_dict):
        '''
        Formats exceptions using rich.traceback.Traceback with simplified options.

        Args:
            event_dict: The structlog event dictionary, may contain 'exc_info'.

        Returns:
            A rich.traceback.Traceback object if exc_info is present, else None.
        '''
        exc_info = event_dict.get('exc_info')
        if exc_info:
            # Instantiate Traceback *here*, when the exception data is available
            return rich.traceback.Traceback(
                exc_info=exc_info,  # Pass the actual exception info
                show_locals=False,  # Don't show local variables
                extra_lines=1,      # Show fewer surrounding code lines
                word_wrap=True      # Enable word wrapping
                # suppress=[]       # Optionally suppress frames from specific libraries
                # max_frames=...    # Optionally limit frames
            )
        return None # No exception info, nothing to format

    console_renderer_kwargs = {'colors': True}

    if classic_tracebacks:
        # Assign the *function* itself, not the result of calling it now
        console_renderer_kwargs['exception_formatter'] = classic_rich_exception_formatter
        logger.info('Configuring logging with classic (simplified) tracebacks.')
    else:
         logger.info('Configuring logging with standard (rich) tracebacks.')
         # When not classic, ConsoleRenderer uses its default (usually rich-based) exception formatting

    # Structlog and Standard Library Logging Setup
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt='iso'),
    ]

    structlog.configure(
        processors=shared_processors + [
            # Ensure exc_info is available for the formatter if logged via stdlib logging
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        # The ConsoleRenderer will now use classic_rich_exception_formatter
        # when an exception occurs *if* classic_tracebacks was True
        processor=structlog.dev.ConsoleRenderer(**console_renderer_kwargs),
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    # Clear existing handlers to avoid duplicates if reconfiguring
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO) # Or DEBUG, etc.


def load_config(config_path: str):
    '''Loads and validates the TOML configuration file.'''
    if not config_path:
        # fire might pass None or empty string if not provided and no default
        raise ValueError('Configuration file path is required.')

    try:
        with open(config_path, 'rb') as fp:
            config = tomllib.load(fp)
    except FileNotFoundError:
        logger.error(f'Configuration file not found: {config_path}')
        raise
    except tomllib.TOMLDecodeError:
        logger.error(f'Error decoding TOML file: {config_path}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error loading config: {config_path}', exc_info=True)
        raise

    # Basic Validation and Defaults
    if 'mcp' not in config:
        raise ValueError('Config missing required table (section) mcp')
    if 'llm_endpoint' not in config:
        raise ValueError('Config missing required table (section) llm_endpoint')

    # Set defaults directly in the loaded config dict
    config['llm_endpoint'].setdefault('base_url', 'http://localhost:1234/v1')
    config['llm_endpoint'].setdefault('api_key', 'lm-studio')

    logger.info(f'Configuration loaded successfully from {config_path}')
    return config


def main(
    config_path: str = None,
    discord_token: str = None,
    classic_tracebacks: bool = False # Default to fancy tracebacks
    ):
    '''Main function to launch the MCP Discord bot.'''

    # Setup Logging FIRST. Ensures subsequent logs use the correct format
    setup_logging(classic_tracebacks)

    # Argument Handling (including environment variables as fallback)
    if not discord_token:
        discord_token = os.environ.get('MCP_DISCORD_DISCORD_TOKEN')
        if not discord_token:
             raise ValueError('Discord token must be provided via --discord-token or MCP_DISCORD_DISCORD_TOKEN env var.')
        else:
             logger.info('Using Discord token from MCP_DISCORD_DISCORD_TOKEN environment variable.')

    if not config_path:
        config_path = os.environ.get('MCP_DISCORD_CONFIG_PATH')
        if not config_path:
            raise ValueError('Config path must be provided via --config-path or MCP_DISCORD_CONFIG_PATH env var.')
        else:
             logger.info(f'Using config path from MCP_DISCORD_CONFIG_PATH environment variable: {config_path}')

    try:
        config = load_config(config_path)
    except Exception:
        # Error already logged in load_config
        logger.critical('Failed to load configuration. Exiting.')
        sys.exit(1) # Exit if config fails

    logger.info('Setting up Discord bot intents...')
    intents = discord.Intents.default()
    intents.message_content = True
    intents.members = True # Ensure members intent is enabled in Discord Dev Portal too

    bot = commands.Bot(command_prefix=['!'], intents=intents, description='MCP Discord Bot')

    # Create Cog instance
    logger.info('Initializing MCPCog...')
    mcp_cog = MCPCog(bot, config)

    # Define async setup function (called by setup_hook)
    async def setup_bot():
        logger.info('Adding MCPCog to bot...')
        await bot.add_cog(mcp_cog)
        logger.info('MCP Cog added successfully')

    # Use setup_hook for async setup actions like adding cogs and syncing commands
    @bot.event
    async def setup_hook():
        logger.info('Executing setup_hook...')
        await setup_bot()
        logger.info('Syncing application commands...')
        try:
            # Sync commands globally. Specify guild=discord.Object(id=...) for faster testing sync.
            synced = await bot.tree.sync()
            logger.info(f'Synced {len(synced)} application commands.')
        except Exception as e:
            logger.exception('Failed to sync application commands.')
            # Decide if you want to continue running or exit if sync fails
        logger.info('setup_hook completed.')

    @bot.event
    async def on_ready():
        # This event runs after setup_hook is done and the bot is fully connected.
        logger.info(f'Bot logged in as {bot.user.name} (ID: {bot.user.id})')
        logger.info('Bot is ready and listening for commands.')

    logger.info('Starting bot...')
    try:
        bot.run(discord_token, log_handler=None) # Disable default discord.py logging handler if using structlog fully
    except discord.LoginFailure:
        logger.critical('Login failed: Invalid Discord token provided.')
    except Exception as e:
        logger.critical('Unexpected error running the bot.', exc_info=True)
    finally:
        # Code here runs after bot.run finishes (e.g., on bot shutdown)
        logger.info('Bot has shut down.')


if __name__ == '__main__':
    fire.Fire(main)
