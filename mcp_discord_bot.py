# mcp_discord_bot.py
'''
Main launcher for MCP Discord bot. Loads config (including B4A sources) and the main cog.
'''
import os
import tomllib
import logging
import sys
import io
from typing import Dict, Any # Added for type hint

import fire
import discord
from discord.ext import commands
import rich.traceback
import structlog

from mcp_cog import MCPCog
# Import the B4A loader function and loader class
from b4a_config import load_b4a, B4ALoader

logger = structlog.get_logger()

# setup_logging function remains the same...
def setup_logging(classic_tracebacks: bool = False):
    '''Configures structlog processors and rendering.'''
    console_renderer_kwargs = {'colors': True}
    if classic_tracebacks:
        logger.info('Configuring logging (classic_tracebacks=True, using default Rich exceptions).')
    else:
         logger.info('Configuring logging with standard (rich) tracebacks.')

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt='iso'),
    ]

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer(**console_renderer_kwargs),
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)


# Modified load_config - NO LONGER returns B4A results directly
def load_main_config(config_path: str) -> Dict[str, Any]:
    '''Loads ONLY the main TOML configuration file.'''
    if not config_path:
        raise ValueError('Configuration file path is required.')

    try:
        with open(config_path, 'rb') as fp:
            config = tomllib.load(fp)
    except FileNotFoundError:
        logger.error(f'Main configuration file not found: {config_path}')
        raise
    except tomllib.TOMLDecodeError as e:
        logger.error(f'Error decoding main TOML file: {config_path}', error=str(e))
        raise
    except Exception as e:
        logger.error(f'Unexpected error loading main config: {config_path}', exc_info=True)
        raise

    # --- Basic Validation for Sections Needed by the Bot ---
    # B4A sources are checked inside the B4ALoader
    if 'llm_endpoint' not in config:
        logger.warning("Config missing section '[llm_endpoint]'. LLM features may be limited.")
        config['llm_endpoint'] = {}
    if 'model_params' not in config:
        logger.info("Config missing section '[model_params]'. Using default LLM parameters.")
        config['model_params'] = {}
    # [mcp] section is NO LONGER directly used for connections

    # Validate/Default LLM Endpoint config (remains important for the agent itself)
    config.setdefault('llm_endpoint', {})
    config['llm_endpoint'].setdefault('base_url', 'http://localhost:1234/v1')
    if 'api_key' not in config['llm_endpoint']:
         # Resolve directly here or let b4a_config handle env vars? Let b4a handle $VAR style.
         config['llm_endpoint']['api_key'] = os.environ.get('OPENAI_API_KEY', 'lm-studio') # Keep default/env fallback
         logger.info('Using default or environment variable for LLM API key.', key_source='env/default' if config['llm_endpoint']['api_key'] == 'lm-studio' else 'config/env')

    # Validate/Default Model Params
    config.setdefault('model_params', {})
    if not isinstance(config['model_params'], dict):
        logger.warning("Config section '[model_params]' is not a dictionary. Resetting.", invalid_value=config['model_params'])
        config['model_params'] = {}

    logger.info(f'Main configuration loaded successfully from {config_path}')
    return config


def main(
    config_path: str = None,
    discord_token: str = None,
    classic_tracebacks: bool = False
    ):
    '''Main function to launch the MCP Discord bot.'''

    setup_logging(classic_tracebacks)

    # --- Argument Handling (Token & Config Path) ---
    if not discord_token:
        discord_token = os.environ.get('MCP_DISCORD_TOKEN')
        if not discord_token:
             logger.critical('Discord token must be provided via --discord-token or MCP_DISCORD_TOKEN env var.')
             sys.exit(1)
        else: logger.info('Using Discord token from MCP_DISCORD_TOKEN environment variable.')

    if not config_path:
        config_path = os.environ.get('MCP_DISCORD_CONFIG_PATH')
        if not config_path:
            logger.critical('Config path must be provided via --config-path or MCP_DISCORD_CONFIG_PATH env var.')
            sys.exit(1)
        else: logger.info(f'Using config path from MCP_DISCORD_CONFIG_PATH env var: {config_path}')

    # --- Configuration Loading ---
    try:
        # 1. Load the main config file
        main_config = load_main_config(config_path)
        config_directory = os.path.dirname(config_path) # Needed for relative paths

        # 2. Load B4A sources using the loader
        b4a_data: B4ALoader = load_b4a(main_config, config_directory)
        # b4a_data now holds .mcp_sources, .rss_sources, .load_errors etc.

    except Exception:
        logger.critical('Failed during configuration loading phase. Exiting.', exc_info=True)
        sys.exit(1)

    # --- Bot Setup ---
    logger.info('Setting up Discord bot intents…')
    intents = discord.Intents.default()
    intents.message_content = True
    intents.members = True

    bot = commands.Bot(command_prefix=['!'], intents=intents, description='MCP Discord Bot')

    logger.info('Initializing MCPCog…')
    try:
        # Pass the main config AND the loaded B4A data to the Cog
        mcp_cog = MCPCog(bot, main_config, b4a_data)
    except Exception as e:
        logger.critical(f'Failed to initialize MCPCog: {e}', exc_info=True)
        sys.exit(1)

    # --- Async Setup (Cog loading, Command Sync) ---
    async def setup_bot():
        logger.info('Adding MCPCog to bot…')
        await bot.add_cog(mcp_cog)
        # TODO: Add other cogs here if needed (e.g., an RSSCog)
        logger.info('MCP Cog added successfully')

    @bot.event
    async def setup_hook():
        logger.info('Executing setup_hook…')
        await setup_bot()
        logger.info('Syncing application commands…')
        try:
            synced = await bot.tree.sync()
            logger.info(f'Synced {len(synced)} application commands globally.')
        except Exception as e:
            logger.exception('Failed to sync application commands.')
        logger.info('setup_hook completed.')

    @bot.event
    async def on_ready():
        logger.info(f'Bot logged in as {bot.user.name} (ID: {bot.user.id})')
        logger.info('Bot is ready and listening for commands.')

    # --- Bot Run ---
    logger.info('Starting bot…')
    try:
        bot.run(discord_token, log_handler=None, log_level=logging.INFO)
    except discord.LoginFailure: logger.critical('Login failed: Invalid Discord token.')
    except discord.PrivilegedIntentsRequired: logger.critical("Privileged intents required but not enabled.")
    except Exception as e: logger.critical('Unexpected error running the bot.', exc_info=True)
    finally: logger.info('Bot shutdown sequence initiated.')


if __name__ == '__main__':
    rich.traceback.install(show_locals=False, extra_lines=1, word_wrap=True)
    fire.Fire(main)
