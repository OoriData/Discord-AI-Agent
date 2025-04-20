# mcp_discord_bot.py
'''
Main launcher for MCP Discord bot. Loads config and the main cog
'''
import os
import tomllib
import logging
import sys
import io  # Keep io import if needed elsewhere, but not for the formatter below

import fire
import discord
from discord.ext import commands
import rich.traceback # Keep for potential direct use if needed later
import structlog

from mcp_cog import MCPCog

# Initialize structlog early, but configuration will happen in main
logger = structlog.get_logger()


def setup_logging(classic_tracebacks: bool = False): # Keep the argument, but ignore it for now
    '''Configures structlog processors and rendering.'''

    # Simplified: Let ConsoleRenderer handle exceptions using its default rich formatting.
    # We remove the custom exception_formatter logic for now.
    # If truly "classic" (non-rich) tracebacks are needed later,
    # we'd need to investigate the correct signature or use a different formatter.
    console_renderer_kwargs = {'colors': True}

    if classic_tracebacks:
        # Note: Default ConsoleRenderer uses rich. Revisit if non-rich is strictly required.
        logger.info('Configuring logging (classic_tracebacks=True, using default Rich exceptions).')
    else:
         logger.info('Configuring logging with standard (rich) tracebacks.')

    # Structlog and Standard Library Logging Setup
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt='iso'),
        # Add process info if desired
        # structlog.processors.add_process_info,
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
        # Use the standard ConsoleRenderer; it handles exceptions well by default
        processor=structlog.dev.ConsoleRenderer(**console_renderer_kwargs),
        # foreign_pre_chain is useful if you want stdlib logs formatted too
        # foreign_pre_chain=shared_processors, # Uncomment if needed
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    # Clear existing handlers to avoid duplicates if reconfiguring
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO) # Or DEBUG, etc.

    # Configure logging for libraries that use standard logging, if needed
    # logging.getLogger("discord").setLevel(logging.WARNING)
    # logging.getLogger("httpx").setLevel(logging.WARNING)
    # logging.getLogger("anyio").setLevel(logging.WARNING)

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
    except tomllib.TOMLDecodeError as e:
        logger.error(f'Error decoding TOML file: {config_path}', error=str(e))
        raise
    except Exception as e:
        logger.error(f'Unexpected error loading config: {config_path}', exc_info=True)
        raise

    # Basic Validation and Defaults
    if 'mcp' not in config:
        logger.warning('Config missing table (section) [mcp]. MCP features will be unavailable.')
        config['mcp'] = {'server': []} # Ensure mcp section exists even if empty
    if 'llm_endpoint' not in config:
        # Allow running without LLM if only using direct MCP calls? For now, require it.
        raise ValueError('Config missing required table (section) [llm_endpoint]')

    # Set defaults directly in the loaded config dict
    config.setdefault('llm_endpoint', {}) # Ensure section exists
    config['llm_endpoint'].setdefault('base_url', 'http://localhost:1234/v1')
    # Ensure API key is handled (required by openai>=1.0)
    if 'api_key' not in config['llm_endpoint']:
         config['llm_endpoint']['api_key'] = os.environ.get('OPENAI_API_KEY', 'lm-studio') # Default if not set

    # Validate MCP server entries
    if 'server' in config['mcp'] and isinstance(config['mcp']['server'], list):
        valid_servers = []
        for i, server in enumerate(config['mcp']['server']):
            if isinstance(server, dict) and 'name' in server and 'url' in server:
                valid_servers.append(server)
            else:
                logger.warning(f'Ignoring invalid MCP server entry at index {i} in config', entry=server)
        config['mcp']['server'] = valid_servers
    elif 'mcp' in config: # mcp section exists but no valid server list
        config['mcp']['server'] = []


    logger.info(f'Configuration loaded successfully from {config_path}')
    return config


def main(
    config_path: str = None,
    discord_token: str = None,
    classic_tracebacks: bool = False # Argument kept but currently ignored by setup_logging
    ):
    '''Main function to launch the MCP Discord bot.'''

    # Setup Logging FIRST. Ensures subsequent logs use the correct format
    setup_logging(classic_tracebacks) # Pass argument, even if ignored for now

    # Argument Handling (including environment variables as fallback)
    if not discord_token:
        discord_token = os.environ.get('MCP_DISCORD_TOKEN')
        if not discord_token:
             logger.critical('Discord token must be provided via --discord-token or MCP_DISCORD_DISCORD_TOKEN env var.')
             sys.exit(1)
        else:
             logger.info('Using Discord token from MCP_DISCORD_TOKEN environment variable.')

    if not config_path:
        config_path = os.environ.get('MCP_DISCORD_CONFIG_PATH')
        if not config_path:
            logger.critical('Config path must be provided via --config-path or MCP_DISCORD_CONFIG_PATH env var.')
            sys.exit(1)
        else:
             logger.info(f'Using config path from MCP_DISCORD_CONFIG_PATH environment variable: {config_path}')

    try:
        config = load_config(config_path)
    except Exception:
        # Error already logged in load_config
        logger.critical('Failed to load configuration. Exiting.')
        sys.exit(1) # Exit if config fails

    logger.info('Setting up Discord bot intents…')
    intents = discord.Intents.default()
    intents.message_content = True
    intents.members = True # Ensure members intent is enabled in Discord Dev Portal too

    # Consider adding presences intent if needed for user status checks, etc.
    # intents.presences = False

    bot = commands.Bot(command_prefix=['!'], intents=intents, description='MCP Discord Bot')

    # Create Cog instance
    logger.info('Initializing MCPCog…')
    try:
        mcp_cog = MCPCog(bot, config)
    except Exception as e:
        logger.critical(f'Failed to initialize MCPCog: {e}', exc_info=True)
        sys.exit(1)


    # Define async setup function (called by setup_hook)
    async def setup_bot():
        logger.info('Adding MCPCog to bot…')
        await bot.add_cog(mcp_cog)
        logger.info('MCP Cog added successfully')

    # Use setup_hook for async setup actions like adding cogs and syncing commands
    @bot.event
    async def setup_hook():
        logger.info('Executing setup_hook…')
        await setup_bot()
        logger.info('Syncing application commands…')
        try:
            # Sync commands globally. Specify guild=discord.Object(id=…) for faster testing sync.
            # guild_id = os.environ.get('DISCORD_DEV_GUILD_ID') # Example for guild-specific sync
            # if guild_id:
            #    guild = discord.Object(id=int(guild_id))
            #    synced = await bot.tree.sync(guild=guild)
            #    logger.info(f'Synced {len(synced)} application commands to guild {guild_id}.')
            # else:
            synced = await bot.tree.sync()
            logger.info(f'Synced {len(synced)} application commands globally.')
        except Exception as e:
            logger.exception('Failed to sync application commands.')
            # Decide if you want to continue running or exit if sync fails
        logger.info('setup_hook completed.')

    @bot.event
    async def on_ready():
        # This event runs after setup_hook is done and the bot is fully connected.
        logger.info(f'Bot logged in as {bot.user.name} (ID: {bot.user.id})')
        logger.info('Bot is ready and listening for commands.')
        # Optionally set bot presence
        # await bot.change_presence(activity=discord.Activity(type=discord.ActivityType.listening, name="!chat or /chat"))


    logger.info('Starting bot…')
    try:
        # Disable default discord.py logging handler if using structlog fully
        # Set log_handler=None if you want structlog to be the *only* handler.
        # Set log_handler=handler if you want discord.py logs routed through your structlog handler.
        # Let's keep it None to rely solely on the root logger setup above.
        bot.run(discord_token, log_handler=None, log_level=logging.INFO) # Match root logger level
    except discord.LoginFailure:
        logger.critical('Login failed: Invalid Discord token provided.')
    except discord.PrivilegedIntentsRequired:
        logger.critical(f"Privileged intents (Members{', Presence' if intents.presences else ''}) are required but not enabled in the Discord Developer Portal.")
    except Exception as e:
        logger.critical('Unexpected error running the bot.', exc_info=True)
    finally:
        # Code here runs after bot.run finishes (e.g., on bot shutdown Ctrl+C)
        # asyncio cleanup is handled internally by bot.run() closing the loop
        logger.info('Bot shutdown sequence initiated.')


if __name__ == '__main__':
    # Setup rich traceback hook for uncaught exceptions BEFORE fire.Fire
    rich.traceback.install(show_locals=False, extra_lines=1, word_wrap=True)
    fire.Fire(main)
