# mcp_discord_bot.py
'''
Main launcher for MCP Discord bot. Loads config (including B4A sources) and the main cog.

See README.md for info on how to run
'''
import os
import tomllib
import logging
import sys
import io
from typing import Any
from pathlib import Path

# Save original exception hook before any imports that might install Rich tracebacks
_original_excepthook = sys.excepthook

import fire
import discord
from discord.ext import commands
import rich.traceback
import structlog
from sentence_transformers import SentenceTransformer

from mcp_cog import MCPCog
# Import the B4A loader function and loader class
from b4a_config import load_b4a, B4ALoader
from discord_aiagent import assemble_pgvector_config

logger = structlog.get_logger()

# List all loggers
# for name in logging.root.manager.loggerDict:
#     print(name)

# List of known loggers with too much chatter at debug level
TAME_LOGGERS = ['asyncio', 'httpcore', 'httpx', 'discord', 'aiohttp', 'urllib3', 'torch', 'requests'
                'openai', 'sentence_transformers', 'transformers', 'asyncpg',
                'mcp']
for tl in TAME_LOGGERS:
    logging.getLogger(tl).setLevel(logging.WARNING)


def setup_logging(classic_tracebacks: bool = False, log_level_str: str = 'INFO'):
    '''Configures structlog processors and rendering.'''
    if classic_tracebacks:
        # Use a simple but readable renderer that doesn't use Rich
        renderer = structlog.dev.ConsoleRenderer(colors=False, exception_formatter=structlog.dev.plain_traceback)
    else:
        # Use Rich renderer for enhanced output
        console_renderer_kwargs = {'colors': True}
        renderer = structlog.dev.ConsoleRenderer(**console_renderer_kwargs)

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
        processor=renderer,
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    # Convert string level name to logging constant (e.g., 'DEBUG' -> logging.DEBUG)
    # Default to INFO if the level is invalid
    numeric_level = getattr(logging, log_level_str.upper(), None)
    if not isinstance(numeric_level, int):
        # Use initial logger config which might still be INFO level before this point
        logging.warning(f"Invalid log level '{log_level_str}' provided. Defaulting to INFO.")
        numeric_level = logging.INFO
        log_level_str = 'INFO'  # Ensure we log the level actually being set

    # Log the level being set *before* actually setting it on the root logger
    # This message will appear if the *current* level allows INFO messages.
    logger.info(f"Setting root logger level to: {log_level_str.upper()}")
    root_logger.setLevel(numeric_level)


def load_main_config(config_path: Path) -> dict[str, Any]:
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

    # Basic Validation for Sections Needed by the Bot
    # B4A sources are checked inside the B4ALoader
    if 'llm_endpoint' not in config:
        logger.warning("Config missing section '[llm_endpoint]'. LLM features may be limited.")
        config['llm_endpoint'] = {}
    if 'model_params' not in config:
        logger.info("Config missing section '[model_params]'. Using default LLM parameters.")
        config['model_params'] = {}

    # Validate/Default LLM Endpoint config (remains important for the agent itself)
    config.setdefault('llm_endpoint', {})
    config['llm_endpoint'].setdefault('base_url', 'http://localhost:1234/v1')
    if 'api_key' not in config['llm_endpoint']:
         # Resolve directly here or let b4a_config handle env vars? Let b4a handle $VAR style.
         config['llm_endpoint']['api_key'] = os.environ.get('OPENAI_API_KEY', 'lm-studio')  # Keep default/env fallback
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
    classic_tracebacks: bool = False,
    loglevel: str = 'INFO'
    ):
    '''Main function to launch the MCP Discord bot.'''

    # Priority: CLI > Environment Variable > Default
    env_loglevel = os.environ.get('AIBOT_LOGLEVEL')
    effective_loglevel = loglevel  # Start with CLI arg (or its default 'INFO')

    log_source = "default"
    if loglevel != 'INFO':  # CLI was explicitly set to something other than default
        log_source = "CLI argument (--loglevel)"
    elif env_loglevel:  # CLI was default ('INFO'), but ENV var is set
        effective_loglevel = env_loglevel
        log_source = "environment variable (AIBOT_LOGLEVEL)"

    # Call setup_logging *before* other logging, passing the determined level
    # Use upper case for consistency
    effective_loglevel_upper = effective_loglevel.upper()
    setup_logging(classic_tracebacks=classic_tracebacks, log_level_str=effective_loglevel_upper)
    
    # Configure traceback handling based on classic_tracebacks option
    if classic_tracebacks:
        logger.info('Using classic Python tracebacks (Rich tracebacks disabled).')
        # Restore the original exception hook to disable Rich tracebacks
        sys.excepthook = _original_excepthook
        logger.debug('Restored original exception hook.')
    else:
        logger.info('Installing Rich tracebacks for enhanced error display.')
        rich.traceback.install(show_locals=False, extra_lines=1, word_wrap=True)
    
    # Log the source *after* setup_logging might have changed the level
    if log_source != "default":
        logger.info(f"Log level '{effective_loglevel_upper}' set via {log_source}.")
 
    # Argument Handling (Token & Config Path)
    if not discord_token or discord_token is True:  # If user accidentally does --discord-token alone, it resolves to True
        discord_token = os.environ.get('AIBOT_DISCORD_TOKEN')
        if not discord_token:
             logger.critical('Discord token must be provided via --discord-token or AIBOT_DISCORD_TOKEN env var.')
             sys.exit(1)
        else: logger.info('Using Discord token from AIBOT_DISCORD_TOKEN environment variable.')

    if not config_path:
        config_path = os.environ.get('AIBOT_DISCORD_CONFIG_PATH')
        if not config_path:
            logger.critical('Config path must be provided via --config-path or AIBOT_DISCORD_CONFIG_PATH env var.')
            sys.exit(1)
        else: logger.info(f'Using config path from AIBOT_DISCORD_CONFIG_PATH env var: {config_path}')

    # Configuration Loading
    try:
        # Load the main config file. Use pathlib. Look for a file named main.toml within the config_path directory.
        config_path = Path(config_path)
        if not config_path.exists():
            logger.critical(f'Config path does not exist: {config_path}')
            sys.exit(1)

        main_config_fpath = config_path / 'main.toml'
        main_config = load_main_config(main_config_fpath)

        # Load B4A sources using the loader
        b4a_data: B4ALoader = load_b4a(main_config, config_path)
        # b4a_data now holds .mcp_sources, .rss_sources, .load_errors etc.

    except Exception:
        logger.critical('Failed during configuration loading phase. Exiting.', exc_info=True)
        sys.exit(1)

    logger.info('Setting up Discord bot intents…')
    intents = discord.Intents.default()
    intents.message_content = True
    intents.members = True

    bot = commands.Bot(command_prefix=['!'], intents=intents, description='MCP Discord Bot')

    pgvector_config = assemble_pgvector_config()
    #     try:
    #         pgvector_config = assemble_pgvector_config()
    #     except ImportError:
    #         logger.error('PGVector enabled, but required libraries (ogbujipt, sentence_transformers, asyncpg, pgvector?) are not installed. Disabling.')
    #         pgvector_enabled = False
    #         pgvector_config = {'enabled': False}
    #     except Exception as e:
    #         logger.exception('Error initializing PGVector configuration or loading embedding model. Disabling PGVector history.')
    #         pgvector_enabled = False
    #         pgvector_config = {'enabled': False}
    # else:
    #     logger.info('PGVector history is DISABLED.')
    #     pgvector_config = {'enabled': False}

    logger.info('Initializing MCPCog…')
    try:
        # Pass main config, loaded B4A data & chat history DB config
        mcp_cog = MCPCog(bot, main_config, b4a_data, pgvector_config)
    except Exception as e:
        logger.critical(f'Failed to initialize MCPCog: {e}', exc_info=True)
        sys.exit(1)

    # Async Setup (Cog loading, Command Sync)
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

    logger.info('Starting bot…')
    try:
        # Convert the effective level string back to a logging constant for discord.py
        discord_log_level = getattr(logging, effective_loglevel_upper, logging.INFO)
        if not isinstance(discord_log_level, int):  # Safety check
            discord_log_level = logging.INFO

        bot.run(discord_token, log_handler=None, log_level=discord_log_level)  # Use dynamic level
    except discord.LoginFailure:
        logger.critical('Login failed: Invalid Discord token.')
    except discord.PrivilegedIntentsRequired:
        logger.critical("Privileged intents required but not enabled.")
    except Exception as e:
        logger.critical('Unexpected error running the bot.', exc_info=True)
    finally:
        logger.info('Bot shutdown sequence initiated.')


if __name__ == '__main__':
    fire.Fire(main)
