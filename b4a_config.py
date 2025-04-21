# b4a_config.py
'''
Handles loading and processing of B4A source definitions (.b4a.toml files).
Discovers sources listed in the main config, parses them, and uses type-specific
handlers to prepare configurations for different integrations (MCP, RSS, etc.).
'''
import os
import tomllib
from typing import List, Dict, Any, Optional, Tuple, Union
import structlog
import re
import dataclasses # Using dataclasses for structured config

logger = structlog.get_logger(__name__)

# Data Classes for Structured Config

@dataclasses.dataclass
class MCPConfig:
    '''Configuration pulled from an @mcp B4A source.'''
    name: str
    url: str
    # Add auth fields later if needed (e.g., token_env_var)

@dataclasses.dataclass
class RSSConfig:
    '''Configuration pulled from an @rss B4A source.'''
    name: str
    url: str
    refresh_interval: Optional[str] = None
    max_items: Optional[int] = None

B4ASourceConfig = Union[  # For hinting
    MCPConfig,
    RSSConfig
]

# Environment Variable Resolution
# XXX: Use google.re2 to avoid backtracking attachs?
ENV_VAR_PATTERN = re.compile(r'^\$([a-zA-Z_][a-zA-Z0-9_]*)$')

def resolve_value(value: Any) -> Any:
    '''Resolves values, checking for $ENV_VAR patterns.'''
    if isinstance(value, str):
        match = ENV_VAR_PATTERN.match(value)
        if match:
            var_name = match.group(1)
            env_value = os.environ.get(var_name)
            if env_value is None:
                logger.warning(f'Environment variable `{var_name}` requested in config not found.', requested_var=var_name)
                return None # Or raise error, or return the pattern? Returning None for now.
            logger.debug(f'Resolved environment variable `{var_name}`.')
            return env_value
    # Recursively resolve in lists and dicts if needed in future
    # elif isinstance(value, list):
    #     return [resolve_value(item) for item in value]
    # elif isinstance(value, dict):
    #     return {k: resolve_value(v) for k, v in value.items()}
    return value


class B4ALoader:
    '''Loads and processes B4A configuration sources.'''
    def __init__(self):
        # Store results keyed by type
        self.mcp_sources: List[MCPConfig] = []
        self.rss_sources: List[RSSConfig] = []
        # Add lists for other source types as handlers are implemented
        self.unhandled_sources: List[Dict[str, Any]] = []
        self.load_errors: List[Tuple[str, str]] = [] # (filepath, error_message)

    def load_and_process_sources(self, main_config: Dict[str, Any], config_dir: str):
        '''
        Main entry point. Finds source files in main_config, loads, and processes them.
        config_dir: The directory containing the main config file, used for resolving relative paths.
        '''
        logger.info('Starting B4A source loading process...')
        self.mcp_sources = []
        self.rss_sources = []
        self.unhandled_sources = []
        self.load_errors = []

        # Find source file list (assuming [b4a].sources)
        b4a_section = main_config.get('b4a', {})
        source_files_relative = b4a_section.get('sources', [])

        if not isinstance(source_files_relative, list):
            logger.error('Main config `[b4a].sources` is not a list. Cannot load B4A sources.', config_sources=source_files_relative)
            return # Cannot proceed

        if not source_files_relative:
            logger.info('No B4A sources listed in `[b4a].sources`.')
            return # Nothing to load

        logger.info(f'Found {len(source_files_relative)} B4A source files listed.', sources=source_files_relative)

        # Iterate and process each source file
        for rel_path in source_files_relative:
            if not isinstance(rel_path, str):
                logger.warning('Ignoring non-string entry in B4A sources list.', entry=rel_path)
                continue

            abs_path = os.path.abspath(os.path.join(config_dir, rel_path))
            logger.debug(f'Attempting to load B4A source file.', relative_path=rel_path, absolute_path=abs_path)

            try:
                with open(abs_path, 'rb') as fp:
                    source_config = tomllib.load(fp)
            except FileNotFoundError:
                err_msg = f'B4A source file not found: {abs_path}'
                logger.error(err_msg, filepath=abs_path)
                self.load_errors.append((abs_path, err_msg))
                continue
            except tomllib.TOMLDecodeError as e:
                err_msg = f'Error decoding TOML in B4A source file: {abs_path} - {e}'
                logger.error(err_msg, filepath=abs_path, error=str(e))
                self.load_errors.append((abs_path, err_msg))
                continue
            except Exception as e:
                err_msg = f'Unexpected error loading B4A source file: {abs_path} - {e}'
                logger.exception(err_msg, filepath=abs_path) # Use exception for full trace
                self.load_errors.append((abs_path, err_msg))
                continue

            self._process_single_source(source_config, abs_path)

        logger.info('B4A source loading finished.',
                    mcp_loaded=len(self.mcp_sources),
                    rss_loaded=len(self.rss_sources),
                    unhandled=len(self.unhandled_sources),
                    errors=len(self.load_errors))

    def _process_single_source(self, config: Dict[str, Any], filepath: str):
        '''Determines the type and calls the appropriate handler.'''
        source_type = config.get('type')
        source_name = config.get('name', os.path.splitext(os.path.basename(filepath))[0]) # Default name from filename

        if not source_type or not isinstance(source_type, str):
            err_msg = f'B4A source file missing or invalid `type` field.'
            logger.error(err_msg, filepath=filepath, config_preview=dict(list(config.items())[:3])) # Log first few items
            self.load_errors.append((filepath, err_msg))
            return

        logger.info(f'Processing B4A source.', name=source_name, type=source_type, filepath=filepath)

        # Resolve values in the config dict before passing to handlers
        resolved_config = {k: resolve_value(v) for k, v in config.items()}

        # Call handlers based on type
        if source_type == '@mcp':
            self._handle_mcp_source(resolved_config, source_name, filepath)
        elif source_type == '@rss':
            self._handle_rss_source(resolved_config, source_name, filepath)
        # Add elif for other types like .openapi, .file, etc.
        else:
            logger.warning(f'No handler found for B4A source type `{source_type}`. Storing as unhandled.', name=source_name, type=source_type, filepath=filepath)
            self.unhandled_sources.append({**resolved_config, '_filepath': filepath, '_sourcename': source_name}) # Store with context

    # Type-specific handlers

    def _handle_mcp_source(self, config: Dict[str, Any], name: str, filepath: str):
        '''Handles type = '@mcp' sources.'''
        logger.debug('Handling .mcp source.', name=name, filepath=filepath)
        mcp_url = config.get('url')
        # Example: Add connection_method, auth handling later based on spec details
        # connection_method = config.get('connection_method', 'default') # e.g., http_sse, grpc?
        # auth_config = config.get('auth', {}) # e.g., {'type': 'token', 'token': '$MCP_TOKEN'}

        if not mcp_url or not isinstance(mcp_url, str):
            err_msg = f'`@mcp` source `{name}` is missing required `url` field or it\'s not a string.'
            logger.error(err_msg, filepath=filepath)
            self.load_errors.append((filepath, err_msg))
            return

        # Basic validation passed, create MCPConfig object
        mcp_conf = MCPConfig(
            name=name,
            url=mcp_url
            # Add other extracted/validated fields here
        )
        self.mcp_sources.append(mcp_conf)
        logger.info(f'Successfully processed .mcp source: `{name}`.', url=mcp_url)

    def _handle_rss_source(self, config: Dict[str, Any], name: str, filepath: str):
        '''Handles type = '@rss' sources.'''
        logger.debug('Handling .rss source.', name=name, filepath=filepath)

        rss_url = config.get('url')
        if not rss_url or not isinstance(rss_url, str):
            err_msg = f'`@rss` source `{name}` is missing required `url` field or it\'s not a string.'
            logger.error(err_msg, filepath=filepath)
            self.load_errors.append((filepath, err_msg))
            return

        # Extract optional fields
        refresh = config.get('refresh_interval')
        max_items = config.get('max_items')
        validated_max_items: Optional[int] = None
        if max_items is not None:
            try: validated_max_items = int(max_items)
            except (ValueError, TypeError):
                 logger.warning(f'`@rss` source `{name}` has invalid `max_items` value. Ignoring.', value=max_items, filepath=filepath)

        # Basic validation passed, create RSSConfig object
        rss_conf = RSSConfig(
            name=name,
            url=rss_url,
            refresh_interval=str(refresh) if refresh else None,
            max_items=validated_max_items
        )
        self.rss_sources.append(rss_conf)
        logger.info(f'Successfully processed .rss source: `{name}`.', url=rss_url, refresh=rss_conf.refresh_interval, max_items=rss_conf.max_items)

        # TODO: Implement actual RSS fetching/tool registration logic
        # Could use feedparser, possibly schedule updates & decide how to expose the data (e.g., register a simple tool
        # in the MCPCog or another cog). For now, just store the config.
        # Example placeholder:
        # try:
        #     feed_data = feedparser.parse(rss_url)
        #     logger.info(f'Successfully fetched initial data for RSS feed '{name}'. Found {len(feed_data.entries)} entries.', url=rss_url)
        # except Exception as e:
        #     logger.error(f'Error fetching initial data for RSS feed '{name}'.', url=rss_url, error=e)


def load_b4a(main_config: Dict[str, Any], config_dir: str) -> B4ALoader:
    '''Loads B4A sources based on the main config and returns the loader object.'''
    loader = B4ALoader()
    loader.load_and_process_sources(main_config, config_dir)
    return loader
