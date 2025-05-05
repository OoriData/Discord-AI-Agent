# SPDX-FileCopyrightText: 2025-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# mcp_cog.py
'''
Discord cog for MCP integration using mcp-sse-client. Connects to MCP servers
and provides commands to interact with MCP tools via an LLM.
'''
import os
import json
import asyncio
from typing import Any
import uuid
# import time
# import traceback

import discord
from discord import app_commands, abc as discord_abc
from discord.ext import commands

import structlog
# Openai types might be useful for clarity
from ogbujipt.embedding.pgvector import MessageDB

from mcp_sse_client import MCPClient, ToolDef, ToolInvocationResult, ToolParameter
from b4a_config import B4ALoader, MCPConfig, resolve_value
from source_handlers import FEEDPARSER_AVAILABLE

# XXX: Clean up exception handling & imports by better study of mcp-sse-client
# Needed for ClosedResourceError/BrokenResourceError handling if not imported via sse/mcp
# MCPClient might raise its own connection errors or underlying httpx/anyio errors
# We'll catch broader exceptions initially and refine if needed.
try:
    from anyio import ClosedResourceError, BrokenResourceError, WouldBlock
except ImportError:
    class ClosedResourceError(Exception): pass
    class BrokenResourceError(Exception): pass
    class WouldBlock(Exception): pass
# Add httpx errors which MCPClient might expose
try:
    import httpx
    ConnectError = httpx.ConnectError
    ReadTimeout = httpx.ReadTimeout
except ImportError:
    class ConnectError(IOError): pass
    class ReadTimeout(TimeoutError): pass

from discord_aiagent.rssutil import RSSAgent
from discord_aiagent.discordutil import send_long_message, handle_attachments
from discord_aiagent.openaiutil import OpenAILLMWrapper, MissingLLMResponseError, LLMResponseError, ToolCallExecutionError


logger = structlog.get_logger(__name__)

# Define a constant namespace UUID for generating channel-specific history keys
# Generated one-time by Uche using uuid.uuid4()
DISCORD_CHANNEL_HISTORY_NAMESPACE = uuid.UUID('faa9e9c4-1f01-4e27-933d-6460b8924924')

ALLOWED_FOR_LLM_INIT = ['base_url', 'api_key'] # Adjusted for openai > 1.0
ALLOWED_FOR_LLM_CHAT = ['model', 'temperature', 'max_tokens', 'top_p'] # Added common chat params

# FIXME: Use Word Loom
DEFAULT_SYSTEM_MESSAGE = 'You are a helpful AI assistant. When you need to use a tool, you MUST use the provided function-calling mechanism. Do NOT simply describe the tool call in your text response.'

# API Key Handling - ensure OPENAI_API_KEY is checked/set if needed by the AsyncOpenAI client
# The logic in __init__ already handles this using the config/env var.

os.environ['TOKENIZERS_PARALLELISM'] = 'false' # Disable parallelism for tokenizers to avoid warnings


class MCPCog(commands.Cog, RSSAgent):
    def __init__(self, bot: commands.Bot, config: dict[str, Any], b4a_data: B4ALoader, pgvector_config: dict[str, Any]):
        RSSAgent.__init__(self, b4a_data, {}, 0) # Initialize RSSAgent with empty cache
        self.bot = bot
        self.config = config  # Agent config (LLM, etc.)
        self.mcp_connections: dict[str, MCPClient] = {}  # Keyed by MCPConfig.name
        self.mcp_tools: dict[str, list[ToolDef]] = {}  # Keyed by MCPConfig.name
        self.message_history: dict[int, list[dict[str, Any]]] = {}

        self._connection_tasks: dict[str, asyncio.Task] = {}
        self._shutdown_event = asyncio.Event()

        self.pgvector_config = pgvector_config
        self.pgvector_db: MessageDB | None = None
        self.pgvector_enabled = self.pgvector_config.get('enabled', False)

        self._rss_feed_cache: dict[str, tuple[float, Any]] = {}
        self._rss_cache_ttl: int = self.config.get('rss_cache_ttl', 300) # Default 5 mins, configurable in agent TOML

        # LLM Client and Parameter Initialization
        llm_endpoint_config = self.config.get('llm_endpoint', {})
        model_params_config = self.config.get('model_params', {})

        # Initialize LLM Client
        llm_init_params = {k: resolve_value(v) for k, v in llm_endpoint_config.items() if k in ALLOWED_FOR_LLM_INIT}
        if 'api_key' not in llm_init_params:
            llm_init_params['api_key'] = 'missing-key' # Or handle error
        if not llm_init_params.get('base_url'):
            raise ValueError('LLM \'base_url\' is required.')

        # Configure LLM Chat Parameters (Combine defaults, model_params, llm_endpoint)
        llm_chat_params = {
             'model': llm_endpoint_config.get('label', 'local-model'), # Default model name
             'temperature': 0.3, 'max_tokens': 1000, # Base defaults
        }
        # Apply overrides from [model_params] first
        model_param_overrides = {k: resolve_value(v) for k, v in model_params_config.items() if k in ALLOWED_FOR_LLM_CHAT}
        llm_chat_params.update(model_param_overrides)
        # Apply any *other* allowed chat params found in [llm_endpoint]
        endpoint_chat_overrides = {k: resolve_value(v) for k, v in llm_endpoint_config.items() if k in ALLOWED_FOR_LLM_CHAT and k not in model_param_overrides}
        llm_chat_params.update(endpoint_chat_overrides)
        logger.info('LLM chat parameters configured', **llm_chat_params)

        # Configure System Message
        sysmsg = resolve_value(llm_endpoint_config.get('sysmsg', DEFAULT_SYSTEM_MESSAGE))
        sys_postscript = resolve_value(llm_endpoint_config.get('sys_postscript', ''))
        self.system_message = sysmsg + ('\n\n' + sys_postscript if sys_postscript else '')
        logger.info('System message configured.', sysmsg_len=len(sysmsg), postscript_len=len(sys_postscript))
        logger.debug(f'Full system message: {self.system_message}') # Debug level for full message

        try:
            self.llm_client = OpenAILLMWrapper(llm_init_params, llm_chat_params, self)
            logger.info('AsyncOpenAI client initialized.', **llm_init_params)
        except Exception as e:
            logger.exception('Failed to initialize OpenAI client', config=llm_init_params)
            raise e

        # Log B4A Loading Summary
        if self.b4a_data.load_errors:
            logger.warning('Errors occurred during B4A source loading.', count=len(self.b4a_data.load_errors), errors=self.b4a_data.load_errors)
        logger.info('MCP Cog initialized',
            mcp_sources_found=len(self.b4a_data.mcp_sources),
            rss_sources_found=len(self.b4a_data.rss_sources),
            pgvector_enabled=self.pgvector_enabled,
            rss_handler_available=FEEDPARSER_AVAILABLE)

    async def _manage_mcp_connection(self, mcp_config: MCPConfig):
        '''Persistent task to manage a single MCP connection using MCPClient.'''
        url = mcp_config.url
        name = mcp_config.name # Use the name from B4A config
        reconnect_delay = 15 # Initial delay seconds
        max_reconnect_delay = 300 # Max delay seconds

        while not self._shutdown_event.is_set():
            logger.info(f'Attempting connection for MCP source: \'{name}\'', url=url)
            client: MCPClient | None = None
            try:
                # TODO: Add auth handling here based on mcp_config fields if needed
                client = MCPClient(url)
                logger.info(f'MCPClient instance created for \'{name}\'. listing tools…')

                # list Tools acts as connection check & fetches tool info
                # Will likely handle the underlying SSE connection setup and MCP handshake implicitly
                # Add timeout? Check whether MCPClient handles that, but add ours, for now
                tools_list: list[ToolDef] = await asyncio.wait_for(client.list_tools(), timeout=45.0)

                self.mcp_connections[name] = client
                self.mcp_tools[name] = tools_list
                logger.info(f'Connected to MCP source \'{name}\' and listed {len(tools_list)} tools.', server_name=name)
                reconnect_delay = 15 # Reset delay on success

                # Connection Maintenance (MCPClient handles underlying keep-alive/SSE)
                # Just need to wait until shutdown or an error occurs during a tool call.
                # Task can potentially sleep or periodically check status if MCPClient provides a method.
                # For now, rely on execute_tool to detect issues. Wait for shutdown signal.
                await self._shutdown_event.wait()
                logger.info(f'Shutdown signal received for {name}. Ending management task.')
                # Loop will terminate because _shutdown_event is set.

            except (ConnectError, ReadTimeout, ConnectionRefusedError) as conn_err:
                 logger.warning(f'Connection failed for MCP server {name}: {type(conn_err).__name__}. Retrying…', server_name=name, error=str(conn_err))
            except asyncio.TimeoutError:
                logger.error(f'Timeout connecting or listing tools for MCP server {name}. Retrying…', server_name=name)
            except asyncio.CancelledError:
                logger.info(f'Connection task for {name} cancelled.')
                break # Exit loop immediately
            except Exception as e:
                # Catch other errors from MCPClient init or list_tools
                logger.exception(f'Unexpected error managing connection for {name}. Retrying…', server_name=name, url=url)
                # Optionally log traceback: logger.error('Traceback:', exc_info=True)

            # Cleanup before Reconnect/Shutdown
            logger.debug(f'Cleaning up resources for {name} before retry/shutdown.')
            if name in self.mcp_connections:
                # Does MCPClient need explicit closing? Check its implementation.
                # Assuming no explicit close needed for now, just remove reference.
                del self.mcp_connections[name]
            if name in self.mcp_tools:
                del self.mcp_tools[name]
            client = None  # Ensure client object is cleared

            # Wait before Retrying (if not shutting down)
            if not self._shutdown_event.is_set():
                logger.info(f'Waiting {reconnect_delay}s before reconnecting to {name}.')
                try:
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=reconnect_delay)
                    logger.info(f'Shutdown signaled during reconnect delay for {name}.')
                    break # Exit loop if shutdown signaled
                except asyncio.TimeoutError:
                    reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay) # Exponential backoff
                except asyncio.CancelledError:
                     logger.info(f'Reconnect wait for {name} cancelled.')
                     break # Exit loop

        # Final Task Cleanup (on shutdown or cancellation)
        logger.info(f'Connection management task for {name} finished cleanly.')
        if name in self.mcp_connections: del self.mcp_connections[name]
        if name in self.mcp_tools: del self.mcp_tools[name]
        client = None

    async def cog_load(self):
        '''
        Set up MCP connections when the cog is loaded by starting persistent tasks.
        '''
        logger.info('Loading MCPCog and starting connection managers…')
        self._shutdown_event.clear() # Ensure shutdown is not set initially

        if self.pgvector_enabled and self.pgvector_db is None:
            logger.info('Initializing PGVector MessageDB connection...')
            try:
                if self.pgvector_config['conn_str']:
                    # Use connection string if provided
                    self.pgvector_db = await MessageDB.from_conn_string(
                        self.pgvector_config['conn_str'],
                        self.pgvector_config['embedding_model'],
                        self.pgvector_config['table_name'],
                        # stringify_json=False
                    )
                else:
                    self.pgvector_db = await MessageDB.from_conn_params(
                        embedding_model=self.pgvector_config['embedding_model'],
                        table_name=self.pgvector_config['table_name'],
                        db_name=self.pgvector_config['db_name'],
                        host=self.pgvector_config['host'],
                        port=self.pgvector_config['port'],
                        user=self.pgvector_config['user'],
                        password=self.pgvector_config['password']
                    )
                try:
                    # Might not be running as superuser, so don't assume table creation is possible
                    # await self.pgvector_db.create_table()
                    logger.info(f'PGVector table `{self.pgvector_config['table_name']}` ensured.')
                except Exception as table_err:
                    # Log error, but maybe don't disable PGVector entirely if table exists
                    logger.exception(f'Error ensuring PGVector table exists (it might already exist): {table_err}')

                logger.info('PGVector MessageDB connection successful.')
            except Exception as e:
                logger.exception('Failed to initialize PGVector MessageDB connection during cog_load. Disabling PGVector.')
                self.pgvector_enabled = False
                self.pgvector_db = None

        mcp_source_configs = self.b4a_data.mcp_sources # Get sources from B4A data
        if not mcp_source_configs:
            logger.warning('No valid @mcp B4A sources loaded. No MCP connections will be established.')
            return

        active_task_count = 0
        for mcp_conf in mcp_source_configs: # Iterate B4A sources
            name = mcp_conf.name
            if name not in self._connection_tasks or self._connection_tasks[name].done():
                logger.info(f'Creating connection task for B4A MCP source: \'{name}\'', url=mcp_conf.url)
                # Pass the MCPConfig object
                task = asyncio.create_task(self._manage_mcp_connection(mcp_conf), name=f'mcp_conn_{name}')
                self._connection_tasks[name] = task
                active_task_count += 1
            else:
                logger.info(f'Connection task for B4A MCP source \'{name}\' already running.')
                active_task_count += 1

        logger.info(f'MCPCog load processed. {active_task_count} connection tasks are active or starting.')

    async def cog_unload(self):
        '''
        Clean up MCP connections when the cog is unloaded by stopping tasks.
        '''
        # XXX Shouldn't be eny PGVector Cleanup needed, but maybe set self.pgvector_db to None?
        logger.info('Unloading MCPCog and stopping connection tasks…')
        self._shutdown_event.set() # Signal all tasks to stop their loops

        tasks_to_wait_for = list(self._connection_tasks.values())

        if tasks_to_wait_for:
            logger.info(f'Waiting for {len(tasks_to_wait_for)} MCP connection tasks to complete shutdown…')
            # Wait for tasks to finish processing the shutdown signal
            # Give them a bit more time as they might be in a wait/retry loop
            done, pending = await asyncio.wait(tasks_to_wait_for, timeout=15.0)

            if pending:
                logger.warning(f'{len(pending)} connection tasks did not shut down cleanly within timeout. Attempting cancellation.')
                for task in pending:
                    task_name = task.get_name() if hasattr(task, 'get_name') else 'unknown task'
                    logger.debug(f'Cancelling task: {task_name}')
                    task.cancel()
                    try:
                        # Allow cancellation to propagate briefly
                        await asyncio.wait_for(task, timeout=1.0)
                    except asyncio.CancelledError:
                        logger.debug(f'Task {task_name} confirmed cancelled.')
                    except asyncio.TimeoutError:
                        logger.warning(f'Task {task_name} did not respond to cancellation quickly.')
                    except Exception as e:
                         logger.error(f'Error during task cancellation for {task_name}', exc_info=e)

            else:
                 logger.info('All connection tasks shut down gracefully.')
        else:
             logger.info('No active connection tasks to stop.')

        # Clear stored data regardless of task shutdown status
        self.mcp_connections.clear()
        self.mcp_tools.clear()
        self._connection_tasks.clear() # Clear task references
        self._rss_feed_cache.clear()
        logger.info('MCPCog unloaded and connection resources cleared.')

    async def _get_channel_history(self, channel_id: int) -> list[dict[str, Any]]:
        '''
        Get message history for a channel. Prioritizes PGVector if enabled,
        falls back to in-memory cache.
        '''
        # Generate deterministic UUIDv5 from channel ID
        history_key = uuid.uuid5(DISCORD_CHANNEL_HISTORY_NAMESPACE, str(channel_id))

        if self.pgvector_enabled and self.pgvector_db:
            logger.debug(f'Retrieving history from PGVector for key: {history_key}')
            try:
                # Retrieve messages from PGVector
                # FIXME: Set limit in config
                pg_messages_gen = await self.pgvector_db.get_messages(history_key=history_key, limit=20)
                # Convert generator to list and reverse it to get chronological order
                pg_messages = list(pg_messages_gen)
                pg_messages.reverse() # Reverse to chronological

                # Format for LLM (ensure 'role' and 'content' keys)
                formatted_history = []
                # Check if system message needs to be prepended manually or if it's stored
                # Assuming it IS stored as the first message during insertion.
                # If not, prepend: formatted_history = [{'role': 'system', 'content': self.system_message}]
                # import pprint; pprint.pprint(pg_messages)
                for msg in pg_messages:
                    formatted_history.append({'role': msg['role'], 'content': msg['content']})

                logger.debug(f'Retrieved {len(formatted_history)} messages from PGVector for key {history_key}')
                # Optional: Update in-memory cache as well? Or rely solely on DB?
                # For simplicity, let's just return the DB result.
                # self.message_history[channel_id] = formatted_history # Update cache
                return formatted_history
            except Exception as e:
                logger.exception(f'Error retrieving history from PGVector for key {history_key}. Falling back to in-memory.')
                # Fall through to use in-memory cache on error

        # Fallback to in-memory cache if PGVector disabled, errored, or not initialized
        logger.debug(f'Using in-memory history cache for channel {channel_id}')
        max_history_len = self.config.get('max_history_length', 20)
        if channel_id not in self.message_history:
            self.message_history[channel_id] = [
                {'role': 'system', 'content': self.system_message}
            ]
            logger.info(f'Initialized in-memory message history for channel {channel_id}')
        else:
            current_len = len(self.message_history[channel_id])
            target_len = max_history_len + 1 # Keep system message + N history items
            if current_len > target_len:
                amount_to_remove = current_len - target_len
                self.message_history[channel_id] = [self.message_history[channel_id][0]] + self.message_history[channel_id][amount_to_remove + 1:]
                logger.debug(f'Trimmed in-memory message history for channel {channel_id}, removed {amount_to_remove} messages.')

        return self.message_history[channel_id]

    async def execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> ToolInvocationResult | dict[str, str]:
        '''
        Execute an MCP tool or the RSS query tool.
        Returns ToolInvocationResult for MCP tools, or a dict for RSS/errors.
        '''
        logger.debug('Executing tool', tool_name=tool_name, tool_input=tool_input)

        if tool_name == 'query_rss_feed':
            if not FEEDPARSER_AVAILABLE:
                return {'error': 'RSS querying is disabled because the `feedparser` library is not installed.'}
            return await self.execute_rss_query(tool_input)

        mcp_client: MCPClient | None = None
        server_name_found: str | None = None  # B4A source name

        # Find the server/client (keyed by B4A source name) that has the tool
        for source_name, client in self.mcp_connections.items():
            # Check if the tool name exists in the list of tools for this server
            if source_name in self.mcp_tools and any(tool.name == tool_name for tool in self.mcp_tools[source_name]):
                mcp_client = client
                server_name_found = source_name
                logger.debug(f'Found active client for tool `{tool_name}` on MCP source \'{server_name_found}\'.')
                break

        if not mcp_client or not server_name_found:
            logger.warning(f'MCP Tool `{tool_name}` not found on any *actively connected* MCP source.')
            # Provide more specific error based on B4A data
            if not self.b4a_data.mcp_sources:
                return {'error': f'Tool `{tool_name}` cannot be executed: No @mcp B4A sources configured.'}
            if not self.mcp_connections:
                return {'error': f'Tool `{tool_name}` cannot be executed: No MCP sources currently connected.'}
            # Check if tool exists on a configured but disconnected source
            origin_source = next((s.name for s in self.b4a_data.mcp_sources if s.name in self.mcp_tools and any(t.name == tool_name for t in self.mcp_tools[s.name])), None)
            if origin_source and origin_source not in self.mcp_connections:
                return {'error': f'Tool `{tool_name}` exists (on MCP source \'{origin_source}\'), but it\'s currently disconnected.'}
            return {'error': f'Tool `{tool_name}` not found on any configured and connected MCP source.'}

        logger.info(f'Calling MCP tool `{tool_name}` on MCP server `{server_name_found}`')
        try:
            # Call the tool using MCPClient's invoke_tool method
            # Add a timeout for the specific tool call.
            result: ToolInvocationResult = await asyncio.wait_for(
                mcp_client.invoke_tool(tool_name, tool_input),
                timeout=120.0  # FIXME: Make configurable
            )
            logger.debug(f'Tool `{tool_name}` executed via MCPClient.', result_content=result.content, error_code=result.error_code)

            # Check for tool-specific errors reported by MCPClient/Server
            # Check specifically for non-zero error codes. Treat None and 0 as success.
            if result.error_code is not None and result.error_code != 0:
                logger.warning(f'Tool `{tool_name}` executed but reported an error.', error_code=result.error_code, content=result.content)
                # Format the error content for the user/LLM
                error_message = f'Tool reported error code {result.error_code}'
                if result.content: # Include content if provided with the error
                     error_message += f': {str(result.content)}'
                return {'error': error_message} # Return standard error dict

            # If error_code is None or 0, it's a success
            return result # Return the successful ToolInvocationResult object

        except asyncio.TimeoutError:
             logger.error(f'Timeout calling MCP tool `{tool_name}` on server `{server_name_found}`', tool_input=tool_input)
             # Attempt to trigger reconnect by removing client? Or let manager handle?
             # Let manager handle based on future failures.
             return {'error': f'Tool call `{tool_name}` timed out.'}
        except (ConnectError, ReadTimeout, BrokenResourceError, ClosedResourceError, ConnectionError, WouldBlock) as conn_err:
            # Catch connection-related errors during the tool call
            logger.error(f'Connection error during MCP tool call `{tool_name}` on server `{server_name_found}`: {type(conn_err).__name__}', tool_input=tool_input, error=str(conn_err))
            # Remove the client reference; the management task will handle reconnection.
            if server_name_found in self.mcp_connections: del self.mcp_connections[server_name_found]
            if server_name_found in self.mcp_tools: del self.mcp_tools[server_name_found] # Also clear tools
            return {'error': f'Connection error executing MCP tool `{tool_name}`. The server may be temporarily unavailable. Please try again.'}
        except Exception as e:
            logger.exception(f'Unexpected error calling tool `{tool_name}` on server `{server_name_found}`', tool_input=tool_input)
            return {'error': f'Error calling tool `{tool_name}`: {str(e)}'}

    def _map_mcp_type_to_json_type(self, mcp_type: str) -> str:
        '''Maps MCP parameter types to JSON Schema types.'''
        # Based on mcp-sse-client ToolParameter.parameter_type which seems to be string
        type_map = {
            'string': 'string',
            'integer': 'integer',
            'number': 'number',
            'boolean': 'boolean',
            'array': 'array', # Assuming direct mapping for array/object might need refinement
            'object': 'object',
            # Add other mappings if MCP uses different type names
        }
        json_type = type_map.get(mcp_type.lower(), 'string') # Default to string if unknown
        if json_type != mcp_type.lower():
            logger.debug(f'Mapped MCP type `{mcp_type}` to JSON type `{json_type}`.')
        return json_type

    async def format_tools_for_openai(self) -> list[dict[str, Any]]:
        '''
        Format active MCP tools and the RSS query tool for OpenAI API.
        '''
        openai_tools = []
        active_server_names = list(self.mcp_connections.keys())

        for server_name in active_server_names:
            if server_name not in self.mcp_tools:
                logger.warning(f'Server `{server_name}` is connected but has no tools listed. Skipping for OpenAI format.', server_name=server_name)
                continue

            tool_defs: list[ToolDef] = self.mcp_tools[server_name]
            logger.debug(f'Formatting {len(tool_defs)} MCP tools from active server `{server_name}` for OpenAI.')

            for tool in tool_defs:
                # Validate ToolDef structure (basic check)
                if not isinstance(tool, ToolDef) or not hasattr(tool, 'name') or not tool.name:
                    logger.warning(f'Skipping malformed/nameless ToolDef from server `{server_name}`', tool_data=tool)
                    continue

                # Build JSON schema for parameters
                properties = {}
                required_params = []
                if isinstance(tool.parameters, list):
                    for param in tool.parameters:
                        if not isinstance(param, ToolParameter) or not hasattr(param, 'name') or not param.name:
                            logger.warning(f'Skipping malformed parameter in tool `{tool.name}`', param_data=param)
                            continue

                        param_schema = {
                            'type': self._map_mcp_type_to_json_type(param.parameter_type),
                            'description': param.description or f'Parameter {param.name}' # Add default desc
                        }
                        # Add enum if present (assuming param.enum is a list)
                        if hasattr(param, 'enum') and isinstance(param.enum, list) and param.enum:
                           param_schema['enum'] = param.enum

                        # Add default if present (assuming param.default exists)
                        if hasattr(param, 'default') and param.default is not None:
                            param_schema['default'] = param.default

                        # Handle arrays (needs 'items' schema) - Basic handling
                        if param_schema['type'] == 'array':
                            # Default to array of strings if item type not specified
                            # MCP ToolParameter needs more info for complex types
                            param_schema['items'] = {'type': 'string'}
                            logger.debug(f'Using default `items: string` for array param `{param.name}` in tool `{tool.name}`. Specify item type in MCP if needed.')

                        # Handle objects (needs 'properties' schema) - Basic handling
                        if param_schema['type'] == 'object':
                            # Need more schema info from MCP ToolParameter for object properties
                            param_schema['properties'] = {}
                            logger.debug(f'Using default `properties: {{}}` for object param `{param.name}` in tool `{tool.name}`. Specify properties in MCP if needed.')


                        properties[param.name] = param_schema

                        # Check for required flag (assuming param.required is a boolean)
                        if hasattr(param, 'required') and param.required:
                            required_params.append(param.name)

                # Final JSON Schema for the tool
                parameters_schema = {
                    'type': 'object',
                    'properties': properties,
                }
                if required_params:
                    parameters_schema['required'] = required_params

                # OpenAI Tool structure
                openai_tool = {
                    'type': 'function',
                    'function': {
                        'name': tool.name,
                        'description': tool.description or f'Executes the {tool.name} tool.', # Default desc
                        'parameters': parameters_schema,
                    },
                }
                openai_tools.append(openai_tool)

        # Format RSS Query Tool
        if FEEDPARSER_AVAILABLE and self.b4a_data.rss_sources:
            rss_feed_names = [conf.name for conf in self.b4a_data.rss_sources]
            if rss_feed_names:
                logger.debug(f'Formatting RSS query tool for {len(rss_feed_names)} feeds.')
                rss_tool_def = {
                    'type': 'function',
                    'function': {
                        'name': 'query_rss_feed',
                        'description': 'Retrieves the latest entries from a configured RSS feed. Can optionally filter entries by a query string in the title or summary.',
                        'parameters': {
                            'type': 'object',
                            'properties': {
                                'feed_name': {
                                    'type': 'string',
                                    'description': 'The name of the configured RSS feed to query.',
                                    'enum': rss_feed_names # Use loaded feed names
                                },
                                'query': {
                                    'type': 'string',
                                    'description': 'Optional: A keyword or phrase to search for in entry titles or summaries.'
                                },
                                'limit': {
                                    'type': 'integer',
                                    'description': 'Optional: The maximum number of entries to return (default is 5).'
                                }
                            },
                            'required': ['feed_name'] # Only feed_name is strictly required
                        }
                    }
                }
                openai_tools.append(rss_tool_def)
            else:
                 logger.debug('No RSS feed names found, skipping RSS tool formatting.')
        elif not FEEDPARSER_AVAILABLE and self.b4a_data.rss_sources:
             logger.warning('RSS sources configured but `feedparser` not installed. RSS tool disabled.')

        return openai_tools

    async def _handle_chat_logic(
        self,
        sendable: commands.Context | discord.Interaction | discord_abc.Messageable,
        message: str,
        channel_id: int,
        user_id: int,
        stream: bool,
        attachments: list[discord.Attachment] | None = None
    ):
        '''
        Core logic for handling chat interactions, LLM calls, and tool execution.
        Uses PGVector/in-memory history and handles MCP/RSS tools.
        '''
        # Determine if we need to use followup/ephemeral (only for Interactions)
        is_interaction = isinstance(sendable, discord.Interaction)
        send_followup = is_interaction and sendable.response.is_done()

        # Ephemeral only makes sense for interactions
        can_be_ephemeral = is_interaction
        processed_message = message # Start with the original text message

        history_key = uuid.uuid5(DISCORD_CHANNEL_HISTORY_NAMESPACE, str(channel_id)) # Generate UUID key once

        if attachments:
            logger.info(f'Processing {len(attachments)} attachments for channel {channel_id}', user_id=user_id)
            processed_message = await handle_attachments(attachments, message)

        try:
            # Get and update history
            channel_history = await self._get_channel_history(channel_id)

            # If history was just initialized (only contains system message) and PGVector is on, store system message.
            # Do we want to store the system message in the DB?
            # if self.pgvector_enabled and self.pgvector_db and len(channel_history) == 1 and channel_history[0]['role'] == 'system':
            #      try:
            #          logger.debug(f'Storing initial system message to PGVector for key: {history_key}')
            #          await self.pgvector_db.insert(
            #              history_key=history_key,
            #              role='system',
            #              content=channel_history[0]['content'],
            #              metadata={'user_id': 'system'} # Example metadata
            #          )
            #      except Exception as e:
            #          logger.exception(f'Failed to store initial system message to PGVector for key {history_key}.')

            user_message_dict = {'role': 'user', 'content': processed_message}
            channel_history.append(user_message_dict)
            logger.debug(f'User message added to history for channel {channel_id}')

            if self.pgvector_enabled and self.pgvector_db:
                try:
                    logger.debug(f'Storing user message to PGVector for key: {history_key}')
                    metadata = {'user_id': str(user_id), 'discord_role': 'user'}
                    await self.pgvector_db.insert(
                        history_key=history_key,
                        role='user', # Store standard role
                        content=processed_message,
                        metadata=metadata
                    )
                except Exception as e:
                    logger.exception(f'Failed to store user message to PGVector for key {history_key}.')

            openai_tools = await self.format_tools_for_openai() # Gets both MCP and RSS tools
            if openai_tools:
                extra_chat_params = {
                    'tools': openai_tools,
                    'tool_choice': 'auto'
                }
                logger.debug(f'Including {len(openai_tools)} tools in LLM call for channel {channel_id}')
            else:
                extra_chat_params = {}
                logger.debug(f'No active tools available for LLM call for channel {channel_id}')

            try:
                initial_response_content, tool_calls_aggregated = await self.llm_client(channel_history, str(user_id), extra_chat_params, stream=stream)
            except MissingLLMResponseError | LLMResponseError as llm_err:
                await send_long_message(sendable, llm_err.message, followup=send_followup, ephemeral=can_be_ephemeral)
                return
            except Exception as e: # Catch other unexpected errors (e.g., network)
                logger.exception('Unexpected error during LLM communication', channel_id=channel_id, user_id=user_id)
                await send_long_message(sendable, f'⚠️ Unexpected error communicating with AI: {str(e)}', followup=send_followup, ephemeral=can_be_ephemeral)
                return

            # Send Initial Response & Update History
            assistant_message_dict: dict[str, Any] = {'role': 'assistant', 'content': None}
            sent_initial_message = False
            if initial_response_content.strip():
                logger.debug(f'Sending initial LLM response to channel {channel_id}', length=len(initial_response_content), stream=stream)
                await send_long_message(sendable, initial_response_content, followup=send_followup)
                assistant_message_dict['content'] = initial_response_content # Ephemeral handled by send_long_message
                sent_initial_message = True
                send_followup = True # Any subsequent messages MUST be followups

            if tool_calls_aggregated:
                assistant_message_dict['tool_calls'] = tool_calls_aggregated
                logger.info(f'LLM requested {len(tool_calls_aggregated)} tool call(s) for channel {channel_id}', tool_names=[tc['function']['name'] for tc in tool_calls_aggregated], stream=stream)

            if assistant_message_dict.get('content') or assistant_message_dict.get('tool_calls'):
                channel_history.append(assistant_message_dict)
                sent_initial_message = True # Flag if we added *something* from the assistant

                # PGVector Insertion: Assistant Message / Tool Call Request
                if self.pgvector_enabled and self.pgvector_db:
                    try:
                        logger.debug(f'Storing assistant message/tool request to PGVector for key: {history_key}')

                        # Ensure content is not None before insertion
                        db_content = assistant_message_dict.get('content')
                        content_to_insert = db_content if db_content is not None else ''

                        metadata = {'user_id': str(self.bot.user.id), 'discord_role': 'assistant'}
                        if assistant_message_dict.get('tool_calls'):
                            metadata['tool_calls'] = assistant_message_dict['tool_calls'] # Store tool calls in metadata

                        await self.pgvector_db.insert(
                            history_key=history_key,
                            role='assistant', # Standard role
                            content=content_to_insert,
                            metadata=metadata
                        )
                    except Exception as e:
                        logger.exception(f'Failed to store assistant message/tool request to PGVector for key {history_key}.')

            elif not sent_initial_message:
                logger.info(f'LLM call finished for channel {channel_id} with no text/tools.', stream=stream)
                await send_long_message(sendable, 'I received your message but didn\'t have anything specific to add or do.', followup=send_followup, ephemeral=can_be_ephemeral)
                return

            # Process Tool Calls
            async def tool_call_result_hook(tool_name, tool_call_id, tool_result_content_formatted, exc=None):
                '''
                Hook to handle tool call results and errors. Called for each tool call in the aggregated list.
                '''
                nonlocal send_followup
                send_followup = True
                if exc:
                    await send_long_message(sendable, f'⚠️ Error during tool call `{tool_name}` execution: {str(exc)}', followup=send_followup, ephemeral=can_be_ephemeral)
                else:
                    await send_long_message(sendable, f'```Tool Call: {tool_name}\nResult:\n{tool_result_content_formatted}```', followup=True)

                # PGVector Insertion: Tool Result
                if self.pgvector_enabled and self.pgvector_db:
                    try:
                        logger.debug(f'Storing tool result ({tool_name}) to PGVector for key: {history_key}')
                        metadata = {'user_id': 'tool_executor', 'discord_role': 'tool', 'tool_name': tool_name, 'tool_call_id': tool_call_id}
                        await self.pgvector_db.insert(
                            history_key=history_key,
                            role='tool',
                            content=tool_result_content_formatted, # Store formatted result
                            metadata=metadata
                        )
                    except Exception as e:
                        logger.exception(f'Failed to store tool result ({tool_name}) to PGVector for key {history_key}.')

            if tool_calls_aggregated:
                try:
                    follow_up_text = await self.llm_client.process_tool_calls(tool_calls_aggregated, channel_history, channel_id, str(user_id), tool_call_result_hook, stream)
                    # Send Follow-up & Update History
                    if follow_up_text.strip():
                        logger.debug(f'Sending follow-up LLM response to channel {channel_id}', length=len(follow_up_text), stream=stream)
                        await send_long_message(sendable, follow_up_text, followup=True) # Must be followup
                        follow_up_dict = {'role': 'assistant', 'content': follow_up_text}
                        channel_history.append(follow_up_dict)

                        if self.pgvector_enabled and self.pgvector_db:
                            try:
                                logger.debug(f'Storing follow-up assistant message to PGVector for key: {history_key}')
                                metadata = {'user_id': str(self.bot.user.id), 'discord_role': 'assistant'}
                                await self.pgvector_db.insert(
                                    history_key=history_key,
                                    role='assistant',
                                    content=follow_up_text,
                                    metadata=metadata
                                )
                            except Exception as e:
                                logger.exception(f'Failed to store follow-up assistant message to PGVector for key {history_key}.')

                    else:
                        logger.info(f'LLM follow-up call finished for channel {channel_id} with no text content.', stream=stream)

                except MissingLLMResponseError as llm_err:
                    await send_long_message(sendable, llm_err.message, followup=True, ephemeral=can_be_ephemeral)
                    return
                except Exception as e:
                    await send_long_message(sendable, f'⚠️ Unexpected error getting follow-up AI response: {str(e)}', followup=True, ephemeral=can_be_ephemeral)
                    return

        except Exception as e:
            logger.exception(f'Unhandled error during chat logic execution for channel {channel_id}', user_id=user_id, stream=stream)
            try:
                error_message = 'An unexpected error occurred while processing your request. Please check the logs.'
                # Send ephemeral only if it's an interaction
                await send_long_message(sendable, error_message, followup=send_followup, ephemeral=is_interaction)

            except Exception as send_err:
                logger.error('Failed to send error message back to user after main chat logic failure', send_error=send_err)

    # Event Listeners

    @commands.Cog.listener()
    async def on_ready(self):
        ''' Cog ready listener. Connection tasks are started in cog_load. '''
        logger.info(f'Cog {self.__class__.__name__} is ready.')
        # Log PGVector status after cog is ready and init attempted
        if self.pgvector_config.get('enabled', False):
            if self.pgvector_enabled: logger.info('PGVector history persistence is active.')
            else: logger.warning('PGVector history persistence was configured but failed to initialize. History disabled.')
        else: logger.info('Using in-memory chat history.')
        # Log RSS status
        if self.b4a_data.rss_sources:
            if FEEDPARSER_AVAILABLE: logger.info('RSS query tool is available.')
            else: logger.warning('RSS sources configured but `feedparser` not installed. RSS tool disabled.')

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        ''' Listener to handle direct messages (DMs) to the bot. '''
        # Ignore messages from bots (including self)
        if message.author.bot:
            return

        # Only process messages in DMs (where message.guild is None)
        if message.guild is not None:
            return

        # Ignore messages that start with typical command prefixes
        # to avoid potential conflicts if user accidentally uses one in DM
        # You might adjust this list based on other bots or expected usage
        common_prefixes = ['!', '/', '$', '%', '?', '.', ',']
        if any(message.content.startswith(p) for p in common_prefixes):
            # Optionally, you could inform the user prefixes aren't needed in DMs
            # await message.channel.send('You don't need a prefix when talking to me in DMs!')
            logger.debug(f'Ignoring DM from {message.author} starting with a common prefix: {message.content[:10]}...')
            return

        # Check for empty message content AND no attachments
        if not message.content.strip() and not message.attachments:
            logger.debug(f'Ignoring empty DM from {message.author}')
            # Avoid sending 'provide a message' error for empty DMs, just ignore.
            return

        logger.info(f'Received DM from {message.author} ({message.author.id}) in channel {message.channel.id}')

        # Use typing context manager for user feedback in the DM channel
        async with message.channel.typing():
            await self._handle_chat_logic(
                sendable=message.channel, # Pass the DMChannel directly
                message=message.content,
                channel_id=message.channel.id,
                user_id=message.author.id,
                stream=False, # Defaulting DMs to non-streaming for simplicity
                attachments=message.attachments
            )
    # Discord Commands (mcp_list, chat_command, chat_slash)

    @app_commands.command(name='context_info', description='list configured B4A (model context) sources and their status')
    async def context_info_slash(self, interaction: discord.Interaction):
        ''' Command to list configured B4A sources and MCP connection status. '''
        logger.info(f'Command \'/context_info\' invoked by {interaction.user}')
        await interaction.response.defer(thinking=True, ephemeral=False)

        message = '**B4A Sources Status:**\n\n'

        # MCP Sources
        message += f'**MCP Sources (@mcp): {len(self.b4a_data.mcp_sources)} configured**\n'
        if not self.b4a_data.mcp_sources: message += '  *None configured or loaded.*\n'
        for mcp_conf in self.b4a_data.mcp_sources:
            name = mcp_conf.name
            status_icon = '❓'
            status_text = 'Unknown'
            if name in self.mcp_connections and name in self.mcp_tools:
                status_icon = '🟢'; status_text = f'Connected ({len(self.mcp_tools[name])} tools)'
            elif name in self.mcp_connections:
                status_icon = '🟡'; status_text = 'Connected (Tool list issue?)'
            elif name in self._connection_tasks and not self._connection_tasks[name].done():
                status_icon = '🟠'; status_text = 'Connecting…'
            else:
                status_icon = '🔴'; status_text = 'Disconnected / Failed' # Assumes task finished if not connecting
            message += f'- **{name}**: {status_icon} {status_text} ({mcp_conf.url})\n'
        message += '\n'

        # RSS Sources
        # Update status based on FEEDPARSER_AVAILABLE
        message += f'**RSS Sources (@rss): {len(self.b4a_data.rss_sources)} configured**\n'
        if self.b4a_data.rss_sources:
            for rss_conf in self.b4a_data.rss_sources:
                if FEEDPARSER_AVAILABLE: status_icon = '🟢'; status_text = 'Active (Tool Available)'
                else: status_icon = '⚠️'; status_text = 'Inactive (`feedparser` missing)'
                message += f'- **{rss_conf.name}**: {status_icon} {status_text} ({rss_conf.url})\n'
        else:
            message += '  *None configured or loaded.*\n'
        message += '\n'

        # Unhandled Sources
        if self.b4a_data.unhandled_sources:
            message += f'**Unhandled Sources: {len(self.b4a_data.unhandled_sources)} found**\n'
            for src in self.b4a_data.unhandled_sources:
                name = src.get('_sourcename', 'Unknown Name')
                type = src.get('type', 'Unknown Type')
                reason = src.get('_reason', 'No handler')
                message += f'- **{name}**: ⚠️ Type \'{type}\' - {reason}\n'
            message += '\n'

        # Loading Errors
        if self.b4a_data.load_errors:
            message += f'**Loading Errors: {len(self.b4a_data.load_errors)} encountered**\n'
            for path, err in self.b4a_data.load_errors[:5]: # Show first 5 errors
                message += f'- `{os.path.basename(path)}`: {err[:150]}…\n'
            if len(self.b4a_data.load_errors) > 5: message += '- … (see logs for more details)\n'
            message += '\n'

        # PGVector Status
        message += '**Chat History Storage:**\n'
        if self.pgvector_config.get('enabled', False):
            if self.pgvector_enabled: message += '- 🟢 PGVector: **Enabled and Active**\n'
            else: message += '- 🔴 PGVector: **Configured but FAILED to initialize** (Check logs)\n'
        else: message += '- ⚪ In-Memory: **Enabled** (PGVector not configured or disabled)\n'

        await send_long_message(interaction, message.strip() or 'No B4A sources found or configured.', followup=True)

    @app_commands.command(name='context_details', description='Show detailed status of B4A sources, including available tools.')
    async def context_details_slash(self, interaction: discord.Interaction):
        ''' Command to list B4A sources with detailed MCP tool lists and RSS feed names. '''
        logger.info(f'Command \'/context_details\' invoked by {interaction.user}')
        await interaction.response.defer(thinking=True, ephemeral=False)

        message_parts = ['**B4A Context Details:**\n']

        # MCP Sources
        message_parts.append(f'\n**MCP Sources (@mcp): {len(self.b4a_data.mcp_sources)} configured**')
        if not self.b4a_data.mcp_sources:
            message_parts.append('  *None configured or loaded.*')
        else:
            for mcp_conf in self.b4a_data.mcp_sources:
                name = mcp_conf.name
                status_icon = '❓'; status_text = 'Unknown'
                tools_available = False
                if name in self.mcp_connections and name in self.mcp_tools:
                    status_icon = '🟢'; status_text = f'Connected ({len(self.mcp_tools[name])} tools listed)'
                    tools_available = bool(self.mcp_tools[name])
                elif name in self.mcp_connections:
                    status_icon = '🟡'; status_text = 'Connected (Tool list issue?)'
                elif name in self._connection_tasks and not self._connection_tasks[name].done():
                    status_icon = '🟠'; status_text = 'Connecting…'
                else:
                    status_icon = '🔴'; status_text = 'Disconnected / Failed'

                message_parts.append(f'\n- **{name}**: {status_icon} {status_text} ({mcp_conf.url})')
                if tools_available:
                    message_parts.append('  *Available Tools:*')
                    for tool in self.mcp_tools[name]:
                        desc = (tool.description or 'No description').split('\n')[0] # First line of desc
                        desc_short = (desc[:70] + '...') if len(desc) > 70 else desc
                        message_parts.append(f'    - `{tool.name}`: {desc_short}')
                elif status_icon == '🟢' and not tools_available:
                     message_parts.append('  *No tools listed by this source.*')

        # RSS Sources
        message_parts.append(f'\n\n**RSS Sources (@rss): {len(self.b4a_data.rss_sources)} configured**')
        if not self.b4a_data.rss_sources:
            message_parts.append('  *None configured or loaded.*')
        else:
            rss_tool_status = 'Active (`query_rss_feed` tool available)' if FEEDPARSER_AVAILABLE else 'Inactive (`feedparser` missing)'
            message_parts.append(f'  *Status:* {rss_tool_status}')
            feed_names = [conf.name for conf in self.b4a_data.rss_sources]
            message_parts.append(f'  *Configured Feeds:* {', '.join(f'`{name}`' for name in feed_names)}')

        # Chat history storage (same logic from /context_info)
        message_parts.append('\n\n**Chat History Storage:**')
        if self.pgvector_config.get('enabled', False):
            if self.pgvector_enabled: message_parts.append('- 🟢 PGVector: **Enabled and Active**')
            else: message_parts.append('- 🔴 PGVector: **Configured but FAILED to initialize** (Check logs)')
        else: message_parts.append('- ⚪ In-Memory: **Enabled** (PGVector not configured or disabled)')

        # Unhandled sources & loading errors display can be copied from /context_info

        final_message = '\n'.join(message_parts)
        await send_long_message(interaction, final_message.strip() or 'No B4A sources found or configured.', followup=True)

    # Chat Commands (Prefix and Slash)

    # Prefix form
    @commands.command(name='chat', help='Chat with the AI assistant (Prefix Command)')
    @commands.cooldown(1, 10, commands.BucketType.user)
    async def chat_command(self, ctx: commands.Context, *, message: str):
        ''' Prefix command to chat with the LLM, using MCP tools. '''
        # Prevent prefix command in DMs, as on_message handles DMs now
        if ctx.guild is None:
            await ctx.send('You don\'t need a prefix in DMs! Just send your message directly.')
            return

        logger.info(f'Prefix command `!chat` invoked by {ctx.author} ({ctx.author.id}) in channel {ctx.channel.id}')
        channel_id = ctx.channel.id
        user_id = ctx.author.id

        # Check if message is empty OR if there are only attachments and no message text
        # Not allowing attachment-only messages in this case, but let's check that
        if not message.strip() and not ctx.message.attachments:
            await ctx.send('⚠️ Please provide a message or attach a file to chat about!')
            return

        # Get attachments
        attachments = ctx.message.attachments
        if attachments:
            logger.info(f'Prefix command `!chat` received {len(attachments)} attachments.')

        async with ctx.typing():
            await self._handle_chat_logic(
                sendable=ctx,
                message=message,
                channel_id=channel_id,
                user_id=user_id,
                stream=False, # Keep Prefix non-streaming for simplicity
                attachments=attachments
            )

    @chat_command.error
    async def chat_command_error(self, ctx: commands.Context, error: commands.CommandError):
        ''' Error handler for the prefix chat command. '''
        if isinstance(error, commands.CommandOnCooldown):
            await ctx.send(f'⏳ Woah there! This command is on cooldown. Try again in {error.retry_after:.1f} seconds.', delete_after=10)
        elif isinstance(error, commands.MissingRequiredArgument):
            await ctx.send(f'⚠️ You need to provide a message! Usage: `{ctx.prefix}chat <your message>`')
        elif isinstance(error, commands.NoPrivateMessage): # Should be caught by the check in the command now
            await ctx.send('This prefix command cannot be used in DMs. Just send your message directly!')
        else:
            logger.error(f'Error in prefix command chat_command dispatch: {error}', exc_info=error)
            # Avoid sending generic error here if _handle_chat_logic sends its own

    # Slash form
    @app_commands.command(name='chat', description='Chat with the AI assistant')
    @app_commands.describe(message='Your message to the AI', attachment='Optional file to discuss')
    @app_commands.checks.cooldown(1, 10, key=lambda i: i.user.id) # Cooldown per user
    async def chat_slash(self, interaction: discord.Interaction,
        message: str, attachment: discord.Attachment | None = None):
        ''' Slash command version of the chat command. '''
        # Slash commands can technically be invoked in DMs if enabled globally or for DMs.
        # Let's keep it consistent and allow it, as _handle_chat_logic works fine.
        # if interaction.guild is None:
        #     await interaction.response.send_message('You can just send messages directly in DMs!', ephemeral=True)
        #     return

        logger.info(f'Slash command `/chat` invoked by {interaction.user} ({interaction.user.id}) in channel {interaction.channel_id}')
        channel_id = interaction.channel_id
        user_id = interaction.user.id

        # Check for empty message/attachment
        if not message.strip() and not attachment:
            await interaction.response.send_message('⚠️ Please provide a message or attach a file to chat about!', ephemeral=True)
            return

        # Prepare attachments list (even if only one)
        attachments_list = [attachment] if attachment else []
        if attachments_list:
            logger.info(f'Slash command /chat received an attachment: {attachment.filename}')

        # Defer response
        await interaction.response.defer(thinking=True, ephemeral=False) # Non-ephemeral for public results

        await self._handle_chat_logic(
            sendable=interaction,
            message=message,
            channel_id=channel_id,
            user_id=user_id,
            stream=False, # Keep Slash non-streaming for now
            attachments=attachments_list
        )

    @chat_slash.error
    async def on_chat_slash_error(self, interaction: discord.Interaction, error: app_commands.AppCommandError):
        ''' Error Handler for Slash Command Cooldowns/Checks. '''
        if interaction.is_expired():
             logger.warning(f'Interaction {interaction.id} expired before error handler could process: {error}')
             return

        error_message = 'An unexpected error occurred with this command.'
        ephemeral = True # Most check errors should be ephemeral

        if isinstance(error, app_commands.CommandOnCooldown):
            error_message = f'⏳ Woah there! This command is on cooldown. Try again in {error.retry_after:.1f} seconds.'
        elif isinstance(error, app_commands.CheckFailure):
            error_message = 'You don\'t have the necessary permissions or conditions met to use this command.'
            logger.warning(f'CheckFailure for /chat by {interaction.user}: {error}')
        # elif isinstance(error, app_commands.NoPrivateMessage): # If we added a guild_only check
        #      error_message = 'This command cannot be used in Direct Messages.'
        else:
            # Log other errors (before our main logic runs)
            logger.error(f'Unhandled error in slash command chat_slash check/dispatch: {error}', interaction_id=interaction.id, exc_info=error)
            # Avoid sending generic user message if _handle_chat_logic sends its own

        # Try to send the error message using followup if deferred, or initial response otherwise
        try:
            if interaction.response.is_done():
                await interaction.followup.send(error_message, ephemeral=ephemeral)
            else:
                # This case implies defer() failed or the error occurred before defer()
                await interaction.response.send_message(error_message, ephemeral=ephemeral)
        except discord.HTTPException as e:
            logger.error(f'Failed to send slash command error message for interaction {interaction.id}: {e}')
        except Exception as e:
            logger.error(f'Generic exception sending slash command error for interaction {interaction.id}: {e}')
