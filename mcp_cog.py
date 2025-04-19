# mcp_cog.py
'''
Discord cog for MCP integration. Connects to MCP servers and provides
commands to interact with MCP tools.
'''
import os
import json
import asyncio
from typing import Dict, Any, List, Union, Optional, Tuple

# import httpx # Not used directly here
# import anyio # Used internally by MCP/SSE client
import discord
from discord.ext import commands
from discord import app_commands
import structlog
from openai import AsyncOpenAI
# Assuming openai types might be useful for clarity if not already imported
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall, ChoiceDelta

from mcp.client.session import ClientSession
# from mcp.client.sse import sse_client # Assuming sse_client handles anyio internally
from sse_client_vendored import sse_client # Vendored version includes the endpoint_url in return
from mcp.types import CallToolResult, TextContent

# Needed for ClosedResourceError/BrokenResourceError handling if not imported via sse/mcp
try:
    from anyio import ClosedResourceError, BrokenResourceError
except ImportError:
    # Define dummy exceptions if anyio isn't a direct dependency or accessible
    class ClosedResourceError(Exception): pass
    class BrokenResourceError(Exception): pass


logger = structlog.get_logger(__name__)

ALLOWED_FOR_LLM_INIT = ['base_url']
ALLOWED_FOR_LLM_CHAT = ['model', 'temperature']

if 'OPENAI_API_KEY' not in os.environ:
    logger.warning('OPENAI_API_KEY not set in environment variables. Using default value.')
    # Set a default placeholder if not present, required by openai lib >= 1.0
    OPENAI_API_KEY = os.environ['OPENAI_API_KEY'] = 'lm-studio'


# Helper Function for Message Splitting (Unchanged)
async def send_long_message(sendable: Union[commands.Context, discord.Interaction], content: str, followup: bool = False, ephemeral: bool = False):
    '''
    Sends a message, splitting it into multiple parts if it exceeds Discord's limit.
    Handles both Context and Interaction objects.
    '''
    max_length = 2000
    start = 0
    content_str = str(content) # Be paranoid?
    if not content_str.strip(): # Don't try to send empty messages
        logger.debug('Attempted to send empty message.')
        return

    first_chunk = True
    while start < len(content_str):
        end = start + max_length
        chunk = content_str[start:end]
        try:
            if isinstance(sendable, commands.Context):
                # Context doesn't support followup or ephemeral directly in send
                await sendable.send(chunk)
            elif isinstance(sendable, discord.Interaction):
                # Determine if we should use followup or the initial response method
                use_followup = followup or start > 0 or sendable.response.is_done()
                # Determine if ephemerality is requested and possible
                # Ephemeral usually only works on the *first* message sent via interaction response/followup
                # Making subsequent chunks ephemeral is generally not possible/reliable.
                # We only make it ephemeral if requested AND it's the first chunk AND we're using followup (or initial response if short)
                can_be_ephemeral = (use_followup and first_chunk) or (not use_followup and len(content_str) <= max_length)
                send_ephemeral = ephemeral and can_be_ephemeral

                if use_followup:
                     # Ensure followup is used if interaction is already responded to (e.g., deferred)
                     if not sendable.response.is_done():
                         # This case should ideally not happen if defer() was used and followup=True passed,
                         # but handle defensively. If not deferred, initial response needed.
                         await sendable.response.send_message(chunk, ephemeral=send_ephemeral)
                     else:
                         await sendable.followup.send(chunk, ephemeral=send_ephemeral)
                else:
                     # Initial response, only possible if not deferred and message fits
                     await sendable.response.send_message(chunk, ephemeral=send_ephemeral)

        except discord.HTTPException as e:
             logger.error(f'Failed to send message chunk: {e}', chunk_start=start, sendable_type=type(sendable), ephemeral=ephemeral)
             # Optionally break or try to send an error message
             try:
                  error_msg = f'Error sending part of the message: {e}'
                  if isinstance(sendable, commands.Context): await sendable.send(error_msg)
                  # Send error ephemerally if possible for interactions
                  elif isinstance(sendable, discord.Interaction):
                       if sendable.response.is_done(): await sendable.followup.send(error_msg, ephemeral=True)
                       else: await sendable.response.send_message(error_msg, ephemeral=True) # Try initial response for error
             except Exception: pass # Avoid error loops
             break # Stop sending further chunks on error

        start = end
        first_chunk = False
        # Avoid potential rate limits if sending many chunks quickly
        if len(content_str) > max_length:
             await asyncio.sleep(0.2)


class MCPCog(commands.Cog):
    def __init__(self, bot: commands.Bot, config: Dict[str, Any]):
        self.bot = bot
        self.config = config
        # Store live ClientSession objects directly, keyed by server name
        self.mcp_connections: Dict[str, ClientSession] = {}
        self.mcp_tools: Dict[str, List[Dict[str, Any]]] = {}
        self.message_history: Dict[int, List[Dict[str, Any]]] = {}

        self._connection_tasks: Dict[str, asyncio.Task] = {}
        self._shutdown_event = asyncio.Event()

        llm_init_params = {k: v for k, v in config.get('llm_endpoint', {}).items() if k in ALLOWED_FOR_LLM_INIT}

        if 'llm_endpoint' not in config:
            logger.error('LLM endpoint configuration ("llm_endpoint") missing!')
            raise ValueError('LLM endpoint configuration missing.')
        try:
            # Ensure API key is passed if required by the library version >= 1.0
            if 'api_key' not in llm_init_params:
                llm_init_params['api_key'] = os.environ.get('OPENAI_API_KEY', 'lm-studio') # Use env var or default
            self.llm_client = AsyncOpenAI(**llm_init_params)
        except Exception as e:
            logger.exception('Failed to initialize OpenAI client', endpoint=config.get('llm_endpoint'))
            raise e

        self.llm_chat_params = {k: v for k, v in config.get('llm_endpoint', {}).items() if k in ALLOWED_FOR_LLM_CHAT}
        # Ensure model is included if present in config, otherwise provide default
        if 'model' not in self.llm_chat_params and 'model' in config.get('llm_endpoint', {}):
             self.llm_chat_params['model'] = config['llm_endpoint']['model']
        elif 'model' not in self.llm_chat_params:
            logger.warning('Model not specified in LLM config llm_endpoint. Using default "local-model".')
            self.llm_chat_params['model'] = 'local-model'

        # LLM Settings (potentially overrides from llm_endpoint if keys overlap, like 'model')
        self.llm_settings = config.get('llm_settings', {})
        self.llm_settings.setdefault('model', self.llm_chat_params['model']) # Default to endpoint model if not specified here
        self.llm_settings.setdefault('temperature', 0.3)

        self.system_message = config.get('system_message', 'You are a helpful AI assistant. You can access tools using MCP servers.')
        logger.info('MCPCog initialized.')


    async def _manage_mcp_connection(self, url: str, name: str):
        '''Persistent task to manage a single MCP connection.'''
        reconnect_delay = 15 # Initial delay in seconds
        max_reconnect_delay = 300 # Max delay

        while not self._shutdown_event.is_set():
            logger.info(f'Attempting to connect to MCP server {name} at {url}')
            session: Optional[ClientSession] = None
            try:
                # Keep the SSE client context active for the duration of the connection
                async with sse_client(url) as (read_stream, write_stream, endpoint_url):
                    logger.info(f'SSE connection established for {name}', endpoint=endpoint_url)
                    session = ClientSession(read_stream=read_stream, write_stream=write_stream)

                    # Initialize MCP Protocol Session
                    try:
                        await asyncio.wait_for(session.initialize(), timeout=30.0)
                        logger.info(f'MCP Session initialized for {name}')
                    except asyncio.TimeoutError:
                        logger.error(f'Session initialization timeout for {name}.')
                        raise ConnectionAbortedError('Initialization timeout') # Trigger reconnect
                    except Exception as init_exc:
                        logger.exception(f'Session initialization failed for {name}', exc_info=init_exc)
                        raise # Trigger reconnect

                    # List Tools
                    try:
                        result = await asyncio.wait_for(session.list_tools(), timeout=15.0)
                        tools = [
                            {
                                'name': t.name,
                                'description': t.description,
                                'input_schema': t.inputSchema,
                            } for t in result.tools
                        ]
                        # Store live session and tools
                        self.mcp_connections[name] = session
                        self.mcp_tools[name] = tools
                        logger.info(f'Successfully listed {len(tools)} tools for {name}. Connection active.')
                        reconnect_delay = 15 # Reset delay on successful connection

                    except asyncio.TimeoutError:
                         logger.error(f'Tool listing timed out for {name}.')
                         raise ConnectionAbortedError('Tool listing timeout') # Trigger reconnect
                    except Exception as list_exc:
                        logger.exception(f'Tool listing failed for {name}', exc_info=list_exc)
                        # Clean up potentially inconsistent state if list fails after init
                        if name in self.mcp_connections: del self.mcp_connections[name]
                        if name in self.mcp_tools: del self.mcp_tools[name]
                        raise # Trigger reconnect

                    # Connection is live - Keep context open
                    # Wait indefinitely until shutdown is signaled or connection breaks
                    await self._shutdown_event.wait()

                # If we exit the 'async with sse_client' normally (e.g., via shutdown)
                logger.info(f'SSE context for {name} exited cleanly due to shutdown signal.')
                # Loop will terminate because _shutdown_event is set

            except (BrokenResourceError, ClosedResourceError, ConnectionError, ConnectionAbortedError, EOFError) as conn_err:
                # Handle expected connection drops/errors
                logger.warning(f'Connection to MCP server {name} lost or failed: {type(conn_err).__name__}. Attempting reconnect...', server_name=name)
            except asyncio.CancelledError:
                logger.info(f'Connection task for {name} cancelled.')
                break # Exit loop immediately on cancellation
            except Exception as e:
                # Catch unexpected errors during connection or the wait loop
                logger.exception(f'Unexpected error in connection task for {name}. Will attempt reconnect.', server_name=name, url=url)

            # Cleanup before potential reconnect
            if name in self.mcp_connections: del self.mcp_connections[name]
            if name in self.mcp_tools: del self.mcp_tools[name]
            session = None # Clear session object

            # Wait before retrying, unless shutting down
            if not self._shutdown_event.is_set():
                logger.info(f'Waiting {reconnect_delay}s before reconnecting to {name}.')
                try:
                    # Wait for the delay OR the shutdown signal
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=reconnect_delay)
                    # If wait_for doesn't timeout, shutdown was signaled
                    logger.info(f'Shutdown signaled during reconnect delay for {name}.')
                    break # Exit loop
                except asyncio.TimeoutError:
                    # Delay finished, continue to next loop iteration to reconnect
                    reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay) # Exponential backoff
                except asyncio.CancelledError:
                     logger.info(f'Reconnect wait for {name} cancelled.')
                     break # Exit loop

        # Final Task Cleanup
        logger.info(f'Connection management task for {name} finished.')
        if name in self.mcp_connections: del self.mcp_connections[name]
        if name in self.mcp_tools: del self.mcp_tools[name]


    async def cog_load(self):
        '''
        Set up MCP connections when the cog is loaded by starting persistent tasks.
        '''
        logger.info('Loading MCPCog and starting connection managers...')
        self._shutdown_event.clear() # Ensure shutdown is not set initially
        mcp_servers = self.config.get('mcp', {}).get('server', [])
        if not mcp_servers:
            logger.warning('No MCP servers defined in configuration.')
            return

        active_task_count = 0
        for server in mcp_servers:
            if 'url' in server and 'name' in server:
                name = server['name']
                url = server['url']
                # Start task only if it doesn't exist or has finished (e.g., after previous error/unload)
                if name not in self._connection_tasks or self._connection_tasks[name].done():
                    logger.info(f'Creating connection task for MCP server: {name}')
                    # Create and store the task
                    task = asyncio.create_task(self._manage_mcp_connection(url, name), name=f'mcp_conn_{name}')
                    self._connection_tasks[name] = task
                    active_task_count += 1
                else:
                    # Task already running, presumably from a previous load or it never stopped
                    logger.info(f'Connection task for {name} already running.')
                    active_task_count += 1
            else:
                 logger.warning('Invalid MCP server config entry, missing "url" or "name"', entry=server)

        logger.info(f'MCPCog load processed. {active_task_count} connection tasks are active or starting.')
        # Connections establish asynchronously in the background.


    async def cog_unload(self):
        '''
        Clean up MCP connections when the cog is unloaded by stopping tasks.
        '''
        logger.info('Unloading MCPCog and stopping connection tasks...')
        self._shutdown_event.set() # Signal all tasks to stop their loops

        tasks_to_wait_for = list(self._connection_tasks.values())

        if tasks_to_wait_for:
            logger.info(f'Waiting for {len(tasks_to_wait_for)} MCP connection tasks to complete shutdown...')
            # Wait for tasks to finish processing the shutdown signal
            done, pending = await asyncio.wait(tasks_to_wait_for, timeout=10.0) # Adjust timeout as needed

            if pending:
                logger.warning(f'{len(pending)} connection tasks did not shut down cleanly within timeout. Attempting cancellation.')
                for task in pending:
                    # Check if task exists before cancelling (paranoid check)
                    task_name = task.get_name() if hasattr(task, 'get_name') else 'unknown task'
                    logger.debug(f'Cancelling task: {task_name}')
                    task.cancel()
                # Optionally wait a moment for cancellations to be processed
                await asyncio.sleep(1)
            else:
                 logger.info('All connection tasks shut down gracefully.')
        else:
             logger.info('No active connection tasks to stop.')

        # Clear stored data regardless of task shutdown status
        self.mcp_connections.clear()
        self.mcp_tools.clear()
        self._connection_tasks.clear() # Clear the task references
        logger.info('MCPCog unloaded and connection resources cleared.')


    # Removed the old connect_mcp function


    async def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Union[CallToolResult, Dict[str, str]]:
        '''
        Execute an MCP tool using the live session managed by background tasks.
        '''
        logger.debug('Executing tool', tool_name=tool_name, tool_input=tool_input)

        mcp_session: Optional[ClientSession] = None
        server_name_found: Optional[str] = None

        # Check active connections (live sessions stored directly)
        for server_name, session in self.mcp_connections.items():
            # Verify this connected server is known to have the tool
            if server_name in self.mcp_tools and any(tool.get('name') == tool_name for tool in self.mcp_tools[server_name]):
                # Found an active session for the server hosting the tool
                mcp_session = session
                server_name_found = server_name
                break # Stop searching once found


        if not mcp_session or not server_name_found:
            logger.warning(f'Tool "{tool_name}" not found on any *actively connected* MCP server.')
            # Provide a more specific error message
            if not self.mcp_connections:
                return {'error': f'Tool "{tool_name}" cannot be executed because no MCP servers are currently connected.'}
            else:
                 # Check if the tool *exists* in the config, even if disconnected
                 tool_server_config = next((s['name'] for s in self.config.get('mcp', {}).get('server', []) if s.get('name') and name in self.mcp_tools and any(t.get('name') == tool_name for t in self.mcp_tools[s.get('name')])), None)

                 if tool_server_config:
                     return {'error': f'Tool "{tool_name}" exists (on server "{tool_server_config}"), but the server is currently disconnected or unavailable.'}
                 else:
                     # Tool wasn't found even in the configuration/initial tool lists
                     return {'error': f'Tool "{tool_name}" not found on any configured MCP server.'}


        logger.info(f'Calling tool "{tool_name}" on MCP server "{server_name_found}"')
        try:
            # The session retrieved from self.mcp_connections should be live.
            # Add a timeout for the call itself.
            result = await asyncio.wait_for(mcp_session.call_tool(tool_name, tool_input), timeout=60.0)
            logger.debug(f'Tool "{tool_name}" executed successfully.', result_type=type(result))
            return result

        except asyncio.TimeoutError:
             logger.error(f'Timeout calling tool "{tool_name}" on server "{server_name_found}"', tool_input=tool_input)
             return {'error': f'Tool call "{tool_name}" timed out.'}
        except (BrokenResourceError, ClosedResourceError, ConnectionError) as conn_err:
            # Catch connection errors that might happen if the connection drops *exactly* during the call
            logger.error(f'Connection error during tool call "{tool_name}" on server "{server_name_found}": {type(conn_err).__name__}', tool_input=tool_input)
            # Remove the dead session reference; the management task will handle reconnection.
            if server_name_found in self.mcp_connections:
                 del self.mcp_connections[server_name_found]
                 logger.warning(f'Removed dead connection reference for "{server_name_found}" due to error during tool call.')
            return {'error': f'Connection error executing tool "{tool_name}". The server may be temporarily unavailable. Please try again.'}
        except Exception as e:
            # Catch other unexpected errors from call_tool or result processing
            logger.exception(f'Unexpected error calling tool "{tool_name}" on server "{server_name_found}"', tool_input=tool_input)
            return {'error': f'Error calling tool "{tool_name}": {str(e)}'}


    # format_calltoolresult_content (Unchanged)
    def format_calltoolresult_content(self, result: Union[CallToolResult, Dict[str, str]]) -> str:
        '''
        Extract text content from a CallToolResult object or format error dict.
        '''
        if isinstance(result, CallToolResult):
            text_contents = []
            # Check structure defensively
            content_list = getattr(result, 'content', None)
            if isinstance(content_list, list):
                for content_item in content_list:
                    # Check for TextContent type more robustly
                    if isinstance(content_item, TextContent) and hasattr(content_item, 'text'):
                        # Ensure text is not None before converting to string
                        if content_item.text is not None:
                            text_contents.append(str(content_item.text))
                        else:
                            logger.warning('TextContent item had None text', item=content_item)
                    # Optionally handle other content types here if needed in the future
                    # elif isinstance(content_item, OtherContentType): ...
                if text_contents:
                    return '\n'.join(text_contents)
                else:
                     logger.warning('CallToolResult received but contained no parsable TextContent text', result_details=str(result)[:200]) # Limit log length
                     # Try to return *something* useful if possible, fallback to generic message
                     try:
                         return f'Tool executed. Result: {json.dumps(result.model_dump())}' # Use model_dump if pydantic
                     except Exception:
                         return f'Tool executed, but returned no displayable text. (Result: {str(result)[:100]})'
            else:
                 logger.warning('CallToolResult received with no "content" list or invalid format', result_details=str(result)[:200])
                 return f'Tool executed, but returned no parseable content. (Result: {str(result)[:100]})'
        elif isinstance(result, dict) and 'error' in result:
            return f'Tool Error: {result["error"]}'
        else:
            # Gracefully handle unexpected types
            logger.warning('Unexpected format received in format_calltoolresult_content', received_result=result)
            return f'Unexpected tool result format: {str(result)[:100]}'


    # get_channel_history (Unchanged)
    def get_channel_history(self, channel_id: int) -> List[Dict[str, Any]]:
        '''
        Get message history for a channel, or initialize if not exists.
        Includes basic history length management.
        '''
        max_history_len = self.config.get('max_history_length', 20)
        if channel_id not in self.message_history:
            self.message_history[channel_id] = [
                {'role': 'system', 'content': self.system_message}
            ]
            logger.info(f'Initialized message history for channel {channel_id}')
        else:
             # Ensure history trimming logic handles the system message correctly
             current_len = len(self.message_history[channel_id])
             # We want to keep max_history_len messages + 1 system message
             target_len = max_history_len + 1
             if current_len > target_len:
                 # Calculate how many non-system messages to remove from the start
                 amount_to_remove = current_len - target_len
                 # Keep the system message ([0]) and the last `max_history_len` messages
                 self.message_history[channel_id] = [self.message_history[channel_id][0]] + self.message_history[channel_id][amount_to_remove + 1:]
                 logger.debug(f'Trimmed message history for channel {channel_id}, removed {amount_to_remove} messages. New length: {len(self.message_history[channel_id])}.')

        return self.message_history[channel_id]


    # format_tools_for_openai (Minor adjustment to use active connections)
    async def format_tools_for_openai(self) -> List[Dict[str, Any]]:
        '''
        Format active MCP tools (from connected servers) for OpenAI API.
        '''
        openai_tools = []
        # Iterate through servers confirmed to be actively connected
        active_server_names = list(self.mcp_connections.keys())

        for server_name in active_server_names:
            if server_name not in self.mcp_tools:
                logger.warning(f'Server "{server_name}" is connected but has no tools listed. Skipping for OpenAI format.', server_name=server_name)
                continue # Should ideally not happen if connection logic is sound

            tools = self.mcp_tools[server_name]
            logger.debug(f'Formatting {len(tools)} tools from active server "{server_name}" for OpenAI.')

            for tool in tools:
                # Add more robust checking for tool structure
                if not isinstance(tool, dict) or not all(k in tool for k in ['name', 'description', 'input_schema']):
                    logger.warning(f'Skipping malformed tool definition from server "{server_name}"', tool_data=tool)
                    continue
                if not isinstance(tool['name'], str) or not tool['name']:
                    logger.warning(f'Skipping tool with invalid name from server "{server_name}"', tool_data=tool)
                    continue

                parameters = tool['input_schema']
                # Ensure parameters is a valid JSON schema object
                if not isinstance(parameters, dict):
                    logger.warning(f'Tool "{tool["name"]}" from server "{server_name}" has non-dict input_schema. Using empty schema.', schema=parameters)
                    # Provide a minimally valid JSON schema
                    parameters = {'type': 'object', 'properties': {}}

                # Basic validation: ensure type is object (common requirement)
                # OpenAI often requires 'object' type for parameters root.
                if 'type' not in parameters or parameters.get('type') != 'object':
                     logger.warning(f'Tool "{tool["name"]}" input_schema type is not "object". Formatting as-is, but may cause issues with LLM.', schema=parameters)
                     # Allow it but log warning. If LLM fails, this could be why.

                openai_tool = {
                    'type': 'function',
                    'function': {
                        'name': tool['name'],
                        'description': tool.get('description', ''), # Use empty string if missing
                        'parameters': parameters,
                    },
                }
                openai_tools.append(openai_tool)
        return openai_tools


    # _handle_chat_logic (Unchanged)
    async def _handle_chat_logic(
        self,
        sendable: Union[commands.Context, discord.Interaction],
        message: str,
        channel_id: int,
        user_id: int,
        stream: bool
    ):
        '''
        Core logic for handling chat interactions, LLM calls, and tool execution.
        '''
        is_interaction = isinstance(sendable, discord.Interaction)
        # For interactions, followup=True must be used after the initial deferral.
        # For context, followup=False. send_long_message handles the details.
        # If interaction isn't deferred yet, the first send won't use followup.
        send_followup = is_interaction and sendable.response.is_done()

        try:
            # Get and update history
            channel_history = self.get_channel_history(channel_id)
            channel_history.append({'role': 'user', 'content': message})
            logger.debug(f'User message added to history for channel {channel_id}', user_id=user_id, stream=stream)

            # Prepare chat parameters
            chat_params = {**self.llm_chat_params, **self.llm_settings, 'stream': stream} # Combine endpoint & settings params
            openai_tools = await self.format_tools_for_openai()
            if openai_tools:
                chat_params['tools'] = openai_tools
                chat_params['tool_choice'] = 'auto' # Let model decide or use tools
                logger.debug(f'Including {len(openai_tools)} tools in LLM call for channel {channel_id}')
            else:
                 logger.debug(f'No active tools available for LLM call for channel {channel_id}')


            # Storing results
            initial_response_content = ''
            tool_calls_aggregated: List[Dict[str, Any]] = [] # Store unified tool call format
            assistant_message_dict: Dict[str, Any] = {'role': 'assistant', 'content': None} # Start with None content

            # Initial LLM Call (Streaming or Non-Streaming)
            logger.debug(f'Initiating LLM call for channel {channel_id}', user_id=user_id, stream=stream, model=chat_params.get('model'))
            if stream:
                # Streaming Logic
                try:
                    llm_stream = await self.llm_client.chat.completions.create(messages=channel_history, user=str(user_id), **chat_params)
                    current_tool_calls: List[Dict[str, Any]] = [] # Temp list for streamed tool calls
                    stream_error = False # Flag for stream errors
                    async for chunk in llm_stream:
                        delta: Optional[ChoiceDelta] = chunk.choices[0].delta if chunk.choices else None
                        if not delta: continue

                        # Aggregate content
                        if token := delta.content or '': initial_response_content += token

                        # Aggregate tool calls (handle potential partials)
                        if delta.tool_calls:
                            for tool_call_chunk in delta.tool_calls:
                                idx = tool_call_chunk.index
                                # Ensure list is long enough
                                while len(current_tool_calls) <= idx:
                                    current_tool_calls.append({'id': None, 'type': 'function', 'function': {'name': '', 'arguments': ''}})

                                tc_ref = current_tool_calls[idx] # Reference for modification
                                chunk_func: Optional[ChoiceDeltaToolCall.Function] = tool_call_chunk.function

                                # Populate parts as they arrive
                                if tool_call_chunk.id: tc_ref['id'] = tool_call_chunk.id
                                if chunk_func:
                                    if chunk_func.name: tc_ref['function']['name'] += chunk_func.name
                                    if chunk_func.arguments: tc_ref['function']['arguments'] += chunk_func.arguments

                    # Finalize tool calls from stream
                    tool_calls_aggregated = [tc for tc in current_tool_calls if tc.get('id') and tc['function'].get('name')]

                except Exception as stream_exc:
                     logger.exception('Error during LLM stream processing', channel_id=channel_id, user_id=user_id)
                     await send_long_message(sendable, '⚠️ Error receiving response from AI.', followup=send_followup, ephemeral=is_interaction)
                     return # Abort processing on stream error


            else: # Non-Streaming Logic
                 try:
                    response = await self.llm_client.chat.completions.create(messages=channel_history, user=str(user_id), **chat_params)

                    response_message: Optional[ChatCompletionMessage] = response.choices[0].message if response.choices else None
                    if not response_message:
                         logger.error('LLM response missing message object', response_data=response.model_dump_json(indent=2))
                         await send_long_message(sendable, '⚠️ Received an empty response from AI.', followup=send_followup, ephemeral=is_interaction)
                         return

                    initial_response_content = response_message.content or ''
                    raw_tool_calls: Optional[List[ChatCompletionMessageToolCall]] = response_message.tool_calls

                    if raw_tool_calls:
                         # Convert to the same dict format used by streaming for consistency
                         tool_calls_aggregated = [
                             {
                                 'id': tc.id,
                                 'type': tc.type, # Should be 'function'
                                 'function': {
                                     'name': tc.function.name,
                                     'arguments': tc.function.arguments
                                 }
                             } for tc in raw_tool_calls if tc.type == 'function' and tc.function and tc.id and tc.function.name # Basic validation
                         ]
                         logger.debug(f'Non-streaming response included {len(tool_calls_aggregated)} raw tool calls.', channel_id=channel_id)


                 except Exception as api_exc:
                     logger.exception('Error during non-streaming LLM API call', channel_id=channel_id, user_id=user_id)
                     await send_long_message(sendable, f'⚠️ Error communicating with AI: {str(api_exc)}', followup=send_followup, ephemeral=is_interaction)
                     return # Abort processing on API error

            # Send Initial Response & Update History (if content or tools exist)
            sent_initial_message = False
            if initial_response_content.strip():
                logger.debug(f'Sending initial LLM response to channel {channel_id}', length=len(initial_response_content), stream=stream)
                await send_long_message(sendable, initial_response_content, followup=send_followup)
                assistant_message_dict['content'] = initial_response_content # Update history content
                sent_initial_message = True
                send_followup = True # Any subsequent messages MUST be followups

            if tool_calls_aggregated:
                 assistant_message_dict['tool_calls'] = tool_calls_aggregated
                 logger.info(f'LLM requested {len(tool_calls_aggregated)} tool call(s) for channel {channel_id}', tool_names=[tc['function']['name'] for tc in tool_calls_aggregated], stream=stream)

            # Add assistant message to history *if* it had content OR requested tools
            if assistant_message_dict.get('content') or assistant_message_dict.get('tool_calls'):
                channel_history.append(assistant_message_dict)
            elif not sent_initial_message:
                # Handle case where LLM returns nothing (no text, no tools) - after checks above, this should be rare
                logger.info(f'LLM call finished for channel {channel_id} with no text/tools.', stream=stream)
                await send_long_message(sendable, 'I received your message but didn\'t have anything specific to add or do.', followup=send_followup, ephemeral=is_interaction)
                return # End processing here if there's nothing to do

            # Process Tool Calls (if any)
            if tool_calls_aggregated:
                tool_results_for_history = []
                any_tool_executed = False # Track if we actually ran any tool

                for tool_call in tool_calls_aggregated:
                    # Defensive checks for tool call structure
                    if not isinstance(tool_call, dict) or 'function' not in tool_call or 'id' not in tool_call:
                        logger.error('Malformed tool call structure received from LLM', tool_call_data=tool_call)
                        continue
                    if not isinstance(tool_call['function'], dict) or 'name' not in tool_call['function'] or 'arguments' not in tool_call['function']:
                         logger.error('Malformed tool call function structure received from LLM', tool_call_data=tool_call)
                         continue

                    tool_name = tool_call['function']['name']
                    tool_call_id = tool_call['id']
                    arguments_str = tool_call['function']['arguments']

                    # Default error content in case of failure before formatting
                    tool_result_content_for_llm = f'Error: Tool call processing failed internally before execution for {tool_name}'
                    try:
                        tool_args = json.loads(arguments_str)
                        logger.info(f'Executing tool `{tool_name}` for channel {channel_id}', args=tool_args, tool_call_id=tool_call_id)
                        tool_result_obj = await self.execute_tool(tool_name, tool_args)
                        any_tool_executed = True # Mark that we attempted execution

                        # Format result for user AND for next LLM call
                        # User sees formatted string, LLM sees potentially structured data (though often just string content is fine)
                        tool_result_content_for_user = self.format_calltoolresult_content(tool_result_obj)

                        # For the LLM history, send back the formatted string content for now
                        # Future enhancement: could send back JSON string if tool returns complex data LLM might parse
                        tool_result_content_for_llm = tool_result_content_for_user

                        # Send result back to user (non-ephemeral)
                        # Use code block for better readability
                        await send_long_message(sendable, f'```Tool Call: {tool_name}\nResult:\n{tool_result_content_for_user}```', followup=True) # Must be followup now
                        send_followup = True # Redundant but safe

                    except json.JSONDecodeError:
                         logger.error(f'Failed to decode JSON args for tool `{tool_name}`', args_str=arguments_str, tool_call_id=tool_call_id)
                         tool_result_content_for_llm = f'Error: Invalid JSON arguments provided for tool "{tool_name}": {arguments_str}'
                         await send_long_message(sendable, f'⚠️ Error: Couldn\'t understand arguments for tool `{tool_name}`. LLM provided: `{arguments_str}`', followup=True, ephemeral=is_interaction)
                         send_followup = True
                    except Exception as exec_e:
                        # Catch errors from execute_tool or format_calltoolresult_content
                        logger.exception(f'Error executing tool `{tool_name}` or processing its result', tool_call_id=tool_call_id)
                        tool_result_content_for_llm = f'Error executing tool "{tool_name}": {str(exec_e)}'
                        await send_long_message(sendable, f'⚠️ Error running tool `{tool_name}`: {str(exec_e)}', followup=True, ephemeral=is_interaction)
                        send_followup = True

                    # Always append a result to history for the LLM's context
                    tool_results_for_history.append({
                        'role': 'tool',
                        'tool_call_id': tool_call_id,
                        'name': tool_name, # Redundant but part of OpenAI spec
                        'content': tool_result_content_for_llm # Content sent back to LLM
                    })

                # Only proceed to follow-up call if at least one tool was actually processed (even if it failed)
                if tool_results_for_history:
                    channel_history.extend(tool_results_for_history)
                    logger.debug(f'Added {len(tool_results_for_history)} tool results to history for channel {channel_id}')

                    # --- Get Follow-up Response (Streaming or Non-Streaming) ---
                    logger.debug(f'Initiating follow-up LLM call for channel {channel_id} after tools.', stream=stream)
                    # Use same stream setting as initial call for consistency? Or force non-stream? Let's keep it consistent.
                    follow_up_params = {**self.llm_chat_params, **self.llm_settings, 'stream': stream}
                    # Remove tool parameters for the follow-up call unless we expect chained tool use
                    follow_up_params.pop('tools', None)
                    follow_up_params.pop('tool_choice', None)

                    follow_up_text = ''
                    try:
                        if stream:
                            # Streaming Follow-up
                            follow_up_stream = await self.llm_client.chat.completions.create(messages=channel_history, user=str(user_id), **follow_up_params)
                            async for chunk in follow_up_stream:
                                if token := chunk.choices[0].delta.content or '': follow_up_text += token
                        else:
                            # Non-Streaming Follow-up
                            follow_up_response = await self.llm_client.chat.completions.create(messages=channel_history, user=str(user_id), **follow_up_params)
                            follow_up_message = follow_up_response.choices[0].message if follow_up_response.choices else None
                            if follow_up_message:
                                follow_up_text = follow_up_message.content or ''

                    except Exception as followup_exc:
                         logger.exception('Error during follow-up LLM call', channel_id=channel_id, user_id=user_id)
                         await send_long_message(sendable, '⚠️ Error getting follow-up response from AI after tool use.', followup=True, ephemeral=is_interaction)
                         # Don't return here, history might still be useful? Or return? Let's return to be safe.
                         return

                    # Send Follow-up & Update History
                    if follow_up_text.strip():
                        logger.debug(f'Sending follow-up LLM response to channel {channel_id}', length=len(follow_up_text), stream=stream)
                        await send_long_message(sendable, follow_up_text, followup=True) # Must be followup
                        channel_history.append({'role': 'assistant', 'content': follow_up_text})
                    else:
                         logger.info(f'LLM follow-up call finished for channel {channel_id} with no text content.', stream=stream)
                         # Optionally send a message like "Tool execution complete." if no text.
                         # await send_long_message(sendable, '✅ Tool execution complete.', followup=True, ephemeral=is_interaction)


        except Exception as e:
            logger.exception(f'Unhandled error during chat logic execution for channel {channel_id}', user_id=user_id, stream=stream)
            # Try to send an error message back to the user
            try:
                 error_message = f'An unexpected error occurred while processing your request: {str(e)}'
                 # Send ephemeral if possible (interaction), otherwise normal send (context)
                 await send_long_message(sendable, error_message, followup=send_followup, ephemeral=is_interaction)
            except Exception as send_err:
                 logger.error('Failed to send error message back to user after main chat logic failure', send_error=send_err)


    @commands.Cog.listener()
    async def on_ready(self):
        ''' Cog ready listener. Connection tasks are started in cog_load. '''
        logger.info(f'Cog {self.__class__.__name__} is ready.')
        # Bot ready message is usually handled in the main bot file after login.
        # print(f'{self.bot.user} is ready. MCPCog listener active.') # Optional: Keep if useful


    @app_commands.command(name='mcp_list', description='List connected MCP servers and available tools')
    async def mcp_list_slash(self, interaction: discord.Interaction):
        ''' Command to list connected MCP servers and tools. '''
        logger.info(f'Command "/mcp_list" invoked by {interaction.user} in channel {interaction.channel_id}')
        await interaction.response.defer(thinking=True, ephemeral=False) # Use non-ephemeral thinking

        if not self.mcp_connections and not self.mcp_tools:
            # Check tasks too, maybe they are trying to connect
            if not self._connection_tasks:
                 await interaction.followup.send('No MCP servers are configured or attempting to connect.')
            else:
                 await interaction.followup.send('MCP servers are configured, but none are currently connected. They might be attempting to reconnect.')
            return

        message = '**MCP Server Status & Tools:**\n\n'
        configured_servers = {s['name'] for s in self.config.get('mcp', {}).get('server', []) if 'name' in s}
        connected_servers = set(self.mcp_connections.keys())
        listed_tool_servers = set(self.mcp_tools.keys())

        all_server_names = configured_servers.union(connected_servers).union(listed_tool_servers)

        if not all_server_names:
             await interaction.followup.send('No MCP servers found in configuration or current state.')
             return

        for name in sorted(list(all_server_names)):
             status = ''
             if name in connected_servers:
                 status = '🟢 Connected'
             elif name in self._connection_tasks and not self._connection_tasks[name].done():
                 status = '🟠 Connecting/Reconnecting'
             elif name in configured_servers:
                  status = '🔴 Disconnected'
             else:
                  status = '⚪ Unknown/Stale?' # Should not happen if logic is correct

             message += f'- **{name}**: {status}\n'

             # List tools only if connected and tools are available
             if name in connected_servers and name in self.mcp_tools:
                 tools = self.mcp_tools[name]
                 if tools:
                     message += f'  **Tools ({len(tools)}):**\n'
                     tool_limit = 7 # Limit displayed tools per server for brevity
                     displayed_count = 0
                     for i, tool in enumerate(tools):
                          # Basic validation before display
                          tool_name = tool.get('name', f'Unnamed Tool {i+1}')
                          if not isinstance(tool_name, str) or not tool_name:
                               tool_name = f'Invalid Tool {i+1}'

                          if displayed_count < tool_limit:
                              desc = tool.get('description', 'No description')
                              # Ensure description is string before slicing
                              if not isinstance(desc, str): desc = 'Invalid description type'
                              message += f'    - `{tool_name}`: {desc[:100]}{"..." if len(desc)>100 else ""}\n' # Limit desc length
                              displayed_count += 1
                          elif displayed_count == tool_limit:
                              remaining = len(tools) - tool_limit
                              if remaining > 0:
                                   message += f'    - ... and {remaining} more\n'
                              break # Stop listing tools for this server
                 else:
                     message += '  *Connected, but no tools reported by the server.*\n'
             elif name in connected_servers and name not in self.mcp_tools:
                  message += '  *Connected, but failed to retrieve tool list.*\n'

             message += '\n' # Add space between servers

        await send_long_message(interaction, message.strip(), followup=True)


    # Prefix form of chat command
    @commands.command(name='chat', help='Chat with the AI assistant (Prefix Command)')
    @commands.cooldown(1, 10, commands.BucketType.user) # Cooldown: 1 use per 10s per user
    async def chat_command(self, ctx: commands.Context, *, message: str):
        '''
        Command to chat with the LLM, possibly using MCP tools. (Uses prefix)
        Set stream=True/False based on desired behavior for prefix commands.
        '''
        logger.info(f'Prefix command "!chat" invoked by {ctx.author} in channel {ctx.channel.id}')
        channel_id = ctx.channel.id
        user_id = ctx.author.id

        async with ctx.typing(): # Shows 'Bot is typing...'
            await self._handle_chat_logic(
                sendable=ctx,
                message=message,
                channel_id=channel_id,
                user_id=user_id,
                stream=False # Set Prefix command non-streaming (matches original uncommented line)
            )

    @chat_command.error
    async def chat_command_error(self, ctx: commands.Context, error: commands.CommandError):
        ''' Error handler for the prefix command. '''
        if isinstance(error, commands.CommandOnCooldown):
            await ctx.send(f'⏳ Woah there! This command is on cooldown. Try again in {error.retry_after:.1f} seconds.', delete_after=10)
        elif isinstance(error, commands.MissingRequiredArgument):
             await ctx.send(f'⚠️ You need to provide a message! Usage: `{ctx.prefix}chat <your message>`')
        else:
            # Log other command framework errors (e.g., CheckFailure)
            logger.error(f'Error in prefix command chat_command dispatch: {error}', exc_info=error)
            # Avoid sending generic error here if _handle_chat_logic sends its own for internal errors.


    # Slash form of chat command
    @app_commands.command(name='chat', description='Chat with the AI assistant')
    @app_commands.describe(message='Your message to the AI')
    @app_commands.checks.cooldown(1, 10, key=lambda i: i.user.id) # Cooldown per user
    async def chat_slash(self, interaction: discord.Interaction, message: str):
        '''
        Slash command version of the chat command.
        Set stream=True/False based on desired behavior for slash commands.
        '''
        logger.info(f'Slash command `/chat` invoked by {interaction.user} in channel {interaction.channel_id}')
        channel_id = interaction.channel_id
        user_id = interaction.user.id

        # Defer response - essential for potentially long operations
        await interaction.response.defer(thinking=True, ephemeral=False) # Non-ephemeral thinking allows public tool results

        await self._handle_chat_logic(
            sendable=interaction,
            message=message,
            channel_id=channel_id,
            user_id=user_id,
            stream=False # Set Slash command non-streaming
        )

    @chat_slash.error
    async def on_chat_slash_error(self, interaction: discord.Interaction, error: app_commands.AppCommandError):
        ''' Error Handler for Slash Command Cooldowns/Checks. '''
        # Check if interaction is still valid before responding
        if interaction.is_expired():
             logger.warning(f'Interaction {interaction.id} expired before error handler could process: {error}')
             return

        error_message = 'An unexpected error occurred with this command.'
        ephemeral = True # Most check errors should be ephemeral

        if isinstance(error, app_commands.CommandOnCooldown):
            error_message = f'⏳ Woah there! This command is on cooldown. Try again in {error.retry_after:.1f} seconds.'
        elif isinstance(error, app_commands.CheckFailure):
            # Handle other check failures (e.g., permissions) ephemerally
             error_message = 'You don\'t have the necessary permissions or conditions met to use this command.'
        else:
            # Log other errors (these might be from discord.py before our main logic runs)
            logger.error(f'Unhandled error in slash command chat_slash check/dispatch: {error}', interaction_id=interaction.id, exc_info=error)
            # Keep the generic error message for unexpected issues
            # Don't send generic error here if _handle_chat_logic handles its own errors.
            # This handler catches errors *before* or *outside* the main command logic.

        # Try to send the error message
        try:
            # Use followup if defer was successful, otherwise try initial response
            if interaction.response.is_done():
                 await interaction.followup.send(error_message, ephemeral=ephemeral)
            else:
                 # Should only happen if defer failed somehow
                 await interaction.response.send_message(error_message, ephemeral=ephemeral)
        except discord.HTTPException as e:
            logger.error(f'Failed to send slash command error message for interaction {interaction.id}: {e}')
        except Exception as e:
             logger.error(f'Generic exception sending slash command error for interaction {interaction.id}: {e}')
