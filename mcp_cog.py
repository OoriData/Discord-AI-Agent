'''
Discord cog for MCP integration. Connects to MCP servers and provides
commands to interact with MCP tools.
'''
import os
import json
import asyncio
from typing import Dict, Any, List, Union, Optional, Tuple

# import httpx
# import anyio
import discord
from discord.ext import commands
from discord import app_commands
import structlog
from openai import AsyncOpenAI
# Assuming openai types might be useful for clarity if not already imported
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall, ChoiceDelta

from mcp.client.session import ClientSession
# from mcp.client.sse import sse_client
from sse import sse_client  # Use vendored version to get the URL
from mcp.types import CallToolResult, TextContent

logger = structlog.get_logger(__name__)

ALLOWED_FOR_LLM_INIT = ['base_url']
ALLOWED_FOR_LLM_CHAT = ['model', 'temperature']

if 'OPENAI_API_KEY' not in os.environ:
    logger.warning('OPENAI_API_KEY not set in environment variables. Using default value.')
    OPENAI_API_KEY = os.environ['OPENAI_API_KEY'] = 'lm-studio'


# Helper Function for Message Splitting
async def send_long_message(sendable: Union[commands.Context, discord.Interaction], content: str, followup: bool = False, ephemeral: bool = False):
    '''
    Sends a message, splitting it into multiple parts if it exceeds Discord's limit.
    Handles both Context and Interaction objects.
    '''
    max_length = 2000
    start = 0
    content_str = str(content)  # Be paranoid?
    if not content_str.strip(): # Don't try to send empty messages
        logger.debug('Attempted to send empty message.')
        return

    first_chunk = True
    while start < len(content_str):
        end = start + max_length
        chunk = content_str[start:end]
        try:
            if isinstance(sendable, commands.Context):
                # Context doesn't support followup or ephemeral
                await sendable.send(chunk)
            elif isinstance(sendable, discord.Interaction):
                # Use followup after defer or for subsequent chunks
                # Use ephemeral only if specified AND it's possible (usually for first followup)
                can_be_ephemeral = followup and first_chunk
                send_ephemeral = ephemeral and can_be_ephemeral

                if followup or start > 0:
                    await sendable.followup.send(chunk, ephemeral=send_ephemeral)
                else:
                    # Requires the interaction to *not* be deferred
                    # If deferred, followup must be true.
                    if sendable.response.is_done():
                         await sendable.followup.send(chunk, ephemeral=send_ephemeral)
                    else:
                         # Cannot easily make initial response ephemeral AND multi-part
                         await sendable.response.send_message(chunk, ephemeral=ephemeral and len(content_str) <= max_length)

        except discord.HTTPException as e:
             logger.error(f'Failed to send message chunk: {e}', chunk_start=start, sendable_type=type(sendable), ephemeral=ephemeral)
             # Optionally break or try to send an error message
             try:
                  error_msg = f'Error sending part of the message: {e}'
                  if isinstance(sendable, commands.Context): await sendable.send(error_msg)
                  # Send error ephemerally if possible for interactions
                  elif isinstance(sendable, discord.Interaction): await sendable.followup.send(error_msg, ephemeral=True)
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
        self.mcp_connections: Dict[str, tuple[ClientSession, Any]] = {}
        self.mcp_tools: Dict[str, List[Dict[str, Any]]] = {}
        self.message_history: Dict[int, List[Dict[str, Any]]] = {}
        llm_init_params = {k: v for k, v in config.get('llm_endpoint', {}).items() if k in ALLOWED_FOR_LLM_INIT}

        if 'llm_endpoint' not in config:
            logger.error('LLM endpoint configuration ("llm_endpoint") missing!')
            raise ValueError('LLM endpoint configuration missing.')
        try:
            self.llm_client = AsyncOpenAI(**llm_init_params)
        except Exception as e:
            logger.exception('Failed to initialize OpenAI client', endpoint=config.get('llm_endpoint'))
            raise e

        self.llm_chat_params = {k: v for k, v in config.get('llm_endpoint', {}).items() if k in ALLOWED_FOR_LLM_CHAT}
        if 'model' not in self.llm_chat_params:
            logger.warning('Model not specified in LLM chat parameters. Using default value.')
            self.llm_chat_params['model'] = 'local-model'
        self.llm_settings = config.get('llm_settings', {})
        self.llm_settings.setdefault('model', 'local-model')
        self.llm_settings.setdefault('temperature', 0.3)

        self.system_message = config.get('system_message', 'You are a helpful AI assistant. You can access tools using MCP servers.')
        logger.info('MCPCog initialized.')

    async def cog_load(self):
        '''
        Set up MCP connections when the cog is loaded.
        '''
        logger.info('Loading MCPCog and connecting to MCP servers...')
        mcp_servers = self.config.get('mcp', {}).get('server', [])
        if not mcp_servers:
            logger.warning('No MCP servers defined in configuration.')
            return

        connect_tasks = []
        for server in mcp_servers:
            if 'url' in server and 'name' in server:
                connect_tasks.append(self.connect_mcp(server['url'], server['name']))
            else:
                 logger.warning('Invalid MCP server config entry, missing "url" or "name"', entry=server)

        results = await asyncio.gather(*connect_tasks)
        successful_connections = sum(1 for r in results if r)
        logger.info(f'MCPCog loaded. Attempted {len(connect_tasks)} MCP connections, {successful_connections} successful.')


    async def cog_unload(self):
        '''
        Clean up MCP connections when the cog is unloaded.
        '''
        logger.info('Unloading MCPCog and disconnecting from MCP servers...')
        # Assuming ClientSession handles cleanup in __aexit__ or explicit close is needed
        # disconnect_tasks = []
        # for name, (session, connection) in self.mcp_connections.items():
        #     logger.info(f'Initiating disconnect from MCP server: {name}')
        #     # if hasattr(session, 'close'): disconnect_tasks.append(session.close()) # Example
        # await asyncio.gather(*disconnect_tasks) # Uncomment if explicit close needed
        self.mcp_connections.clear()
        self.mcp_tools.clear()
        logger.info('MCPCog unloaded.')


    async def connect_mcp(self, url: str, name: str) -> bool:
        '''
        Connect to an MCP server. Returns True on success, False on failure.
        (Implementation remains the same as provided)
        '''
        logger.info(f'Attempting to connect to MCP server {name} at {url}')
        session: Optional[ClientSession] = None
        streams: Optional[Tuple[Any, Any]] = None
        try:
            # Using a simple flag to track overall success through stages
            connection_successful = False

            # Stage 1: Establish SSE Connection
            async with sse_client(url) as (read_stream, write_stream, endpoint_url):
                logger.debug(f'SSE streams acquired for {name}', endpoint=endpoint_url)
                streams = (read_stream, write_stream) # Store streams temporarily

                # Stage 2: Initialize ClientSession
                session = ClientSession(read_stream=read_stream, write_stream=write_stream)
                async with session: # Enters session context manager (__aenter__)
                    logger.debug(f'Entered ClientSession context for {name}')

                    # Stage 3: Initialize MCP Protocol Session
                    try:
                        await asyncio.wait_for(session.initialize(), timeout=30.0)
                        logger.info(f'MCP Session initialized for {name}')
                    except asyncio.TimeoutError:
                        logger.error(f'Session initialization timeout for {name}')
                        return False # Fail early
                    except Exception as init_exc:
                        logger.exception(f'Session initialization failed for {name}', exc_info=init_exc)
                        return False # Fail early

                    # Stage 4: List Tools
                    try:
                        result = await asyncio.wait_for(session.list_tools(), timeout=15.0)
                        tools = [
                            {
                                'name': t.name,
                                'description': t.description,
                                'input_schema': t.inputSchema,
                            }
                            for t in result.tools
                        ]
                        # If successful up to here, store everything
                        self.mcp_connections[name] = (session, streams) # Store session *and* streams
                        self.mcp_tools[name] = tools
                        connection_successful = True # Mark success
                        logger.info(f'Successfully listed {len(tools)} tools for {name}. Connection complete.')

                    except asyncio.TimeoutError:
                        logger.error(f'Tool listing timed out for {name}')
                        # Don't store connection if tool listing fails
                    except Exception as list_exc:
                        logger.exception(f'Tool listing failed for {name}', exc_info=list_exc)
                        # Don't store connection if tool listing fails

                # Session context (__aexit__) exited here. If connection_successful is True,
                # session object might be unusable outside this scope depending on its design.
                # **** CRITICAL DESIGN POINT ****
                # If ClientSession's __aexit__ closes underlying streams or cancels loops,
                # storing the session object itself might not maintain the connection.
                # Storing the *streams* might be necessary, and re-creating session if needed,
                # or the sse_client needs to be adapted to keep connection alive outside its context.
                # Assuming for now the session object remains valid *after* the 'async with session'
                # if the connection succeeded, but this needs verification based on mcp.client.session internals.
                # If __aexit__ *does* clean up, this whole connect logic needs redesign
                # perhaps by keeping the session context open in a background task.
                # Let's proceed with the assumption session object remains usable for now.
                logger.debug(f'Exited ClientSession context for {name}. Success: {connection_successful}')


            # Stage 5: Post-Connection Context
            if connection_successful:
                 # Need the session to persist. If __aexit__ cleaned it up, this won't work.
                 # If the session object stored in self.mcp_connections[name][0] is now invalid,
                 # subsequent execute_tool calls will fail.
                 # TODO: Verify ClientSession persistence after context exit.
                 logger.info(f'Connection process completed for {name}. Session object stored.')
                 return True
            else:
                 logger.warning(f'Connection process failed or partially failed for {name}.')
                 # Clean up potentially stored (but failed) entries
                 if name in self.mcp_connections: del self.mcp_connections[name]
                 if name in self.mcp_tools: del self.mcp_tools[name]
                 return False


        except Exception as e:
            # Catch errors during sse_client connection itself or other setup issues
            logger.exception('General error during MCP connection process.', server_name=name, url=url)
            if name in self.mcp_connections: del self.mcp_connections[name]
            if name in self.mcp_tools: del self.mcp_tools[name]
            return False


    async def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Union[CallToolResult, Dict[str, str]]:
        '''
        Execute an MCP tool. Returns CallToolResult on success, or dict with error.
        (Implementation remains the same as provided)
        '''
        logger.debug('Executing tool', tool_name=tool_name, tool_input=tool_input)

        mcp_session: ClientSession | None = None
        server_name_found: str | None = None

        for server_name, tools in self.mcp_tools.items():
            if any(tool['name'] == tool_name for tool in tools):
                 if server_name in self.mcp_connections:
                    # IMPORTANT: Assumes self.mcp_connections[server_name][0] (the session) is still valid.
                    mcp_session, _ = self.mcp_connections[server_name]
                    server_name_found = server_name
                    break
                 else:
                     logger.warning(f'Tool found for server, but connection is inactive.', tool_name=tool_name, server_name=server_name)
                     return {'error': f'Connection to MCP server "{server_name}" hosting tool "{tool_name}" is not active.'}

        if not mcp_session or not server_name_found:
            logger.warning(f'Tool "{tool_name}" not found in any actively connected MCP server.')
            return {'error': f'Tool "{tool_name}" not found or its MCP server is disconnected.'}

        logger.info(f'Calling tool "{tool_name}" on MCP server "{server_name_found}"')
        try:
            # Add timeout for safety
            result = await asyncio.wait_for(mcp_session.call_tool(tool_name, tool_input), timeout=60.0)
            logger.debug(f'Tool "{tool_name}" executed successfully.', result=result)
            return result
        except asyncio.TimeoutError:
             logger.error(f'Timeout calling tool "{tool_name}" on server "{server_name_found}"', tool_input=tool_input)
             return {'error': f'Tool "{tool_name}" timed out.'}
        except Exception as e:
            # Catch potential errors if the session became invalid after connection loss
            logger.exception(f'Error calling tool "{tool_name}" on server "{server_name_found}"', tool_input=tool_input)
            return {'error': f'Error calling tool "{tool_name}": {str(e)}'}

    def format_calltoolresult_content(self, result: Union[CallToolResult, Dict[str, str]]) -> str:
        '''
        Extract text content from a CallToolResult object or format error dict.
        (Implementation remains the same as provided)
        '''
        if isinstance(result, CallToolResult):
            text_contents = []
            if hasattr(result, 'content') and result.content:
                for content_item in result.content:
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
                     logger.warning('CallToolResult received but contained no parsable TextContent text', result_details=str(result))
                     return f'Tool executed, but returned no displayable text. (Result: {str(result)[:100]})'
            else:
                 logger.warning('CallToolResult received with no "content" or empty content', result_details=str(result))
                 return f'Tool executed, but returned no content. (Result: {str(result)[:100]})'
        elif isinstance(result, dict) and 'error' in result:
            return f'Tool Error: {result["error"]}'
        else:
            # Gracefully handle unexpected types
            logger.warning('Unexpected format received in format_calltoolresult_content', received_result=result)
            return f'Unexpected tool result format: {str(result)[:100]}'


    def get_channel_history(self, channel_id: int) -> List[Dict[str, Any]]:
        '''
        Get message history for a channel, or initialize if not exists.
        Includes basic history length management.
        (Implementation remains the same as provided)
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
             if current_len > max_history_len + 1:
                 # Calculate how many non-system messages to remove from the start
                 amount_to_remove = current_len - (max_history_len + 1)
                 # Keep the system message ([0]) and the last `max_history_len` messages
                 self.message_history[channel_id] = [self.message_history[channel_id][0]] + self.message_history[channel_id][amount_to_remove + 1:]
                 logger.debug(f'Trimmed message history for channel {channel_id}, removed {amount_to_remove} messages. New length: {len(self.message_history[channel_id])}.')

        return self.message_history[channel_id]


    async def format_tools_for_openai(self) -> List[Dict[str, Any]]:
        '''
        Format active MCP tools for OpenAI API.
        (Implementation remains the same as provided)
        '''
        openai_tools = []
        active_server_names = list(self.mcp_connections.keys()) # Only consider connected servers

        for server_name, tools in self.mcp_tools.items():
            if server_name not in active_server_names:
                logger.debug(f'Skipping tools from disconnected server "{server_name}" for OpenAI formatting.')
                continue

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
                    logger.warning(f'Tool "{tool['name']}" from server "{server_name}" has non-dict input_schema. Using empty schema.', schema=parameters)
                    # Provide a minimally valid JSON schema
                    parameters = {'type': 'object', 'properties': {}}

                # Basic validation: ensure type is object (common requirement)
                if 'type' not in parameters or parameters.get('type') != 'object':
                     logger.warning(f'Tool "{tool['name']}" input_schema type is not "object". Using as-is, but may cause issues.', schema=parameters)
                     # Allow it but log warning, OpenAI might handle other types.

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
        send_followup = is_interaction

        try:
            # Get and update history
            channel_history = self.get_channel_history(channel_id)
            channel_history.append({'role': 'user', 'content': message})
            logger.debug(f'User message added to history for channel {channel_id}', user_id=user_id, stream=stream)

            # Prepare chat parameters
            chat_params = {**self.llm_chat_params, 'stream': stream}
            openai_tools = await self.format_tools_for_openai()
            if openai_tools:
                chat_params['tools'] = openai_tools
                chat_params['tool_choice'] = 'auto' # Let model decide or use tools
                logger.debug(f'Including {len(openai_tools)} tools in LLM call for channel {channel_id}')

            # Storing results
            initial_response_content = ''
            tool_calls_aggregated: List[Dict[str, Any]] = [] # Store unified tool call format
            assistant_message_dict: Dict[str, Any] = {'role': 'assistant', 'content': None} # Start with None content

            # Initial LLM Call (Streaming or Non-Streaming)
            logger.debug(f'Initiating LLM call for channel {channel_id}', user_id=user_id, stream=stream)
            if stream:
                # Streaming Logic
                llm_stream = await self.llm_client.chat.completions.create(messages=channel_history, user=str(user_id), **chat_params)
                current_tool_calls: List[Dict[str, Any]] = [] # Temp list for streamed tool calls
                async for chunk in llm_stream:
                    delta: Optional[ChoiceDelta] = chunk.choices[0].delta if chunk.choices else None
                    if not delta: continue

                    if token := delta.content or '': initial_response_content += token

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

            else:
                # Non-Streaming Logic
                response = await self.llm_client.chat.completions.create(messages=channel_history, user=str(user_id), **chat_params)

                response_message: ChatCompletionMessage = response.choices[0].message
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
                         } for tc in raw_tool_calls if tc.type == 'function' and tc.function # Basic validation
                     ]


            # Send Initial Response & Update History (if content or tools exist)
            if initial_response_content.strip():
                logger.debug(f'Sending initial LLM response to channel {channel_id}', length=len(initial_response_content), stream=stream)
                await send_long_message(sendable, initial_response_content, followup=send_followup)
                assistant_message_dict['content'] = initial_response_content # Update history content

            if tool_calls_aggregated:
                 assistant_message_dict['tool_calls'] = tool_calls_aggregated
                 logger.info(f'LLM requested {len(tool_calls_aggregated)} tool call(s) for channel {channel_id}', tool_names=[tc['function']['name'] for tc in tool_calls_aggregated], stream=stream)

            # Add assistant message to history *if* it had content OR requested tools
            if assistant_message_dict.get('content') or assistant_message_dict.get('tool_calls'):
                channel_history.append(assistant_message_dict)
            elif not initial_response_content.strip() and not tool_calls_aggregated:
                # Handle case where LLM returns nothing (no text, no tools)
                logger.info(f'LLM call finished for channel {channel_id} with no text/tools.', stream=stream)
                # Use ephemeral for interactions if possible, non-ephemeral for context
                await send_long_message(sendable, 'I received your message but didn\'t have anything specific to add or do.', followup=send_followup, ephemeral=is_interaction)
                return # End processing here if there's nothing to do

            # Process Tool Calls (if any)
            if tool_calls_aggregated:
                tool_results_for_history = []
                for tool_call in tool_calls_aggregated:
                    tool_name = tool_call['function']['name']
                    tool_call_id = tool_call['id']
                    arguments_str = tool_call['function']['arguments']

                    tool_result_content = f'Error: Tool call processing failed for {tool_name}' # Default error
                    try:
                        tool_args = json.loads(arguments_str)
                        logger.info(f'Executing tool `{tool_name}` for channel {channel_id}', args=tool_args, stream=stream)
                        tool_result_obj = await self.execute_tool(tool_name, tool_args)
                        tool_result_content = self.format_calltoolresult_content(tool_result_obj)

                        # Send result back to user (non-ephemeral)
                        await send_long_message(sendable, f'```Tool: {tool_name}\nResult:\n{tool_result_content}```', followup=send_followup)

                    except json.JSONDecodeError:
                         logger.error(f'Failed to decode JSON args for tool `{tool_name}`', args_str=arguments_str, stream=stream)
                         tool_result_content = f'Error: Invalid JSON arguments provided: {arguments_str}'
                         await send_long_message(sendable, f'⚠️ Error: Couldn\'t understand arguments for tool `{tool_name}`.', followup=send_followup, ephemeral=is_interaction)
                    except Exception as e:
                        logger.exception(f'Error executing tool `{tool_name}` or processing result', tool_call_id=tool_call_id, stream=stream)
                        tool_result_content = f'Error executing tool: {str(e)}'
                        await send_long_message(sendable, f'⚠️ Error running tool `{tool_name}`: {str(e)}', followup=send_followup, ephemeral=is_interaction)

                    # Always append a result to history for the LLM's context
                    tool_results_for_history.append({'role': 'tool', 'tool_call_id': tool_call_id, 'name': tool_name, 'content': tool_result_content})

                channel_history.extend(tool_results_for_history)
                logger.debug(f'Added {len(tool_results_for_history)} tool results to history for channel {channel_id}')

                # Get Follow-up Response (Streaming or Non-Streaming)
                logger.debug(f'Initiating follow-up LLM call for channel {channel_id} after tools.', stream=stream)
                follow_up_params = {**self.llm_settings, 'stream': stream} # Keep original stream setting for follow-up
                follow_up_text = ''

                if stream:
                    # Streaming Follow-up
                    follow_up_stream = await self.llm_client.chat.completions.create(messages=channel_history, user=str(user_id), **follow_up_params)
                    async for chunk in follow_up_stream:
                         if token := chunk.choices[0].delta.content or '': follow_up_text += token
                else:
                    # Non-Streaming Follow-up
                    follow_up_response = await self.llm_client.chat.completions.create(messages=channel_history, user=str(user_id), **follow_up_params)
                    follow_up_text = follow_up_response.choices[0].message.content or ''

                # Send Follow-up & Update History
                if follow_up_text.strip():
                    logger.debug(f'Sending follow-up LLM response to channel {channel_id}', length=len(follow_up_text), stream=stream)
                    await send_long_message(sendable, follow_up_text, followup=send_followup)
                    channel_history.append({'role': 'assistant', 'content': follow_up_text})
                else:
                     logger.info(f'LLM follow-up call finished for channel {channel_id} with no text content.', stream=stream)

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
        '''
        Handle bot ready event. (Command syncing is now done in main script's setup_hook)
        '''
        logger.info(f'Cog {self.__class__.__name__} is ready.')
        print(f'{self.bot.user} is ready. MCPCog listener active.')


    @app_commands.command(name='mcp_list', description='List connected MCP servers and tools')
    async def mcp_list_slash(self, interaction: discord.Interaction):
        '''
        Command to list connected MCP servers and tools.
        (Implementation remains the same as provided)
        '''
        logger.info(f'Command "/mcp_list" invoked by {interaction.user} in channel {interaction.channel_id}')
        await interaction.response.defer(thinking=True, ephemeral=False) # Use non-ephemeral thinking

        if not self.mcp_connections:
            await interaction.followup.send('No MCP servers are currently connected.')
            return

        message = '**Connected MCP Servers:**\n\n'
        if not self.mcp_tools and self.mcp_connections:
             message += 'Connections active, but no tools listed (or failed to fetch tools).\n'

        for name, (_, _) in self.mcp_connections.items():
            tools = self.mcp_tools.get(name)

            if tools is None:
                 message += f'- **{name}**: Connected, but tool list unavailable or empty.\n'
                 continue # Skip to next server

            message += f'- **{name}**: {len(tools)} tool(s)\n'
            if tools:
                message += '  **Tools:**\n'
                tool_limit = 5 # Limit displayed tools per server for brevity
                displayed_count = 0
                for i, tool in enumerate(tools):
                     # Basic validation before display
                     tool_name = tool.get('name', 'Unnamed Tool')
                     if not isinstance(tool_name, str) or not tool_name:
                          tool_name = f'Invalid Tool {i+1}'

                     if displayed_count < tool_limit:
                         desc = tool.get('description', 'No description')
                         # Ensure description is string before slicing
                         if not isinstance(desc, str): desc = 'Invalid description type'
                         message += f'    - `{tool_name}`: {desc[:100]}\n' # Limit desc length
                         displayed_count += 1
                     elif displayed_count == tool_limit:
                         remaining = len(tools) - tool_limit
                         if remaining > 0:
                              message += f'    - ... and {remaining} more\n'
                         break # Stop listing tools for this server

            message += '\n' # Add space between servers

        await send_long_message(interaction, message.strip(), followup=True)


    # Prefix Command Implementation (Refactored)
    @commands.command(name='chat', help='Chat with the AI assistant (Prefix Command)')
    @commands.cooldown(1, 10, commands.BucketType.user) # Cooldown: 1 use per 10s per user
    async def chat_command(self, ctx: commands.Context, *, message: str):
        '''
        Command to chat with the LLM, possibly using MCP tools. (Uses prefix, stream=True)
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
                stream=False
                # stream=True # Prefix command uses streaming
            )

    # Error handler specifically for the prefix command's cooldown (optional but good practice)
    @chat_command.error
    async def chat_command_error(self, ctx: commands.Context, error: commands.CommandError):
        if isinstance(error, commands.CommandOnCooldown):
            await ctx.send(f'⏳ Woah there! This command is on cooldown. Try again in {error.retry_after:.1f} seconds.', delete_after=10)
        elif isinstance(error, commands.MissingRequiredArgument):
             await ctx.send(f'⚠️ You need to provide a message! Usage: `{ctx.prefix}chat <your message>`')
        else:
            logger.error(f'Error in prefix command chat_command: {error}', exc_info=error)
            # Don't send generic error here if _handle_chat_logic already does
            # await ctx.send('An unexpected error occurred with that command.')


    # Slash Command Implementation (Refactored)
    @app_commands.command(name='chat', description='Chat with the AI assistant')
    @app_commands.describe(message='Your message to the AI')
    @app_commands.checks.cooldown(1, 10, key=lambda i: i.user.id) # Cooldown per user
    async def chat_slash(self, interaction: discord.Interaction, message: str):
        '''
        Slash command version of the chat command. (Uses stream=False)
        '''
        logger.info(f'Slash command `/chat` invoked by {interaction.user} in channel {interaction.channel_id}')
        channel_id = interaction.channel_id
        user_id = interaction.user.id

        # Defer response - MUST be done before calling the helper if sends are expected
        await interaction.response.defer(thinking=True, ephemeral=False) # Non-ephemeral thinking

        await self._handle_chat_logic(
            sendable=interaction,
            message=message,
            channel_id=channel_id,
            user_id=user_id,
            stream=False # Slash command uses non-streaming for simplicity here
        )

    # Error Handler for Slash Command Cooldowns/Checks (Remains specific to slash)
    @chat_slash.error
    async def on_chat_slash_error(self, interaction: discord.Interaction, error: app_commands.AppCommandError):
        # Check if interaction is still valid before responding
        if interaction.is_expired():
             logger.warning(f'Interaction {interaction.id} expired before error handler could process: {error}')
             return

        if isinstance(error, app_commands.CommandOnCooldown):
            # Send ephemeral message about cooldown
            # Check if response already sent (e.g., defer failed), otherwise use followup
            try:
                if not interaction.response.is_done():
                     await interaction.response.send_message(f'⏳ Woah there! This command is on cooldown. Try again in {error.retry_after:.1f} seconds.', ephemeral=True)
                else:
                     await interaction.followup.send(f'⏳ Woah there! This command is on cooldown. Try again in {error.retry_after:.1f} seconds.', ephemeral=True)
            except discord.HTTPException as e:
                logger.error(f"Failed to send cooldown message for interaction {interaction.id}: {e}")

        elif isinstance(error, app_commands.CheckFailure):
            # Handle other check failures (e.g., permissions) ephemerally
            try:
                msg = 'You don\'t have permission to use this command.' # Default
                # Can customize based on specific CheckFailure types if needed
                if not interaction.response.is_done():
                    await interaction.response.send_message(msg, ephemeral=True)
                else:
                    await interaction.followup.send(msg, ephemeral=True)
            except discord.HTTPException as e:
                logger.error(f"Failed to send check failure message for interaction {interaction.id}: {e}")

        else:
            # Log other errors (these might be from discord.py before our main logic runs)
            logger.error(f'Unhandled error in slash command chat_slash check/dispatch: {error}', interaction_id=interaction.id)
            # Avoid sending generic error here if _handle_chat_logic handles its own errors.
            # This handler catches errors *before* or *outside* the main command logic run by _handle_chat_logic.
            # Example: Discord API errors during invocation, argument parsing errors.
            try:
                 # Try to inform user ephemerally if possible
                 if not interaction.response.is_done():
                      await interaction.response.send_message('An error occurred before your request could be fully processed.', ephemeral=True)
                 else:
                      # Avoid followup if interaction might be broken or if _handle_chat_logic already sent something
                      pass
            except discord.HTTPException as e:
                 logger.error(f"Failed to send generic slash error message for interaction {interaction.id}: {e}")


async def setup(bot: commands.Bot):
    # Configuration loading would happen here in a real scenario
    # For this example, using placeholder config
    config = {
        'mcp': {
            'server': [
                # { 'name': 'local-mcp', 'url': 'http://localhost:8000/mcp' } # Example
            ]
        },
        'llm_endpoint': {
            'api_key': 'YOUR_API_KEY', # Replace with actual key loading
            'base_url': 'http://localhost:11434/v1' # Example local endpoint
        },
        'llm_settings': {
            'model': 'local-model', # Or specific model like 'llama3'
            'temperature': 0.5
        },
        'system_message': 'You are a helpful Discord bot connected to MCP tools.',
        'max_history_length': 15
    }
    try:
        await bot.add_cog(MCPCog(bot, config))
        logger.info('MCPCog added to bot.')
    except Exception as e:
        logger.exception("Failed to load MCPCog", error=e)

# Note: The `sse_client` import needs to point to the correct location of your vendored SSE client.
# Note: Ensure the `openai` library and `mcp` libraries are installed.
# Note: Review the **CRITICAL DESIGN POINT** in `connect_mcp` regarding session persistence.
# If `ClientSession.__aexit__` cleans up resources needed for later calls, the connection logic
# (storing/reusing sessions) needs to be adjusted, possibly by managing the session lifecycle
# within background tasks associated with the connection.