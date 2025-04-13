'''
Discord cog for MCP integration. Connects to MCP servers and provides
commands to interact with MCP tools.
'''

import json
import asyncio
from typing import Dict, Any, List, Optional, Union # Added Union for type hinting
import discord
from discord.ext import commands
from discord import app_commands
import structlog
from ogbujipt.llm_wrapper import openai_api

from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.types import CallToolResult, TextContent

logger = structlog.get_logger(__name__)

# Helper Function for Message Splitting
async def send_long_message(sendable: Union[commands.Context, discord.Interaction], content: str, followup: bool = False):
    '''Sends a message, splitting it into multiple parts if it exceeds Discord's limit.'''
    max_length = 2000
    start = 0
    content_str = str(content)  # Be paranoid?
    if not content_str.strip(): # Don't try to send empty messages
        logger.debug('Attempted to send empty message.')
        return

    while start < len(content_str):
        end = start + max_length
        chunk = content_str[start:end]
        try:
            if isinstance(sendable, commands.Context):
                await sendable.send(chunk)
            elif isinstance(sendable, discord.Interaction):
                if followup or start > 0: # Use followup after defer or for subsequent chunks
                    await sendable.followup.send(chunk)
                else:
                    # Requires the interaction to *not* be deferred
                    # If deferred, followup must be true.
                    if sendable.response.is_done():
                         await sendable.followup.send(chunk)
                    else:
                         await sendable.response.send_message(chunk)

        except discord.HTTPException as e:
             logger.error(f'Failed to send message chunk: {e}', chunk_start=start, sendable_type=type(sendable))
             # Optionally break or try to send an error message
             try:
                  error_msg = f'Error sending part of the message: {e}'
                  if isinstance(sendable, commands.Context): await sendable.send(error_msg)
                  elif isinstance(sendable, discord.Interaction): await sendable.followup.send(error_msg, ephemeral=True)
             except Exception: pass # Avoid error loops
             break # Stop sending further chunks on error

        start = end
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

        # --- LLM Setup ---
        # Validation here as a safeguard, though load_config primarily handles it
        if 'llm_endpoint' not in config:
            logger.error('LLM endpoint configuration ("llm_endpoint") missing!')
            raise ValueError('LLM endpoint configuration missing.')
        try:
            # load_config sets defaults for base_url and api_key
            self.llm_client = openai_api(**config['llm_endpoint'])
        except Exception as e:
            logger.exception('Failed to initialize OpenAI client', endpoint=config.get('llm_endpoint'))
            raise e

        self.llm_settings = config.get('llm_settings', {})
        # Ensure required settings have defaults if not provided
        self.llm_settings.setdefault('model', 'local-model') # Default model if not set
        self.llm_settings.setdefault('temperature', 0.3)
        # 'stream' setting will be applied per-command type (True for prefix, False for slash)

        self.system_message = config.get('system_message', 'You are a helpful AI assistant. You can access tools using MCP servers.')
        logger.info('MCPCog initialized.')
        # NOTE: cog_load runs automatically when the cog is added via await bot.add_cog()

    async def cog_load(self):
        '''
        Set up MCP connections when the cog is loaded.
        This is called automatically by bot.add_cog().
        '''
        logger.info('Loading MCPCog and connecting to MCP servers...')
        print(self.config)
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
        disconnect_tasks = []
        for name, (session, connection) in self.mcp_connections.items():
            logger.info(f'Disconnecting from MCP server: {name}')
            # Add specific disconnection logic here if required by mcp library
            # e.g., if hasattr(session, 'close'): disconnect_tasks.append(session.close())
            pass

        # await asyncio.gather(*disconnect_tasks) # Uncomment if explicit close needed
        self.mcp_connections.clear()
        self.mcp_tools.clear()
        logger.info('MCPCog unloaded.')


    async def connect_mcp(self, url: str, name: str) -> bool: # Added return type hint
        '''
        Connect to an MCP server. Returns True on success, False on failure.
        '''
        logger.info(f'Attempting to connect to MCP server: {name} at {url}')
        try:
            # Assumes the ClientSession manages the connection state internally
            # after being initialized with the sse_client context. If the connection
            # object returned by sse_client needs to persist, this needs refactoring.
            async with sse_client(url) as connection:
                session = ClientSession(connection)
                await session.connect() # Ensure connection happens within the context

                # Store session. Store connection object if session needs it later.
                self.mcp_connections[name] = (session, connection) # Store both for now

                # List available tools
                result = await session.list_tools()
                tools = [
                    {
                        'name': t.name,
                        'description': t.description,
                        'input_schema': t.inputSchema,
                    }
                    for t in result.tools
                ]
                self.mcp_tools[name] = tools

                logger.info(f'Successfully connected to MCP server.', server_name=name, url=url, tool_count=len(tools))
                return True
        except Exception as e:
            logger.exception('Error connecting to MCP server.', server_name=name, url=url)
            if name in self.mcp_connections: del self.mcp_connections[name]
            if name in self.mcp_tools: del self.mcp_tools[name]
            return False

    async def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Union[CallToolResult, Dict[str, str]]:
        '''
        Execute an MCP tool. Returns CallToolResult on success, or dict with error.
        '''
        logger.debug('Executing tool', tool_name=tool_name, tool_input=tool_input)

        mcp_session: Optional[ClientSession] = None
        server_name_found: Optional[str] = None

        # Find the correct MCP session for the tool
        for server_name, tools in self.mcp_tools.items():
            if any(tool['name'] == tool_name for tool in tools):
                 if server_name in self.mcp_connections:
                    mcp_session, _ = self.mcp_connections[server_name] # Assuming session is enough
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
            # Consider adding timeout via asyncio.wait_for if session doesn't support it
            result = await asyncio.wait_for(mcp_session.call_tool(tool_name, tool_input), timeout=60.0)
            # result = await mcp_session.call_tool(tool_name, tool_input) # Without timeout
            logger.debug(f'Tool "{tool_name}" executed successfully.', result=result)
            return result
        except asyncio.TimeoutError:
             logger.error(f'Timeout calling tool "{tool_name}" on server "{server_name_found}"', tool_input=tool_input)
             return {'error': f'Tool "{tool_name}" timed out.'}
        except Exception as e:
            logger.exception(f'Error calling tool "{tool_name}" on server "{server_name_found}"', tool_input=tool_input)
            return {'error': f'Error calling tool "{tool_name}": {str(e)}'}

    def format_calltoolresult_content(self, result: Union[CallToolResult, Dict[str, str]]) -> str:
        '''
        Extract text content from a CallToolResult object or format error dict.
        '''
        if isinstance(result, CallToolResult):
            text_contents = []
            if hasattr(result, 'content') and result.content:
                for content_item in result.content:
                    if isinstance(content_item, TextContent) and hasattr(content_item, 'text'):
                        text_contents.append(str(content_item.text))
                if text_contents:
                    return '\n'.join(text_contents)
                else:
                     logger.warning('CallToolResult received but contained no TextContent.', result_details=str(result))
                     return f'Tool executed, but returned no displayable text. (Result: {str(result)[:100]})'
            else:
                 logger.warning('CallToolResult received with no "content" or empty content.', result_details=str(result))
                 return f'Tool executed, but returned no content. (Result: {str(result)[:100]})'
        elif isinstance(result, dict) and 'error' in result:
            return f'Tool Error: {result['error']}'
        else:
            logger.warning('Unexpected format received in format_calltoolresult_content', received_result=result)
            return str(result)

    def get_channel_history(self, channel_id: int) -> List[Dict[str, Any]]:
        '''
        Get message history for a channel, or initialize if not exists.
        Includes basic history length management.
        '''
        max_history_len = self.config.get('max_history_length', 20) # Make history length configurable
        if channel_id not in self.message_history:
            self.message_history[channel_id] = [
                {'role': 'system', 'content': self.system_message}
            ]
            logger.info(f'Initialized message history for channel {channel_id}')
        else:
             current_len = len(self.message_history[channel_id])
             if current_len > max_history_len + 1:
                 amount_to_remove = current_len - (max_history_len + 1)
                 self.message_history[channel_id] = [self.message_history[channel_id][0]] + self.message_history[channel_id][amount_to_remove+1:]
                 logger.debug(f'Trimmed message history for channel {channel_id}, removed {amount_to_remove} messages.')

        return self.message_history[channel_id]

    async def format_tools_for_openai(self) -> List[Dict[str, Any]]:
        '''
        Format active MCP tools for OpenAI API.
        Only includes tools from currently connected servers.
        '''
        openai_tools = []
        active_server_names = list(self.mcp_connections.keys())

        for server_name, tools in self.mcp_tools.items():
            if server_name not in active_server_names:
                logger.debug(f'Skipping tools from disconnected server "{server_name}" for OpenAI formatting.')
                continue

            for tool in tools:
                if not all(k in tool for k in ['name', 'description', 'input_schema']):
                    logger.warning(f'Skipping malformed tool definition from server "{server_name}"', tool_data=tool)
                    continue

                parameters = tool['input_schema']
                if not isinstance(parameters, dict):
                    logger.warning(f'Tool "{tool['name']}" from server "{server_name}" has non-dict input_schema. Using empty schema.', schema=parameters)
                    parameters = {'type': 'object', 'properties': {}} # Default valid schema

                openai_tool = {
                    'type': 'function',
                    'function': {
                        'name': tool['name'],
                        'description': tool['description'],
                        'parameters': parameters,
                    },
                }
                openai_tools.append(openai_tool)
        return openai_tools


    @commands.Cog.listener()
    async def on_ready(self):
        '''
        Handle bot ready event. (Command syncing is now done in main script's setup_hook)
        '''
        logger.info(f'Cog {self.__class__.__name__} is ready.')
        # Syncing is handled by setup_hook in the main bot file
        # Can add other on_ready logic specific to the cog here if needed
        print(f'{self.bot.user} is ready. MCPCog listener active.')


    @app_commands.command(name='mcp_list', description='List connected MCP servers and tools')
    async def mcp_list_slash(self, interaction: discord.Interaction):
        '''
        Command to list connected MCP servers and tools.
        '''
        logger.info(f'Command "/mcp_list" invoked by {interaction.user} in channel {interaction.channel_id}')
        await interaction.response.defer(thinking=True, ephemeral=False)

        if not self.mcp_connections:
            await interaction.followup.send('No MCP servers are currently connected.')
            return

        message = '**Connected MCP Servers:**\n\n'
        if not self.mcp_tools and self.mcp_connections:
             message += 'Connections active, but no tools listed (or failed to fetch tools).\n'

        for name, (_, _) in self.mcp_connections.items():
            # Check if tools were successfully retrieved for this connection
            tools = self.mcp_tools.get(name) # Use .get() for safety

            if tools is None: # Explicitly check if tools haven't been loaded for this conn
                 message += f'- **{name}**: Connected, but tool list unavailable or empty.\n'
                 continue

            message += f'- **{name}**: {len(tools)} tool(s)\n'
            if tools:
                message += '  **Tools:**\n'
                tool_limit = 5
                for i, tool in enumerate(tools):
                    if i < tool_limit:
                         desc = tool.get('description', 'No description')[:100]
                         message += f'    - `{tool.get('name', 'Unnamed Tool')}`: {desc}\n'
                    elif i == tool_limit:
                         message += f'    - ... and {len(tools) - tool_limit} more\n'
                         break
            message += '\n'

        await send_long_message(interaction, message.strip(), followup=True)


    # Prefix Command Implementation
    @commands.command(name='chat', help='Chat with the AI assistant (Prefix Command)')
    @commands.cooldown(1, 10, commands.BucketType.user) # Cooldown: 1 use per 10s per user
    async def chat_command(self, ctx: commands.Context, *, message: str):
        '''
        Command to chat with the LLM, possibly using MCP tools. (Uses prefix)
        '''
        logger.info(f'Prefix command "!chat" invoked by {ctx.author} in channel {ctx.channel.id}')
        channel_id = ctx.channel.id
        user_id = ctx.author.id

        async with ctx.typing(): # Shows 'Bot is typing...'
            # 1. Get and update history
            channel_history = self.get_channel_history(channel_id)
            channel_history.append({'role': 'user', 'content': message})
            logger.debug(f'User message added to history for channel {channel_id}', user_id=user_id)


            # 2. Prepare chat parameters (Stream=True for prefix)
            chat_params = {**self.llm_settings, 'stream': True}
            openai_tools = await self.format_tools_for_openai()
            if openai_tools:
                chat_params['tools'] = openai_tools
                chat_params['tool_choice'] = 'auto'
                logger.debug(f'Including {len(openai_tools)} tools in LLM call for channel {channel_id}')

            try:
                # 3. Initial LLM Call (Streaming)
                logger.debug(f'Initiating LLM stream for channel {channel_id}', user_id=user_id)
                stream = await self.llm_client.chat.completions.create(
                    messages=channel_history,
                    user=str(user_id), # Pass user ID if API supports it
                    **chat_params
                )

                initial_response_content = ''
                tool_calls_aggregated = []
                current_tool_calls = [] # Temp list to build streamed tool calls

                async for chunk in stream:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if not delta: continue

                    if token := delta.content or '': initial_response_content += token

                    # Aggregate tool calls robustly from stream
                    if delta.tool_calls:
                        for tool_call_chunk in delta.tool_calls:
                            idx = tool_call_chunk.index
                            while len(current_tool_calls) <= idx:
                                current_tool_calls.append({'id': None, 'type': 'function', 'function': {'name': '', 'arguments': ''}})

                            tc_ref = current_tool_calls[idx] # Reference for easier access
                            if tool_call_chunk.id: tc_ref['id'] = tool_call_chunk.id
                            if tool_call_chunk.function:
                                if tool_call_chunk.function.name: tc_ref['function']['name'] += tool_call_chunk.function.name
                                if tool_call_chunk.function.arguments: tc_ref['function']['arguments'] += tool_call_chunk.function.arguments

                # Finalize valid tool calls
                tool_calls_aggregated = [tc for tc in current_tool_calls if tc.get('id') and tc['function']['name']]


                # 4. Send Initial Response & Update History
                assistant_message = {'role': 'assistant', 'content': initial_response_content if initial_response_content else None}
                if tool_calls_aggregated:
                    assistant_message['tool_calls'] = tool_calls_aggregated
                    logger.info(f'LLM requested {len(tool_calls_aggregated)} tool call(s) for channel {channel_id}', tool_names=[tc['function']['name'] for tc in tool_calls_aggregated])

                if initial_response_content.strip():
                     logger.debug(f'Sending initial LLM response to channel {channel_id}', length=len(initial_response_content))
                     await send_long_message(ctx, initial_response_content)
                     channel_history.append(assistant_message) # Append history if content exists
                elif not tool_calls_aggregated:
                     logger.info(f'LLM stream finished for channel {channel_id} with no text/tools.')
                     await ctx.send('I received your message but didn\'t have anything specific to add or do.')

                # 5. Process Tool Calls
                if tool_calls_aggregated:
                    if not initial_response_content.strip():
                        channel_history.append(assistant_message) # Append even if no content, important for tool context

                    tool_results_for_history = []
                    for tool_call in tool_calls_aggregated:
                        tool_name = tool_call['function']['name']
                        tool_call_id = tool_call['id']
                        arguments_str = tool_call['function']['arguments']

                        try:
                            tool_args = json.loads(arguments_str)
                            logger.info(f'Executing tool `{tool_name}` for channel {channel_id}', args=tool_args)
                            tool_result_obj = await self.execute_tool(tool_name, tool_args)
                            tool_result_content = self.format_calltoolresult_content(tool_result_obj)

                            await send_long_message(ctx, f'```Tool: {tool_name}\nResult:\n{tool_result_content}```')

                            tool_results_for_history.append({'role': 'tool', 'tool_call_id': tool_call_id, 'name': tool_name, 'content': tool_result_content})

                        except json.JSONDecodeError:
                             logger.error(f'Failed to decode JSON args for tool `{tool_name}`', args_str=arguments_str)
                             await ctx.send(f'⚠️ Error: Couldn\'t understand arguments for tool `{tool_name}`.')
                             tool_results_for_history.append({'role': 'tool', 'tool_call_id': tool_call_id, 'name': tool_name, 'content': f'Error: Invalid JSON args: {arguments_str}'})
                        except Exception as e:
                            logger.exception(f'Error executing tool `{tool_name}` or processing result', tool_call_id=tool_call_id)
                            await ctx.send(f'⚠️ Error running tool `{tool_name}`: {str(e)}')
                            tool_results_for_history.append({'role': 'tool', 'tool_call_id': tool_call_id, 'name': tool_name, 'content': f'Error executing tool: {str(e)}'})

                    channel_history.extend(tool_results_for_history)
                    logger.debug(f'Added {len(tool_results_for_history)} tool results to history for channel {channel_id}')

                    # 6. Get Follow-up Response (Stream again)
                    logger.debug(f'Initiating follow-up LLM stream for channel {channel_id} after tools.')
                    follow_up_params = {**self.llm_settings, 'stream': True} # Stream follow-up
                    follow_up_stream = await self.llm_client.chat.completions.create(messages=channel_history, user=str(user_id), **follow_up_params)

                    follow_up_text = ''
                    async for chunk in follow_up_stream:
                         if token := chunk.choices[0].delta.content or '': follow_up_text += token

                    # 7. Send Follow-up & Update History
                    if follow_up_text.strip():
                        logger.debug(f'Sending follow-up LLM response to channel {channel_id}', length=len(follow_up_text))
                        await send_long_message(ctx, follow_up_text)
                        channel_history.append({'role': 'assistant', 'content': follow_up_text})
                    else:
                         logger.info(f'LLM follow-up stream finished for channel {channel_id} with no text content.')

            except Exception as e:
                logger.exception(f'Unhandled error during chat command execution for channel {channel_id}', user_id=user_id)
                await ctx.send(f'An unexpected error occurred: {str(e)}')


    # Slash Command Implementation
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

        await interaction.response.defer(thinking=True, ephemeral=False) # Defer response

        # 1. Get and update history
        channel_history = self.get_channel_history(channel_id)
        channel_history.append({'role': 'user', 'content': message})
        logger.debug(f'User message added to history for channel {channel_id}', user_id=user_id)

        # 2. Prepare chat parameters (Stream=False for slash command simplicity)
        chat_params = {**self.llm_settings, 'stream': False}
        openai_tools = await self.format_tools_for_openai()
        if openai_tools:
            chat_params['tools'] = openai_tools
            chat_params['tool_choice'] = 'auto'
            logger.debug(f'Including {len(openai_tools)} tools in LLM call for channel {channel_id}')

        try:
            # 3. Initial LLM Call (Non-Streaming)
            logger.debug(f'Initiating non-streaming LLM call for channel {channel_id}', user_id=user_id)
            response = await self.llm_client.chat.completions.create(
                messages=channel_history, user=str(user_id), **chat_params
            )

            response_message = response.choices[0].message
            initial_response_content = response_message.content or ''
            tool_calls_data = response_message.tool_calls or []

            # 4. Send Initial Response & Update History
            assistant_message = {'role': 'assistant', 'content': initial_response_content if initial_response_content else None}
            if tool_calls_data:
                assistant_message['tool_calls'] = [{'id': tc.id, 'type': 'function', 'function': {'name': tc.function.name, 'arguments': tc.function.arguments}} for tc in tool_calls_data]
                logger.info(f'LLM requested {len(tool_calls_data)} tool call(s) for channel {channel_id}', tool_names=[tc.function.name for tc in tool_calls_data])

            if initial_response_content.strip():
                logger.debug(f'Sending initial LLM response via followup to channel {channel_id}', length=len(initial_response_content))
                await send_long_message(interaction, initial_response_content, followup=True)
                channel_history.append(assistant_message)
            elif not tool_calls_data:
                 logger.info(f'LLM call finished for channel {channel_id} with no text/tools.')
                 await interaction.followup.send('I received your message but didn\'t have anything specific to add or do.', ephemeral=False)

            # 5. Process Tool Calls
            if tool_calls_data:
                if not initial_response_content.strip():
                    channel_history.append(assistant_message) # Add tool request message to history

                tool_results_for_history = []
                for tool_call in tool_calls_data:
                    tool_name = tool_call.function.name
                    tool_call_id = tool_call.id
                    arguments_str = tool_call.function.arguments

                    try:
                        tool_args = json.loads(arguments_str)
                        logger.info(f'Executing tool `{tool_name}` for channel {channel_id} (slash)', args=tool_args)
                        tool_result_obj = await self.execute_tool(tool_name, tool_args)
                        tool_result_content = self.format_calltoolresult_content(tool_result_obj)

                        await send_long_message(interaction, f'```Tool: {tool_name}\nResult:\n{tool_result_content}```', followup=True)

                        tool_results_for_history.append({'role': 'tool', 'tool_call_id': tool_call_id, 'name': tool_name, 'content': tool_result_content})

                    except json.JSONDecodeError:
                         logger.error(f'Failed to decode JSON args for tool `{tool_name}` (slash)', args_str=arguments_str)
                         await interaction.followup.send(f'⚠️ Error: Couldn\'t understand arguments for tool `{tool_name}`.', ephemeral=False)
                         tool_results_for_history.append({'role': 'tool', 'tool_call_id': tool_call_id, 'name': tool_name, 'content': f'Error: Invalid JSON args: {arguments_str}'})
                    except Exception as e:
                        logger.exception(f'Error executing tool `{tool_name}` or processing result (slash)', tool_call_id=tool_call_id)
                        await interaction.followup.send(f'⚠️ Error running tool `{tool_name}`: {str(e)}', ephemeral=False)
                        tool_results_for_history.append({'role': 'tool', 'tool_call_id': tool_call_id, 'name': tool_name, 'content': f'Error executing tool: {str(e)}'})

                channel_history.extend(tool_results_for_history)
                logger.debug(f'Added {len(tool_results_for_history)} tool results to history for channel {channel_id}')

                # 6. Get Follow-up Response (Non-Streaming)
                logger.debug(f'Initiating follow-up LLM call for channel {channel_id} after tools (slash).')
                follow_up_params = {**self.llm_settings, 'stream': False} # No stream for follow-up either
                follow_up_response = await self.llm_client.chat.completions.create(messages=channel_history, user=str(user_id), **follow_up_params)
                follow_up_text = follow_up_response.choices[0].message.content or ''

                # 7. Send Follow-up & Update History
                if follow_up_text.strip():
                    logger.debug(f'Sending follow-up LLM response via followup to channel {channel_id}', length=len(follow_up_text))
                    await send_long_message(interaction, follow_up_text, followup=True)
                    channel_history.append({'role': 'assistant', 'content': follow_up_text})
                else:
                    logger.info(f'LLM follow-up call finished for channel {channel_id} with no text content.')

        except Exception as e:
            logger.exception(f'Unhandled error during slash command execution for channel {channel_id}', user_id=user_id)
            try:
                 # Check if interaction is still valid before sending followup
                 if not interaction.is_expired():
                      await interaction.followup.send(f'An unexpected error occurred processing your request: {str(e)}', ephemeral=True)
                 else:
                      logger.warning(f'Interaction {interaction.id} expired before error could be sent.')
            except discord.NotFound:
                 logger.warning(f'Interaction {interaction.id} likely expired or deleted before error followup.')
            except Exception as followup_e:
                 logger.exception('Failed to send error message via followup', original_error=str(e))

    # Error Handler for Slash Command Cooldowns/Errors
    @chat_slash.error
    async def on_chat_slash_error(self, interaction: discord.Interaction, error: app_commands.AppCommandError):
        # Check if interaction is still valid before responding
        if interaction.is_expired():
             logger.warning(f'Interaction {interaction.id} expired before error handler could process: {error}')
             return

        if isinstance(error, app_commands.CommandOnCooldown):
            # Send ephemeral message about cooldown
            await interaction.response.send_message(f'⏳ Woah there! This command is on cooldown. Try again in {error.retry_after:.1f} seconds.', ephemeral=True)
        elif isinstance(error, app_commands.CheckFailure):
            # Handle other check failures if more are added (e.g., permissions)
            await interaction.response.send_message('You don\'t have permission to use this command.', ephemeral=True)
        else:
            # Log other errors and inform user generically
            logger.error(f'Unhandled error in chat_slash command check/execution: {error}', interaction_id=interaction.id)
            # Use response if not already done, otherwise followup
            try:
                if not interaction.response.is_done():
                    await interaction.response.send_message('An unexpected error occurred with that command.', ephemeral=True)
                else:
                    # Avoid sending followup if original followup failed leading here
                    pass # Already logged, avoid further interaction errors
                    # await interaction.followup.send('An unexpected error occurred executing that command.', ephemeral=True)
            except Exception as handler_e:
                 logger.error(f'Error within the error handler itself: {handler_e}')
