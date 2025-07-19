# SPDX-FileCopyrightText: 2025-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# discord_aiagent.claude_provider

import json
import os
from typing import Any, Dict, List
import anthropic
from anthropic import AnthropicError

import structlog
from .llm_providers import LLMProvider
from .openaiutil import (
    format_calltoolresult_content,
    MissingLLMResponseError,
    LLMResponseError
)

logger = structlog.get_logger(__name__)


class ClaudeProvider(LLMProvider):
    """
    Anthropic Claude LLM provider implementation.
    """
    
    def __init__(self, init_params: Dict[str, Any], chat_params: Dict[str, Any], tool_handler):
        super().__init__(init_params, chat_params, tool_handler)
        
        # Get API key from environment if not provided
        if 'api_key' not in init_params or init_params['api_key'] == 'missing-key':
            api_key = os.environ.get('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("Anthropic API key not found in config or ANTHROPIC_API_KEY environment variable")
            init_params['api_key'] = api_key
        
        # Remove base_url from init_params as Anthropic doesn't use it
        init_params.pop('base_url', None)
        
        self.llm_client = anthropic.AsyncAnthropic(**init_params)
        logger.info('Anthropic Claude client initialized', api_key_present=bool(init_params.get('api_key')))

    def _convert_messages_to_claude_format(self, messages: List[Dict]) -> List[Dict]:
        """
        Convert OpenAI-style messages to Claude format.
        Claude uses 'user' and 'assistant' roles, and handles system messages differently.
        """
        claude_messages = []
        system_content = None
        
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            # Skip assistant messages with empty content (except for the final one)
            if role == 'assistant' and (not content or not content.strip()):
                continue
                
            if role == 'system':
                # Collect system message to prepend to first user message
                if system_content is None:
                    system_content = content
                else:
                    system_content += '\n\n' + content
            elif role == 'user':
                # If we have a system message, prepend it to the first user message
                if system_content and not claude_messages:
                    content = f"{system_content}\n\n{content}"
                claude_messages.append({'role': 'user', 'content': content})
            elif role == 'assistant':
                # Only add assistant messages with non-empty content
                # Claude doesn't allow empty content except for the final assistant message
                if content and content.strip():
                    claude_messages.append({'role': 'assistant', 'content': content})
            elif role == 'tool':
                # Claude doesn't have a 'tool' role, so we'll format tool results as user messages
                tool_call_id = msg.get('tool_call_id', 'unknown')
                claude_messages.append({
                    'role': 'user', 
                    'content': f"[Tool result for {tool_call_id}]: {content}"
                })
        
        return claude_messages

    def _convert_tools_to_claude_format(self, tools: List[Dict]) -> List[Dict]:
        """
        Convert OpenAI-style tool definitions to Claude format.
        """
        claude_tools = []
        for tool in tools:
            if tool.get('type') == 'function':
                function = tool.get('function', {})
                claude_tool = {
                    'name': function.get('name'),
                    'description': function.get('description', ''),
                    'input_schema': function.get('parameters', {})
                }
                claude_tools.append(claude_tool)
        return claude_tools

    def _extract_tool_calls_from_claude_response(self, response) -> List[Dict]:
        """
        Extract tool calls from Claude's response format.
        """
        tool_calls = []
        if hasattr(response, 'content') and response.content:
            for content_block in response.content:
                if content_block.type == 'tool_use':
                    tool_calls.append({
                        'id': content_block.id,
                        'type': 'function',
                        'function': {
                            'name': content_block.name,
                            'arguments': json.dumps(content_block.input)
                        }
                    })
        return tool_calls

    async def __call__(self, messages: List[Dict], user: str, extra_chat_params: Dict, stream: bool = False):
        """
        Perform the LLM call using Claude API.
        """
        initial_response_content = ''
        tool_calls_aggregated: List[Dict[str, Any]] = []
        
        # Convert messages to Claude format
        claude_messages = self._convert_messages_to_claude_format(messages)
        
        # Prepare parameters
        params = {
            'model': self.chat_params.get('model', 'claude-3-5-sonnet-20241022'),
            'max_tokens': self.chat_params.get('max_tokens', 1000),
            'temperature': self.chat_params.get('temperature', 0.3),
        }
        
        # Add tools if present
        if 'tools' in extra_chat_params:
            claude_tools = self._convert_tools_to_claude_format(extra_chat_params['tools'])
            if claude_tools:
                params['tools'] = claude_tools
                # Claude doesn't use tool_choice parameter like OpenAI
                # It automatically decides when to use tools based on the conversation
        
        logger.debug('Claude LLM call', user=user, stream=stream, model=params.get('model'))
        
        try:
            if stream:
                # Claude streaming
                stream_response = await self.llm_client.messages.create(
                    messages=claude_messages,
                    stream=True,
                    **params
                )
                
                current_tool_calls: List[Dict[str, Any]] = []
                async for chunk in stream_response:
                    if chunk.type == 'content_block_delta':
                        if chunk.delta.type == 'text_delta':
                            initial_response_content += chunk.delta.text
                    elif chunk.type == 'tool_use_delta':
                        # Handle tool use deltas
                        tool_use_id = chunk.id
                        tool_use_name = chunk.delta.name
                        tool_use_input = chunk.delta.input
                        
                        # Find or create tool call
                        tool_call = next((tc for tc in current_tool_calls if tc.get('id') == tool_use_id), None)
                        if not tool_call:
                            tool_call = {
                                'id': tool_use_id,
                                'type': 'function',
                                'function': {'name': '', 'arguments': ''}
                            }
                            current_tool_calls.append(tool_call)
                        
                        if tool_use_name:
                            tool_call['function']['name'] = tool_use_name
                        if tool_use_input:
                            tool_call['function']['arguments'] = json.dumps(tool_use_input)
                
                tool_calls_aggregated = [tc for tc in current_tool_calls if tc.get('id') and tc['function'].get('name')]
                
            else:
                # Non-streaming
                response = await self.llm_client.messages.create(
                    messages=claude_messages,
                    **params
                )
                
                # Extract content
                if response.content:
                    for content_block in response.content:
                        if content_block.type == 'text':
                            initial_response_content += content_block.text
                
                # Extract tool calls
                tool_calls_aggregated = self._extract_tool_calls_from_claude_response(response)

        except AnthropicError as exc:
            logger.error(f'Anthropic API error during LLM call: {exc}', 
                        base_url=self.connection_info.get('base_url'),
                        model=self.connection_info.get('model'))
            raise LLMResponseError(f'⚠️ Claude API error: {str(exc)}')
        except Exception as e:
            logger.exception('Unexpected error during Claude LLM communication')
            raise LLMResponseError(f'⚠️ Unexpected error communicating with Claude: {str(e)}')

        return initial_response_content, tool_calls_aggregated

    async def process_tool_calls(self, tool_calls_aggregated: List[Dict], messages: List[Dict], 
                               thread_id: str, user: str, tool_call_result_hook, stream: bool = False):
        """
        Process tool calls and return follow-up response.
        """
        if not tool_calls_aggregated:
            logger.debug('No tool calls to process', thread_id=thread_id)
            return ''

        tool_results_for_history: List[Dict[str, Any]] = []

        # Process each tool call
        for tool_call in tool_calls_aggregated:
            tool_call_id = tool_call.get('id')
            tool_name = tool_call.get('function', {}).get('name')
            tool_arguments_str = tool_call.get('function', {}).get('arguments', '{}')

            if not tool_call_id or not tool_name:
                logger.warning('Invalid tool call structure', tool_call=tool_call, thread_id=thread_id)
                continue

            logger.debug('Processing tool call', tool_name=tool_name, tool_call_id=tool_call_id, thread_id=thread_id)

            try:
                # Parse tool arguments
                try:
                    tool_arguments = json.loads(tool_arguments_str) if isinstance(tool_arguments_str, str) else tool_arguments_str
                except json.JSONDecodeError as json_err:
                    logger.error('Failed to parse tool arguments as JSON', tool_arguments=tool_arguments_str, error=str(json_err), thread_id=thread_id)
                    tool_arguments = {}

                # Execute the tool
                tool_result = await self.tool_handler.execute_tool(tool_name, tool_arguments)
                tool_result_content_formatted = format_calltoolresult_content(tool_result)

                # Call the result hook
                await tool_call_result_hook(tool_name, tool_call_id, tool_result_content_formatted)

                # Add tool result to history (Claude format)
                # Claude doesn't support tool_calls field in messages, so we format it as a user message
                tool_results_for_history.extend([
                    {
                        'role': 'user',
                        'content': f"[Tool result for {tool_call_id}]: {tool_result_content_formatted}"
                    }
                ])

            except Exception as exc:
                logger.exception('Error executing tool', tool_name=tool_name, tool_call_id=tool_call_id, thread_id=thread_id)
                error_message = f'Error executing tool {tool_name}: {str(exc)}'
                
                # Call the result hook with error
                await tool_call_result_hook(tool_name, tool_call_id, error_message, exc=exc)
                
                # Add error result to history
                tool_results_for_history.extend([
                    {
                        'role': 'user',
                        'content': f"[Tool result for {tool_call_id}]: {error_message}"
                    }
                ])

        # Follow-up LLM Call
        if tool_results_for_history:
            messages.extend(tool_results_for_history)
            logger.debug(f'Added {len(tool_results_for_history)} tool results to history', thread_id=thread_id)

            logger.debug('Initiating follow-up Claude call after tools.', thread_id=thread_id)
            follow_up_params = {
                'model': self.chat_params.get('model', 'claude-3-5-sonnet-20241022'),
                'max_tokens': self.chat_params.get('max_tokens', 1000),
                'temperature': self.chat_params.get('temperature', 0.3),
            }

            # Convert messages to Claude format
            claude_messages = self._convert_messages_to_claude_format(messages)

            follow_up_text = ''
            try:
                if stream:
                    follow_up_stream = await self.llm_client.messages.create(
                        messages=claude_messages,
                        stream=True,
                        **follow_up_params
                    )
                    async for chunk in follow_up_stream:
                        if chunk.type == 'content_block_delta':
                            if chunk.delta.type == 'text_delta':
                                follow_up_text += chunk.delta.text
                else:
                    follow_up_response = await self.llm_client.messages.create(
                        messages=claude_messages,
                        **follow_up_params
                    )
                    if follow_up_response.content:
                        for content_block in follow_up_response.content:
                            if content_block.type == 'text':
                                follow_up_text += content_block.text

            except AnthropicError as exc:
                logger.error(f'Anthropic API error during follow-up call: {exc}')
                raise MissingLLMResponseError(f'⚠️ Claude API error during follow-up: {str(exc)}')

        return follow_up_text 