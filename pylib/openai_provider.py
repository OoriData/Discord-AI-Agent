# SPDX-FileCopyrightText: 2025-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# discord_aiagent.openai_provider

import json
import os
import uuid
from typing import Any, Dict, List
from openai import OpenAIError
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall, ChoiceDelta

import structlog
from .llm_providers import LLMProvider
from .openaiutil import (
    extract_tool_calls_from_content, 
    format_calltoolresult_content,
    _log_openai_error_details,
    MissingLLMResponseError,
    LLMResponseError,
    ToolCallExecutionError
)

logger = structlog.get_logger(__name__)


class OpenAIProvider(LLMProvider):
    """
    OpenAI-compatible LLM provider implementation.
    Works with OpenAI API, Azure OpenAI, and other OpenAI-compatible endpoints.
    """
    
    def __init__(self, init_params: Dict[str, Any], chat_params: Dict[str, Any], tool_handler):
        super().__init__(init_params, chat_params, tool_handler)
        
        # Get API key from environment if not provided
        if 'api_key' not in init_params or init_params['api_key'] == 'missing-key':
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key not found in config or OPENAI_API_KEY environment variable")
            init_params['api_key'] = api_key
        
        self.llm_client = AsyncOpenAI(**init_params)
        logger.info('OpenAI client initialized', **init_params)

    async def __call__(self, messages: List[Dict], user: str, extra_chat_params: Dict, stream: bool = False):
        """
        Perform the LLM call using the provided parameters and channel history
        """
        initial_response_content = ''
        tool_calls_aggregated: List[Dict[str, Any]] = []
        chat_params = {**self.chat_params, **extra_chat_params}  # Combine endpoint & explicit/extra settings

        logger.debug('LLM call', user=user, stream=stream, model=chat_params.get('model'))
        try:
            if stream:
                llm_stream = await self.llm_client.chat.completions.create(messages=messages, user=user, **chat_params)
                current_tool_calls: List[Dict[str, Any]] = []
                async for chunk in llm_stream:
                    delta: ChoiceDelta | None = chunk.choices[0].delta if chunk.choices else None
                    if not delta: continue
                    if token := delta.content or '': initial_response_content += token
                    if delta.tool_calls:
                        for tool_call_chunk in delta.tool_calls:
                            idx = tool_call_chunk.index
                            while len(current_tool_calls) <= idx:
                                current_tool_calls.append({'id': None, 'type': 'function', 'function': {'name': '', 'arguments': ''}})
                            tc_ref = current_tool_calls[idx]
                            chunk_func: ChoiceDeltaToolCall.Function | None = tool_call_chunk.function
                            if tool_call_chunk.id: tc_ref['id'] = tool_call_chunk.id
                            if chunk_func:
                                if chunk_func.name: tc_ref['function']['name'] += chunk_func.name
                                if chunk_func.arguments: tc_ref['function']['arguments'] += chunk_func.arguments
                tool_calls_aggregated = [tc for tc in current_tool_calls if tc.get('id') and tc['function'].get('name')]
            else:  # Non-Streaming
                response = await self.llm_client.chat.completions.create(messages=messages, user=user, **chat_params)
                response_message: ChatCompletionMessage | None = response.choices[0].message if response.choices else None
                if not response_message:
                    logger.error('LLM response missing message object', response_data=response.model_dump_json(indent=2))
                    raise MissingLLMResponseError('⚠️ Received an empty response from AI')
                initial_response_content = response_message.content or ''
                raw_tool_calls: List[ChatCompletionMessageToolCall] | None = response_message.tool_calls
                if raw_tool_calls:
                    tool_calls_aggregated = [{'id': tc.id, 'type': tc.type, 'function': {'name': tc.function.name, 'arguments': tc.function.arguments}}
                                            for tc in raw_tool_calls if tc.type == 'function' and tc.function and tc.id and tc.function.name]
                    logger.debug(f'Non-streaming response included {len(tool_calls_aggregated)} raw tool calls.')
                elif '<tool_call>' in initial_response_content:
                    # Fallback for legacy tool call format
                    logger.warning('LLM generated tool call as text, attempting to parse.')
                    parsed_calls, updated_content = extract_tool_calls_from_content(initial_response_content)
                    if parsed_calls:
                        tool_calls_aggregated, initial_response_content = parsed_calls, updated_content

        except OpenAIError as exc:  # Catch specific OpenAI errors
            error_message = _log_openai_error_details(exc, 'initial LLM call', self.connection_info)
            raise LLMResponseError(error_message)
        except Exception as e:  # Catch other unexpected errors (e.g., network)
            logger.exception('Unexpected error during LLM communication')
            raise LLMResponseError(f'⚠️ Unexpected error communicating with AI: {str(e)}')

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

                # Add tool result to history
                tool_results_for_history.extend([
                    {
                        'role': 'assistant',
                        'content': None,
                        'tool_calls': [tool_call]
                    },
                    {
                        'role': 'tool',
                        'content': tool_result_content_formatted,
                        'tool_call_id': tool_call_id
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
                        'role': 'assistant',
                        'content': None,
                        'tool_calls': [tool_call]
                    },
                    {
                        'role': 'tool',
                        'content': error_message,
                        'tool_call_id': tool_call_id
                    }
                ])

        # Follow-up LLM Call
        if tool_results_for_history:
            messages.extend(tool_results_for_history)
            logger.debug(f'Added {len(tool_results_for_history)} tool results to history', thread_id=thread_id)

            logger.debug('Initiating follow-up LLM call after tools.', thread_id=thread_id)
            follow_up_params = {**self.chat_params, 'stream': stream}
            follow_up_params.pop('tools', None)  # No tools needed for follow-up
            follow_up_params.pop('tool_choice', None)

            follow_up_text = ''
            try:
                if stream:
                    follow_up_stream = await self.llm_client.chat.completions.create(messages=messages, user=user, **follow_up_params)
                    async for chunk in follow_up_stream:
                        if token := chunk.choices[0].delta.content or '': follow_up_text += token
                else:
                    follow_up_response = await self.llm_client.chat.completions.create(messages=messages, user=user, **follow_up_params)
                    follow_up_message = follow_up_response.choices[0].message if follow_up_response.choices else None
                    if follow_up_message: follow_up_text = follow_up_message.content or ''

            except OpenAIError as exc:  # Catch specific OpenAI errors
                error_message = _log_openai_error_details(exc, 'follow-up LLM call', self.connection_info)
                raise MissingLLMResponseError(error_message)

        return follow_up_text 