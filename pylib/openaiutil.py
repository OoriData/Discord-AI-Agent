# SPDX-FileCopyrightText: 2025-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# discord_aiagent.openaiutil
import json
import uuid
from typing import Any
from openai import OpenAIError
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall, ChoiceDelta

import structlog

logger = structlog.get_logger(__name__)

def replace_system_prompt(messages, sysprommpt):
    if messages[0]['role'] == 'system':
        messages[0]['content'] = sysprommpt  # Yeah, just clobber it
    else:
        messages.insert(0, {'role': 'system', 'content': sysprommpt})
    return


def extract_tool_calls_from_content(content: str) -> tuple[list[dict], str]:
    ''' Extracts tool calls from the content string. '''
    import re
    import json  # Ensure json is imported

    parsed_tool_calls = []
    # Basic regex to find the JSON within the tags (adjust if needed)
    matches = re.finditer(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL)
    for match in matches:
        try:
            tool_json_str = match.group(1).strip()
            tool_data = json.loads(tool_json_str)
            # Need to invent a tool_call_id if not provided by LLM text
            tool_call_id = f'parsed_{uuid.uuid4()}'
            if 'name' in tool_data and 'arguments' in tool_data:
                 parsed_tool_calls.append({
                     'id': tool_call_id,
                     'type': 'function',  # Assume function
                     'function': {
                         'name': tool_data['name'],
                         # Ensure arguments are stringified if needed by later code
                         'arguments': json.dumps(tool_data['arguments']) if isinstance(tool_data['arguments'], dict) else str(tool_data['arguments'])
                     }
                 })
                 logger.info(f'Parsed text tool call: {tool_data['name']}')
            else:
                 logger.warning('Parsed tool call text missing name or arguments.', parsed_text=tool_json_str)
        except json.JSONDecodeError:
            logger.error('Failed to decode JSON from text tool call.', matched_text=match.group(1))
        except Exception as parse_err:
            logger.exception('Error parsing text tool call.', error=parse_err)

    if parsed_tool_calls:
        # Optionally remove the <tool_call> text from initial_response_content
        updated_content = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL).strip()
        logger.info(f'Proceeding with {len(parsed_tool_calls)} tool calls parsed from text.')
        return parsed_tool_calls, updated_content
    return [], content  # Return empty list and original content if no calls found/parsed


def format_calltoolresult_content(result: dict[str, Any] | dict[str, str]) -> str:
    '''
    Extract content from a tool result dict.
    Handles common MCP content structures like text content.
    '''
    if isinstance(result, dict):
        # Handle RSS result or any error dict
        if 'result' in result:
            return str(result['result'])  # Return the formatted RSS result string
        elif 'error' in result:
            return f'Tool Error: {result['error']}'  # Return formatted error
        elif 'content' in result:
            content = result['content']
            if content is None:
                return 'Tool executed successfully but returned no content.'
            
            if isinstance(content, dict) and content.get('type') == 'text' and 'text' in content:
                text_content = content['text']
                # Return the text directly, converting simple types to string
                return str(text_content) if text_content is not None else ''
            elif isinstance(content, (str, int, float, bool)):
                return str(content)
            elif isinstance(content, (dict, list)):
                try:
                    # Nicely format other JSON-like structures
                    return json.dumps(content, indent=2)
                except TypeError as json_err:
                    logger.warning('Could not JSON serialize MCP tool result content.', content_type=type(content), error=str(json_err))
                    return str(content)  # Fallback
            else:
                logger.warning('Unexpected type for MCP tool result content.', content_type=type(content))
                return str(content)
        else:
            # Gracefully handle unexpected dict format
            logger.warning('Unexpected dict format received in format_calltoolresult_content', received_result=result)
            try: 
                return json.dumps(result, indent=2)
            except TypeError: 
                return f'Unexpected tool result format: {str(result)[:200]}'
    else:
        # Gracefully handle unexpected types
        logger.warning('Unexpected format received in format_calltoolresult_content', received_result=result)
        return f'Unexpected tool result format: {str(result)[:200]}'


def _log_openai_error_details(error: OpenAIError, context: str, connection_info: dict[str, Any]) -> str:
    """
    Log detailed information about OpenAI errors and return a user-friendly message.
    
    Args:
        error: The OpenAIError that occurred
        context: Context string (e.g., 'initial LLM call', 'follow-up LLM call')
        connection_info: Dict containing connection details (base_url, model, etc.)
    
    Returns:
        User-friendly error message
    """
    error_type = type(error).__name__
    error_str = str(error)
    
    # Extract connection details for logging
    base_url = connection_info.get('base_url', 'unknown')
    model = connection_info.get('model', 'unknown')
    
    # Log the error with connection details
    logger.error(
        f'OpenAI {error_type} during {context}',
        error_type=error_type,
        error_message=error_str,
        base_url=base_url,
        model=model,
        context=context
    )
    
    # Provide more specific error messages based on error type
    if 'Connection' in error_type or 'Connect' in error_type or 'Network' in error_type:
        logger.error(
            f'Connection error details for {context}',
            base_url=base_url,
            model=model,
            error_details=error_str
        )
        return f'⚠️ Connection error: Unable to reach LLM server at {base_url}. Please check if the server is running.'
    
    elif 'Timeout' in error_type:
        logger.error(
            f'Timeout error details for {context}',
            base_url=base_url,
            model=model,
            error_details=error_str
        )
        return f'⚠️ Timeout error: LLM server at {base_url} did not respond in time. The server may be overloaded.'
    
    elif 'Authentication' in error_type or 'API key' in error_str.lower():
        logger.error(
            f'Authentication error details for {context}',
            base_url=base_url,
            model=model,
            error_details=error_str
        )
        return f'⚠️ Authentication error: Invalid API key or authentication failed for {base_url}.'
    
    elif 'Rate limit' in error_str.lower() or 'quota' in error_str.lower():
        logger.error(
            f'Rate limit error details for {context}',
            base_url=base_url,
            model=model,
            error_details=error_str
        )
        return f'⚠️ Rate limit error: Too many requests to LLM server at {base_url}. Please try again later.'
    
    else:
        # Generic error with connection info
        logger.error(
            f'Generic OpenAI error details for {context}',
            base_url=base_url,
            model=model,
            error_details=error_str
        )
        return f'⚠️ Error communicating with AI at {base_url}: {error_str}'


# Corresponding to the errors logged via send_long_message below, which will instead be raised up the stack
class MissingLLMResponseError(Exception):
    ''' Raised when the LLM response is missing a message object'''
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class LLMResponseError(Exception):
    ''' Raised when there is a general error accessing the LLM'''
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


# Catch-all tool-calling exception that holds the underlying exception as well as info on the specific tool call that failed
class ToolCallExecutionError(Exception):
    ''' Raised when there is an error executing a tool call'''
    def __init__(self, message: str, tool_name: str, tool_call_id: str, original_exception: Exception):
        super().__init__(message)
        self.message = message
        self.tool_name = tool_name
        self.tool_call_id = tool_call_id
        self.original_exception = original_exception

    def __str__(self):
        return f'ToolCallExecutionError: {self.message} (Tool: {self.tool_name}, ID: {self.tool_call_id})'


class OpenAILLMWrapper:
    '''
    A wrapper class for OpenAI LLM calls, handling streaming and non-streaming responses.
    This class is designed to work with the OpenAI API and provides methods for making
    chat completions, handling tool calls, and formatting results.
    '''
    def __init__(self, init_params: dict[str, Any], chat_params: dict[str, Any], tool_handler):
        '''
        Initialize the OpenAILLMWrapper with the provided LLM client.
        :param llm_client: The LLM client instance to use for API calls.
        '''
        self.llm_client = AsyncOpenAI(**init_params)
        self.llm_chat_params = chat_params
        self.tool_handler = tool_handler
        # Store connection info for error logging
        self.connection_info = {
            'base_url': init_params.get('base_url', 'unknown'),
            'model': chat_params.get('model', 'unknown')
        }

    async def __call__(self, messages: list[dict], user: str, extra_chat_params: dict,  stream: bool = False):
        '''
        Perform the LLM call using the provided parameters and channel history
        '''
        initial_response_content = ''
        tool_calls_aggregated: list[dict[str, Any]] = []
        chat_params = {**self.llm_chat_params, **extra_chat_params}  # Combine endpoint & explicit/extra settings

        logger.debug('LLM call', user=user, stream=stream, model=chat_params.get('model'))
        try:
            if stream:
                llm_stream = await self.llm_client.chat.completions.create(messages=messages, user=user, **chat_params)
                # llm_stream = await self.llm_client.chat.completions.create(messages=messages, **chat_params)
                current_tool_calls: list[dict[str, Any]] = []
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
                # response = await self.llm_client.chat.completions.create(messages=messages, **chat_params)
                response_message: ChatCompletionMessage | None = response.choices[0].message if response.choices else None
                if not response_message:
                    logger.error('LLM response missing message object', response_data=response.model_dump_json(indent=2))
                    raise MissingLLMResponseError('⚠️ Received an empty response from AI')
                initial_response_content = response_message.content or ''
                raw_tool_calls: list[ChatCompletionMessageToolCall] | None = response_message.tool_calls
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

        except OpenAIError as api_exc:
            error_message = _log_openai_error_details(api_exc, 'initial LLM call', self.connection_info)
            raise MissingLLMResponseError(error_message)
        return initial_response_content, tool_calls_aggregated

    async def process_tool_calls(self, tool_calls_aggregated, messages, thread_id, user, tool_call_result_hook, stream=False):
        tool_results_for_history = []

        for tool_call in tool_calls_aggregated:
            # XXX: Exception handling maybe a bit clunky?
            # Basic validation of structure
            try:
                if not isinstance(tool_call, dict) or 'function' not in tool_call or 'id' not in tool_call:
                    logger.error('Malformed tool call structure received from LLM', tool_call_data=tool_call)
                    continue
                if not isinstance(tool_call['function'], dict) or 'name' not in tool_call['function'] or 'arguments' not in tool_call['function']:
                    logger.error('Malformed tool call function structure received from LLM', tool_call_data=tool_call)
                    continue

                tool_name = tool_call['function']['name']
                tool_call_id = tool_call['id']
                arguments_str = tool_call['function']['arguments']
                tool_result_content_formatted = f'Error: Tool call processing failed internally before execution for {tool_name}'  # Default
                tool_call_exc = None
            except Exception as exc:
                # Catch errors from execute_tool or format_calltoolresult_content itself (should be rare now)
                logger.exception('Error during tool call prep', tool_name=tool_name, tool_call_id=tool_call_id, error=str(exc))
                tool_result_content_formatted = f'Error during tool prep phase for `{tool_name}`: {str(exc)}'
                # Track failure result without interrupting the process with a raise
                tool_call_exc = ToolCallExecutionError(f'⚠️ Error prepping tool `{tool_name}` for execution: {str(exc)}',
                        tool_name=tool_name, tool_call_id=tool_call_id, original_exception=exc)

            # Ensure arguments are valid JSON. Intentionally separated from any JSON errors in the tool itself
            try:
                tool_args = json.loads(arguments_str)
                if not isinstance(tool_args, dict):
                    raise TypeError('Arguments must be a JSON object (dict)')
            except (json.JSONDecodeError, TypeError) as json_err:
                logger.error(f'Failed to decode JSON args or not a dict for tool `{tool_name}`', args_str=arguments_str, tool_call_id=tool_call_id, error=str(json_err))
                tool_result_content_formatted = f'Error: Invalid JSON object arguments provided for tool `{tool_name}`. LLM sent: {arguments_str}'
                tool_call_exc = ToolCallExecutionError(f'⚠️ Error executing tool `{tool_name}`: {str(json_err)}',
                        tool_name=tool_name, tool_call_id=tool_call_id, original_exception=json_err)

            try:

                # XXX: Maybe take some sort of thread/conv key, which would be discord channel name, as used in this project?
                logger.info(f'Executing tool `{tool_name}`', args=tool_args, tool_call_id=tool_call_id)
                tool_result_obj = await self.tool_handler.execute_tool(tool_name, tool_args)

                tool_result_content_formatted = format_calltoolresult_content(tool_result_obj)

                tool_result_dict = {
                    'role': 'tool',
                    'tool_call_id': tool_call_id,
                    'name': tool_name,
                    'content': tool_result_content_formatted
                }
                tool_results_for_history.append(tool_result_dict)

            except Exception as exc:
                # Catch errors from execute_tool or format_calltoolresult_content itself (should be rare now)
                logger.exception('Error during tool call execution/formatting', tool_name=tool_name, tool_call_id=tool_call_id, error=str(exc))
                tool_result_content_formatted = f'Error during tool execution phase for `{tool_name}`: {str(exc)}'
                # Track failure result without interrupting the process with a raise
                tool_call_exc = ToolCallExecutionError(f'⚠️ Error executing tool `{tool_name}`: {str(exc)}',
                        tool_name=tool_name, tool_call_id=tool_call_id, original_exception=exc)

            finally:
                await tool_call_result_hook(tool_name, tool_call_id, tool_result_content_formatted, exc=tool_call_exc)
                # Append result to history (even if it was an error during execution phase)
                tool_results_for_history.append({
                    'role': 'tool',
                    'tool_call_id': tool_call_id,
                    'name': tool_name,
                    'content': tool_result_content_formatted
                })

        # Follow-up LLM Call
        if tool_results_for_history:
            messages.extend(tool_results_for_history)
            logger.debug(f'Added {len(tool_results_for_history)} tool results to history', thread_id=thread_id)

            logger.debug('Initiating follow-up LLM call after tools.', thread_id=thread_id)
            follow_up_params = {**self.llm_chat_params, 'stream': stream}
            follow_up_params.pop('tools', None)  # No tools needed for follow-up
            follow_up_params.pop('tool_choice', None)

            follow_up_text = ''
            try:
                if stream:
                    follow_up_stream = await self.llm_client.chat.completions.create(messages=messages, user=user, **follow_up_params)
                    # follow_up_stream = await self.llm_client.chat.completions.create(messages=messages, **follow_up_params)
                    async for chunk in follow_up_stream:
                        if token := chunk.choices[0].delta.content or '': follow_up_text += token
                else:
                    follow_up_response = await self.llm_client.chat.completions.create(messages=messages, user=user, **follow_up_params)
                    # follow_up_response = await self.llm_client.chat.completions.create(messages=messages, **follow_up_params)
                    follow_up_message = follow_up_response.choices[0].message if follow_up_response.choices else None
                    if follow_up_message: follow_up_text = follow_up_message.content or ''

            except OpenAIError as exc:  # Catch specific OpenAI errors
                error_message = _log_openai_error_details(exc, 'follow-up LLM call', self.connection_info)
                raise MissingLLMResponseError(error_message)

        return follow_up_text
