# SPDX-FileCopyrightText: 2025-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# discord_aiagent.llm_providers

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import structlog

logger = structlog.get_logger(__name__)


class LLMProvider(ABC):
    """Base interface for LLM providers."""
    
    def __init__(self, init_params: Dict[str, Any], chat_params: Dict[str, Any], tool_handler):
        """
        Initialize the LLM provider.
        
        Args:
            init_params: Provider-specific initialization parameters
            chat_params: Chat completion parameters (model, temperature, etc.)
            tool_handler: Reference to the tool handler for executing tools
        """
        self.init_params = init_params
        self.chat_params = chat_params
        self.tool_handler = tool_handler
        self.connection_info = {
            'base_url': init_params.get('base_url', 'unknown'),
            'model': chat_params.get('model', 'unknown')
        }
    
    @abstractmethod
    async def __call__(self, messages: List[Dict], user: str, extra_chat_params: Dict, stream: bool = False):
        """
        Perform an LLM call.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            user: User identifier
            extra_chat_params: Additional parameters for this specific call
            stream: Whether to stream the response
            
        Returns:
            Tuple of (initial_response_content, tool_calls_aggregated)
        """
        pass
    
    @abstractmethod
    async def process_tool_calls(self, tool_calls_aggregated: List[Dict], messages: List[Dict], 
                               thread_id: str, user: str, tool_call_result_hook, stream: bool = False):
        """
        Process tool calls and return follow-up response.
        
        Args:
            tool_calls_aggregated: List of tool calls to execute
            messages: Message history
            thread_id: Thread identifier
            user: User identifier
            tool_call_result_hook: Callback for tool execution results
            stream: Whether to stream the response
            
        Returns:
            Follow-up response text
        """
        pass


class LLMProviderFactory:
    """Factory for creating LLM provider instances."""
    
    _providers: Dict[str, type] = {}
    
    @classmethod
    def register_provider(cls, api_type: str, provider_class: type):
        """Register a provider class for a specific API type."""
        cls._providers[api_type] = provider_class
        logger.info(f"Registered LLM provider: {api_type} -> {provider_class.__name__}")
    
    @classmethod
    def create_provider(cls, api_type: str, init_params: Dict[str, Any], 
                       chat_params: Dict[str, Any], tool_handler) -> LLMProvider:
        """
        Create an LLM provider instance.
        
        Args:
            api_type: Type of API ('openai', 'claude', 'generic', etc.)
            init_params: Provider-specific initialization parameters
            chat_params: Chat completion parameters
            tool_handler: Reference to the tool handler
            
        Returns:
            LLMProvider instance
            
        Raises:
            ValueError: If the API type is not supported
        """
        if api_type not in cls._providers:
            available = list(cls._providers.keys())
            raise ValueError(f"Unsupported API type '{api_type}'. Available types: {available}")
        
        provider_class = cls._providers[api_type]
        return provider_class(init_params, chat_params, tool_handler)
    
    @classmethod
    def get_supported_providers(cls) -> List[str]:
        """Get list of supported provider types."""
        return list(cls._providers.keys())


# Import and register default providers
try:
    from .openai_provider import OpenAIProvider
    LLMProviderFactory.register_provider('openai', OpenAIProvider)
    LLMProviderFactory.register_provider('generic', OpenAIProvider)  # Generic is same as OpenAI
except ImportError as e:
    logger.warning(f"OpenAI provider not available: {e}")

try:
    from .claude_provider import ClaudeProvider
    LLMProviderFactory.register_provider('claude', ClaudeProvider)
except ImportError as e:
    logger.warning(f"Claude provider not available: {e}") 