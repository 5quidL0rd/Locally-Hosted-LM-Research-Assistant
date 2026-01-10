
"""
agents package - Multi-agent research system components
"""

# This file can be empty, it just makes the agents folder a Python package
# But you can add imports here for convenience if you want:

from .llm import LocalLLM
from .memory import MemoryPalace
from .arxiv_agent import ArxivAgent
from .kaggle_agent import KaggleAgent
from .search_agent import SearchAgent
from .orchestrator import ConversationalOrchestrator
from .nn_builder_agent import NeuralNetworkBuilder 

__all__ = [
    'LocalLLM',
    'MemoryPalace', 
    'ArxivAgent',
    'KaggleAgent',
    'SearchAgent',
    'HuggingFaceAgent'
    'ConversationalOrchestrator'
    'NeuralNetworkBuilder'
]