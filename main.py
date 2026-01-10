"""
main.py - Conversational Multi-Agent Research System

A natural language interface to research papers, datasets, and web search.
Talk to it like you'd talk to ChatGPT/Claude.

Requirements:
    pip install requests pymupdf ddgs sentence-transformers faiss-cpu
    pip install arxiv kaggle networkx pyvis tiktoken

Usage:
    python main.py
    
Then just chat naturally:
    "Find me papers about neural networks"
    "Download that second paper"
    "Search Kaggle for stock market data"
    "What's in my memory palace?"
"""

import os
import sys



# =======================================================
# API CREDENTIALS - SET YOUR VALUES HERE
# =======================================================
KAGGLE_USERNAME = ""  # Your Kaggle username
KAGGLE_KEY = ""      # Your Kaggle API key
HUGGINGFACE_TOKEN = ""  # Get from https://huggingface.co/settings/tokens

if KAGGLE_USERNAME != "your_username_here" and KAGGLE_KEY != "your_key_here":
    os.environ['KAGGLE_USERNAME'] = KAGGLE_USERNAME
    os.environ['KAGGLE_KEY'] = KAGGLE_KEY
    print(f"[Config] Kaggle credentials set for user: {KAGGLE_USERNAME}")
else:
    print("[Config] WARNING: Kaggle credentials not set!")

# Ensure the HuggingFace token is set correctly
if HUGGINGFACE_TOKEN and HUGGINGFACE_TOKEN != "your_hf_token_here":
    os.environ['HUGGINGFACE_TOKEN'] = HUGGINGFACE_TOKEN
    print(f"[Config] HuggingFace token configured")
else:
    print("[Config] WARNING: HuggingFace token not set or invalid!")
# =======================================================

from agents.llm import LocalLLM
from agents.memory import MemoryPalace
from agents.arxiv_agent import ArxivAgent
from agents.kaggle_agent import KaggleAgent
from agents.search_agent import SearchAgent
from agents.huggingface_agent import HuggingFaceAgent
from agents.nn_builder_agent import NeuralNetworkBuilder
from agents.orchestrator import ConversationalOrchestrator



class ResearchAssistant:
    def __init__(self):
        print("\n[System] Initializing research assistant...")
        
        # Core components
        self.llm = LocalLLM()
        self.memory = MemoryPalace()
        
        # Specialized agents
        self.arxiv = ArxivAgent(self.llm, self.memory)
        self.kaggle = KaggleAgent(self.memory)
        self.search = SearchAgent()
        self.huggingface = HuggingFaceAgent(self.memory, api_token=HUGGINGFACE_TOKEN)
        
        # ADD THESE TWO LINES:
        self.nn_builder = NeuralNetworkBuilder(self.memory)
      
        
        # Conversational orchestrator
        self.orchestrator = ConversationalOrchestrator(
            llm=self.llm,
            arxiv=self.arxiv,
            kaggle=self.kaggle,
            search=self.search,
            huggingface=self.huggingface,
            memory=self.memory,
            nn_builder = self.nn_builder
        )
    
    def run(self):
        """Main conversational loop"""
        print("=" * 70)
        print("CONVERSATIONAL RESEARCH ASSISTANT")
        print("=" * 70)
        print("\nJust talk naturally! Examples:")
        print('  "Find papers about machine learning"')
        print('  "Download the first paper and analyze it"')
        print('  "Search Kaggle for bitcoin datasets"')
        print('  "Download that dataset"')
        print('  "Search HuggingFace for llama models"')
        print('  "Download gpt2 and run it"')
        print('  "Generate text with gpt2: Once upon a time"')
        print('  "Find datasets on HuggingFace about sentiment"')
        print('  "Search arXiv for transformers, download top 2, analyze them"')
        print('  "Analyze the last Kaggle dataset and add plots to my report"')
        print('  "Add the top arXiv result to my Word report"')
        print('  "What have I researched so far?"')
        print('  "Show me my knowledge graph"')
        print('  "Search the web for latest AI news"')
        print("\nType 'exit' or 'quit' to leave.\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nInput error: {e}")
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                print("\nGoodbye!")
                break
            
            # Let the orchestrator handle the conversation
            response = self.orchestrator.process(user_input)
            print(f"\nAssistant: {response}\n")


if __name__ == "__main__":
    assistant = ResearchAssistant()
    assistant.run()