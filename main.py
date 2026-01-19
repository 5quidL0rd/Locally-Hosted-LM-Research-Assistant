"""
main.py - Conversational Multi-Agent Research System

A natural language interface to research papers, datasets, and web search.
Talk to it like you'd talk to ChatGPT/Claude.

Requirements:
    pip install -r requirements.txt

Usage:
    1. Copy .env.example to .env and fill in your credentials
    2. python main.py

Then just chat naturally:
    "Find me papers about neural networks"
    "Download that second paper"
    "Search Kaggle for stock market data"
    "What's in my memory palace?"
"""

import os
import sys
from pathlib import Path

# =======================================================
# LOAD ENVIRONMENT VARIABLES
# =======================================================
def load_env():
    """Load environment variables from .env file if it exists."""
    env_path = Path(__file__).parent / ".env"

    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Clean up the key and value
                    key = key.strip()
                    value = value.strip()
                    # Remove surrounding quotes if present
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    os.environ.setdefault(key, value)
        print("[Config] Loaded environment from .env file")
    else:
        print("[Config] No .env file found. Using system environment variables.")
        print("[Config] Copy .env.example to .env and add your credentials.")

# Load .env before anything else
load_env()

# =======================================================
# GET CREDENTIALS FROM ENVIRONMENT
# =======================================================
KAGGLE_USERNAME = os.environ.get('KAGGLE_USERNAME', '')
KAGGLE_KEY = os.environ.get('KAGGLE_KEY', '')
HUGGINGFACE_TOKEN = os.environ.get('HUGGINGFACE_TOKEN', '')
LM_STUDIO_URL = os.environ.get('LM_STUDIO_URL', 'http://localhost:1234/v1')

# Validate and set Kaggle credentials
if KAGGLE_USERNAME and KAGGLE_KEY:
    os.environ['KAGGLE_USERNAME'] = KAGGLE_USERNAME
    os.environ['KAGGLE_KEY'] = KAGGLE_KEY
    print(f"[Config] Kaggle credentials configured for user: {KAGGLE_USERNAME}")
else:
    print("[Config] WARNING: Kaggle credentials not set!")
    print("[Config] Set KAGGLE_USERNAME and KAGGLE_KEY in .env file")

# Validate HuggingFace token
if HUGGINGFACE_TOKEN:
    os.environ['HUGGINGFACE_TOKEN'] = HUGGINGFACE_TOKEN
    print("[Config] HuggingFace token configured")
else:
    print("[Config] WARNING: HuggingFace token not set!")
    print("[Config] Set HUGGINGFACE_TOKEN in .env file for model downloads")


from agents.llm import LocalLLM
from agents.memory import MemoryPalace
from agents.arxiv_agent import ArxivAgent
from agents.kaggle_agent import KaggleAgent
from agents.search_agent import SearchAgent
from agents.huggingface_agent import HuggingFaceAgent
from agents.nn_builder_agent import NeuralNetworkBuilder
from agents.orchestrator import ConversationalOrchestrator
from agents.data_profiler import DataProfiler
from agents.literature_review import LiteratureReviewAgent
from agents.writing_assistant import WritingAssistant



class ResearchAssistant:
    def __init__(self):
        print("\n[System] Initializing research assistant...")

        self.llm = LocalLLM(endpoint=LM_STUDIO_URL)
        self.memory = MemoryPalace()

        # Core agents
        self.arxiv = ArxivAgent(self.llm, self.memory)
        self.kaggle = KaggleAgent(self.memory)
        self.search = SearchAgent()
        self.huggingface = HuggingFaceAgent(self.memory, api_token=HUGGINGFACE_TOKEN)
        self.nn_builder = NeuralNetworkBuilder(self.memory)

        # New research-focused agents
        self.data_profiler = DataProfiler()
        self.literature_review = LiteratureReviewAgent(self.llm, self.memory)
        self.writing_assistant = WritingAssistant(self.llm, self.memory)

        # Conversational orchestrator with all agents
        self.orchestrator = ConversationalOrchestrator(
            llm=self.llm,
            arxiv=self.arxiv,
            kaggle=self.kaggle,
            search=self.search,
            huggingface=self.huggingface,
            memory=self.memory,
            nn_builder=self.nn_builder,
            data_profiler=self.data_profiler,
            literature_review=self.literature_review,
            writing_assistant=self.writing_assistant
        )

    def run(self):
        """Main conversational loop"""
        print("=" * 70)
        print("CONVERSATIONAL RESEARCH ASSISTANT")
        print("=" * 70)
        print("\nJust talk naturally! Examples:")
        print("")
        print("  PAPER SEARCH & ANALYSIS:")
        print('    "Find papers about transformer architectures"')
        print('    "Download the first paper and analyze it"')
        print('    "Score paper 2301.00001 on innovation"')
        print('    "Generate literature review on neural machine translation"')
        print("")
        print("  KAGGLE DATASETS (Improved Search):")
        print('    "Search Kaggle for sentiment analysis datasets"')
        print('    "Show popular Kaggle datasets"')
        print('    "Recommend datasets for classification"')
        print('    "Profile this dataset" (after downloading)')
        print("")
        print("  NEURAL NETWORK TRAINING:")
        print('    "Train LSTM and MLP on this dataset"')
        print('    "Tune hyperparameters for MLP with 50 trials"')
        print('    "List my saved models"')
        print("")
        print("  HUGGINGFACE MODEL FINE-TUNING:")
        print('    "Search HuggingFace for sentiment models"')
        print('    "Finetune bert on this dataset"')
        print('    "Finetune distilbert with kaggle kazanova/sentiment140"')
        print('    "Run inference with finetuned model: This movie was great!"')
        print("")
        print("  WRITING ASSISTANCE:")
        print('    "Suggest citations for: attention mechanisms improve accuracy"')
        print('    "Export my research notes to markdown"')
        print('    "Export citations as BibTeX"')
        print('    "Write methodology: We trained a CNN on CIFAR-10..."')
        print("")
        print("  MEMORY & SEARCH:")
        print('    "What have I researched so far?"')
        print('    "Show me my knowledge graph"')
        print('    "Search the web for latest AI news"')
        print("")
        print("Type 'exit' or 'quit' to leave.\n")

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
