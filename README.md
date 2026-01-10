# 🤖 Conversational Research Assistant 🤖
A locally-hosted AI research assistant powered by OpenAI's GPT-OSS-20B that provides intelligent conversation, academic research capabilities, dataset analysis, and neural network training—all through a natural language interface. This uses a multi-agent setup, is all local, and will be seeing improvements in the very near future.
Features
🔬 Academic Research

arXiv Integration: Search, download, and analyze academic papers from arXiv
Natural language queries to find relevant research papers
Automatic paper analysis and summarization

📊 Dataset Management

Kaggle Integration: Search and download datasets directly from Kaggle
Browse available datasets through conversation
Quick download by specifying dataset name after search - after searching datasets, put the name of the dataset as your first input in the chat

🤖 Neural Network Training

Built-in ML Capabilities: Train models on downloaded datasets without writing code
LSTM Networks: For sequential and time-series data analysis
MLP Networks: For general classification and regression tasks
Automatic results visualization and performance metrics

🤗 Hugging Face Integration

Search the Hugging Face model hub for pre-trained models
Download models directly for local experimentation
Easy model discovery through natural language queries

💬 General Conversation

Natural language interface for all operations
Context-aware responses to general questions
Multi-agent architecture for handling complex queries

Architecture
This project uses a multi-agent setup where specialized agents handle different types of requests:

Research Agent: Handles arXiv paper searches and analysis
Dataset Agent: Manages Kaggle dataset operations
Training Agent: Executes neural network training and evaluation
Model Agent: Interfaces with Hugging Face model hub
Conversation Agent: Manages general queries and coordinates between agents

All processing happens locally on your machine—no data is sent to external APIs beyond the necessary searches.
Quick Start
Prerequisites

Python 3.8+
CUDA-compatible GPU (recommended for model inference and training)
Kaggle API credentials (for dataset downloads)

Installation
```bash Clone the repository
git clone https://github.com/yourusername/research-assistant.git
cd research-assistant

# Install dependencies
pip install -r requirements.txt

# Set up Kaggle credentials
# Place your kaggle.json in ~/.kaggle/
```

Usage
```bash
python main.py
Once running, you can interact naturally:
You: Search arXiv for transformers, download top 2, analyze them


You: Find papers about machine learning

You: Search Kaggle for apple datasets

You: download 1
# (After searching datasets, specify which one to download)

You: Train MLP on this dataset

You: Search HuggingFace for llama models

You: Download that dataset
Example Conversations
Academic Research:
You: Find papers about machine learning
Assistant: [Searches arXiv and presents relevant papers]

You: Download the first paper and analyze it
Assistant: [Downloads and provides analysis]
Dataset Analysis:
You: Search Kaggle for bitcoin dataset
Assistant: [Shows available datasets]

You: Download 1
Assistant: [Downloads specified dataset]

You: Train LSTM on this data
Assistant: [Trains model and shows results]
Model Discovery:
You: Search HuggingFace for sentiment analysis models
Assistant: [Lists relevant models with descriptions]

You: Download the top one
Assistant: [Downloads model to local storage]
``` 

Roadmap
This project is under active development. Upcoming improvements include:

Enhanced multi-modal capabilities (image and document analysis)
Additional neural network architectures (CNNs, Transformers)
Improved conversation memory and context handling
Web interface for easier interaction
Batch processing for multiple papers/datasets
Custom training configurations and hyperparameter tuning
Integration with additional data sources
Model fine-tuning capabilities

Technical Details

LLM: OpenAI GPT-OSS-20B (locally hosted)
ML Frameworks: PyTorch, TensorFlow
APIs: arXiv API, Kaggle API, Hugging Face Hub API
Architecture: Multi-agent system with specialized handlers


Acknowledgments

OpenAI for GPT-OSS-20B
arXiv for academic paper access
Kaggle for dataset hosting
Hugging Face for model hosting


Note: This assistant runs entirely on your local machine, giving you full control over your data and research workflow.