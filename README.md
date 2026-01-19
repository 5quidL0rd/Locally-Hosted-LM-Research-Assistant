# Multi-Agent Research Assistant

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A locally-hosted multi-agent AI research assistant that performs literature reviews, manages datasets, trains neural networks, fine-tunes language models, and generates citations—all through natural conversation.

**The idea is simple:** instead of juggling between arXiv, Google Scholar, Kaggle, HuggingFace, and various Python scripts, you just talk to it. *"Find me papers about transformers"*, *"download that dataset"*, *"train an LSTM on it"*, *"fine-tune BERT on this data"*. It figures out what you want and does it.

## Features

### Paper Search & Analysis
- Search arXiv using natural language queries
- Download and analyze papers with automatic PDF parsing
- **Innovation scoring system** (1-10) rating papers on novelty, technical depth, and impact
- Automated literature reviews with theme synthesis and gap identification

```
"Find papers about attention mechanisms"
"Download the first paper and analyze it"
"Score paper 2301.00001 on innovation"
"Generate a literature review on neural machine translation"
```

### Dataset Management
- Search Kaggle with improved query handling and synonym expansion
- Browse popular datasets or get task-specific recommendations
- **Automatic dataset profiling** with statistics, visualizations, and quality scores

```
"Search Kaggle for sentiment analysis datasets"
"Show popular Kaggle datasets"
"Recommend datasets for classification"
"Profile this dataset"
```

### Neural Network Training
- Train **MLP, LSTM, CNN, or Transformer** models on tabular data without writing code
- Automatic data preprocessing, architecture selection, and evaluation
- **Hyperparameter tuning** with Optuna
- Model saving, loading, and comparison

```
"Train LSTM and MLP on this dataset"
"Tune hyperparameters for MLP with 50 trials"
"List my saved models"
"Make predictions with the best model"
```

### HuggingFace Integration
- Search and download models from HuggingFace Hub
- **Fine-tune text classifiers** (sentiment, topic classification)
- Fine-tune directly on CSV files or Kaggle datasets
- Run inference with your fine-tuned models

```
"Search HuggingFace for sentiment models"
"Finetune bert on this dataset"
"Finetune distilbert with kaggle kazanova/sentiment140"
"Run inference with finetuned model: This movie was great!"
```

### Writing Assistance
- Get citation suggestions for claims in your writing
- Export research notes to Markdown
- Generate BibTeX files for LaTeX
- Help writing methodology sections

```
"Suggest citations for: attention mechanisms improve translation accuracy"
"Export my research notes to markdown"
"Export citations as BibTeX"
```

### Knowledge Graph & Memory
- Persistent knowledge graph tracking all your research
- Visualize connections between papers, datasets, and experiments
- Query your research history

```
"What have I researched so far?"
"Show me my knowledge graph"
```

## Installation

### Prerequisites
- Python 3.8+
- [LM Studio](https://lmstudio.ai) for local LLM inference
- CUDA-compatible GPU recommended for training

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/multi-agent-research-assistant.git
cd multi-agent-research-assistant
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure credentials**
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. **Set up LM Studio**
   - Download and install [LM Studio](https://lmstudio.ai)
   - Download a model (recommended: GPT-OSS-20B or similar 7B+ model)
   - Start the local server (default: `http://localhost:1234`)

5. **Run the assistant**
```bash
python main.py
```

### API Credentials

Create a `.env` file with your credentials:

```env
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
HUGGINGFACE_TOKEN=your_huggingface_token
LM_STUDIO_URL=http://localhost:1234/v1
```

- **Kaggle**: Get credentials from [kaggle.com/settings](https://www.kaggle.com/settings) → API → Create New Token
- **HuggingFace**: Get token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

## Architecture

The system uses a multi-agent architecture with specialized agents coordinated by an LLM-powered orchestrator:

```
┌─────────────────────────────────────────────────────────────┐
│                    Orchestrator (LLM Router)                │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  ArXiv Agent  │   │ Kaggle Agent  │   │HuggingFace    │
│  - Search     │   │  - Search     │   │Agent          │
│  - Download   │   │  - Download   │   │  - Search     │
│  - Analyze    │   │  - Profile    │   │  - Fine-tune  │
│  - Score      │   │               │   │  - Inference  │
└───────────────┘   └───────────────┘   └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                              ▼
                    ┌───────────────┐
                    │ Memory Palace │
                    │ (Knowledge    │
                    │  Graph)       │
                    └───────────────┘
```

### Agent Descriptions

| Agent | Purpose |
|-------|---------|
| **ArXiv Agent** | Paper search, download, analysis, and innovation scoring |
| **Kaggle Agent** | Dataset search, download, profiling, and recommendations |
| **HuggingFace Agent** | Model search, download, fine-tuning, and inference |
| **Neural Network Builder** | Training MLP, LSTM, CNN, Transformer with hyperparameter tuning |
| **Literature Review Agent** | Automated literature reviews with synthesis |
| **Writing Assistant** | Citation suggestions, note export, methodology help |
| **Search Agent** | Web search for current information |
| **Memory Palace** | Persistent knowledge graph across sessions |

## Project Structure

```
├── main.py                     # Entry point and configuration
├── .env.example                # Template for credentials
├── requirements.txt            # Python dependencies
├── agents/
│   ├── orchestrator.py         # Routes requests to appropriate agents
│   ├── arxiv_agent.py          # Paper search and analysis
│   ├── kaggle_agent.py         # Dataset management
│   ├── huggingface_agent.py    # Model fine-tuning and inference
│   ├── nn_builder_agent.py     # Neural network training
│   ├── literature_review.py    # Literature review generation
│   ├── writing_assistant.py    # Citation and writing help
│   ├── data_profiler.py        # Dataset profiling
│   ├── search_agent.py         # Web search
│   ├── memory.py               # Knowledge graph persistence
│   └── llm.py                  # Local LLM interface
├── arxiv_papers/               # Downloaded PDFs
├── kaggle_datasets/            # Downloaded datasets
├── hf_cache/                   # Cached HuggingFace models
├── finetuned_models/           # Your fine-tuned models
├── saved_models/               # Trained neural networks
├── literature_reviews/         # Generated reviews
├── data_profiles/              # Dataset analysis reports
└── writing_output/             # Exported notes and citations
```

## Usage Examples

### Research Workflow
```
You: Find papers about vision transformers in medical imaging
Assistant: Found 5 papers...

You: Download the first two and analyze them
Assistant: [Downloads PDFs, extracts content, provides analysis with innovation scores]

You: Generate a literature review on this topic
Assistant: [Creates comprehensive review with themes, gaps, and citations]

You: Export citations as BibTeX
Assistant: [Exports formatted citations to .bib file]
```

### ML Workflow
```
You: Search Kaggle for heart disease datasets
Assistant: Found 8 datasets...

You: Download the first one
Assistant: Downloaded to kaggle_datasets/...

You: Profile this dataset
Assistant: [Generates statistics, visualizations, quality score]

You: Train MLP and LSTM to predict the target
Assistant: [Trains models, shows metrics, saves best model]

You: Tune hyperparameters for the best model
Assistant: [Runs Optuna optimization, reports best parameters]
```

## Limitations

- Local LLM quality depends on the model you're running
- Large dataset training can be slow without a GPU
- HuggingFace fine-tuning requires sufficient VRAM
- Some Kaggle datasets require accepting competition rules first

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [arXiv](https://arxiv.org) for open access to academic papers
- [Kaggle](https://kaggle.com) for dataset hosting
- [HuggingFace](https://huggingface.co) for the transformers library and model hub
- [LM Studio](https://lmstudio.ai) for local LLM inference
- The open source ML community for PyTorch, scikit-learn, Optuna, and countless other tools
