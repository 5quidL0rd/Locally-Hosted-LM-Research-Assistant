<<<<<<< HEAD
# Conversational AI Research Assistant 
A locally-hosted AI research assistant powered by OpenAI's GPT-OSS-20B that provides intelligent conversation, academic research capabilities, dataset analysis, and neural network training—all through a natural language interface. This uses a multi-agent setup, is all local, and will be seeing improvements in the very near future.
Features
🔬 Academic Research
=======
# Locally Hosted AI-Powered Multi-Agent Research Assistant 
>>>>>>> 426ea0b (Literature review, Kaggle dataset profiler, fine-tuning capabilites, writing assistant.)

A locally-hosted multi-agent research assistant that performs a thorough, opinionated literature review with citation compilation and detects gaps in research, manage datasets and vizualizes them automatically, train neural networks, fine-tune language models, and generate literature reviews. All through natural conversation.

The idea is simple: instead of juggling between arXiv, Google Scholar, Kaggle, HuggingFace, and various Python scripts, you just talk to it. "Find me papers about transformers", "download that dataset", "train an LSTM on it", "fine-tune BERT on this data". It figures out what you want and does it. It is a fully-integrated pipeline where researchers can streamline their work with a helpful assistant, all with total control over data. 

## Capabilities

<<<<<<< HEAD
![Assistant Interface](./Screenshot%202026-01-06%20155055.png)

🤖 Neural Network Training

Built-in ML Capabilities: Train models on downloaded datasets without writing code

LSTM Networks: For sequential and time-series data analysis

MLP Networks: For general classification and regression tasks

Automatic results visualization and performance metrics

![Assistant Results](./Screenshot%202026-01-06%20155155.png)

🤗 Hugging Face Integration

Search the Hugging Face model hub for pre-trained models

Download models directly for local experimentation

Easy model discovery through natural language queries

Example:

![Assistant Interface](./Screenshot%202026-01-04%20142844.png)



💬 General Conversation

Natural language interface for all operations

Context-aware responses to general questions

Multi-agent architecture for handling complex queries
=======
### Paper Search and Analysis

The assistant connects to arXiv and lets you search for papers using plain English. When you find something interesting, you can download the PDF and get an analysis that covers the main contributions, methodology, and key findings.

There's also an innovation scoring system. Each paper gets rated 1-10 on innovation, uniqueness, technical depth, and potential impact. This helps when you're doing a literature review and need to prioritize which papers to read in depth.

```
"Find papers about attention mechanisms"
"Download the first paper and analyze it"
"Score paper 2301.00001 on innovation"
```

### Literature Reviews

Point it at a topic and it will search for relevant papers, analyze each one, score them on innovation, and compile everything into a structured literature review. The output includes summaries, methodology comparisons, and a synthesis of the main themes across all papers that also identifies gaps in the research that could be addressed. These findings are exported to neat .md files where they can be analyzed at your leisure. 
>>>>>>> 426ea0b (Literature review, Kaggle dataset profiler, fine-tuning capabilites, writing assistant.)

```
"Generate a literature review on neural machine translation"
"Create a survey of recent transformer architectures"
```

<<<<<<< HEAD
Research Agent: Handles arXiv paper searches and analysis

Dataset Agent: Manages Kaggle dataset operations

Training Agent: Executes neural network training and evaluation

Model Agent: Interfaces with Hugging Face model hub

Conversation Agent: Manages general queries and coordinates between agents
=======
### Kaggle Datasets
>>>>>>> 426ea0b (Literature review, Kaggle dataset profiler, fine-tuning capabilites, writing assistant.)

Search Kaggle for datasets, browse popular ones, or get recommendations based on your task type. The search has been improved with better query handling, synonym expansion, and fallback options when the initial search comes up empty. (Because the Kaggle API key can be a royal pain in the neck)

Once downloaded, you can profile a dataset to get statistics, data types, missing values, and distribution plots. If you want to profile a dataset a full report is generated, complete with png files visualizing the data and a score on how good the dataset is with accompanying rationale. 

```
"Search Kaggle for sentiment analysis datasets"
"Show popular Kaggle datasets"
"Recommend datasets for classification"
"Profile this dataset"
```

### Neural Network Training

Train MLP, LSTM, CNN, or Transformer models on tabular CSV data without writing any code. The system handles data preprocessing, model architecture selection, training loops, and evaluation metrics.

There's also hyperparameter tuning using Optuna. Specify how many trials you want and it will search for the best configuration.

```
"Train LSTM and MLP on this dataset"
"Tune hyperparameters for MLP with 50 trials"
"List my saved models"
"Make predictions with the best model"
```

### HuggingFace Model Download and Fine-Tuning

Search the HuggingFace model hub, download models, and fine-tune them on your datasets. The system supports text classification (sentiment, topic classification), causal language modeling, and sequence-to-sequence tasks.

You can fine-tune directly on a CSV file or combine it with a Kaggle dataset download in one command. The system auto-detects the text and label columns so you don't need to specify them manually.

```
"Search HuggingFace for sentiment models"
"Download distilbert-base-uncased"
"Finetune bert on this dataset"
"Finetune distilbert with kaggle kazanova/sentiment140"
"Run inference with finetuned model: This movie was great!"
```

### Writing Assistance

Get citation suggestions for claims in your writing. The system searches for relevant papers and formats them as proper citations. You can also export your research notes to markdown or generate BibTeX files for LaTeX.

```
"Suggest citations for: attention mechanisms improve translation accuracy"
"Export my research notes to markdown"
"Export citations as BibTeX"
"Help me write: We trained a CNN on CIFAR-10..."
```

### Memory and Knowledge Graph

Everything you research gets stored in a persistent knowledge graph. Papers, datasets, experiments, and their relationships are all tracked. You can query what you've researched, visualize the connections, and export your notes.

```
"What have I researched so far?"
"Show me my knowledge graph"
"Search the web for latest AI news"
```

## Installation

You'll need Python 3.8+ and ideally a CUDA-compatible GPU for training and inference.

```bash
git clone https://github.com/yourusername/research-assistant.git
cd research-assistant

pip install -r requirements.txt
```

<<<<<<< HEAD


Once running, you can interact naturally:
```
You: Search arXiv for transformers, download top 2, analyze them

You: Find papers about machine learning

You: Search Kaggle for apple datasets

You: download 1
# (After searching datasets, specify which one to download)

You: Train MLP on this dataset

You: Search HuggingFace for llama models

You: Download that dataset
```



# Roadmap

This project is under active development. Upcoming improvements include:

1) Enhanced multi-modal capabilities (image and document analysis)
2) Additional neural network architectures (CNNs, Transformers)
3) Improved conversation memory and context handling
4) Web interface for easier interaction
5) Batch processing for multiple papers/datasets
6) Custom training configurations and hyperparameter tuning
7) Integration with additional data sources
8) Model fine-tuning capabilities

Technical Details

LLM: OpenAI GPT-OSS-20B (locally hosted, courtesy of LM Studio) 

ML Frameworks: PyTorch, TensorFlow

APIs: arXiv API, Kaggle API, Hugging Face Hub API

Architecture: Multi-agent system with specialized handlers


# Acknowledgments 

OpenAI for GPT-OSS-20B

arXiv for academic paper access

Kaggle for dataset hosting

Hugging Face for model hosting



## Note: This assistant runs entirely on your local machine, giving you full control over your data and research workflow.
=======
### Setting Up LM Studio

This project uses LM Studio as the local LLM backend. The multi-agent architecture communicates with models running in LM Studio through its OpenAI-compatible API.

1. Download and install LM Studio from [lmstudio.ai](https://lmstudio.ai)
2. Download a model. I recommend OpenAI's GPT-OSS-20B (available in LM Studio's model browser). This 20B parameter model hits the sweet spot for multi-agent work: it's large enough to handle complex reasoning and tool routing, but small enough to run on consumer hardware with decent speed.
3. Load the model and start the local server (usually runs on `http://localhost:1234`)
4. The assistant will connect to this endpoint automatically

One nice thing about using LM Studio: you can open the Developer tab and watch the model's reasoning in real-time. When you send a request, you'll see exactly how the model interprets your prompt, what tools it decides to use, and its internal thought process. This is genuinely useful for debugging when the orchestrator routes something to the wrong agent, or when you want to understand why it made a particular decision.

### API Credentials

Edit `main.py` and set your credentials:

```python
KAGGLE_USERNAME = "your_username"
KAGGLE_KEY = "your_api_key"
HUGGINGFACE_TOKEN = "your_hf_token"
```

For Kaggle, you can also place your `kaggle.json` in `~/.kaggle/`.

### Dependencies

Core dependencies (install via pip):

```
requests pymupdf ddgs sentence-transformers faiss-cpu
arxiv kaggle networkx pyvis tiktoken
torch transformers datasets sklearn
optuna matplotlib pandas numpy
```

## Usage

```bash
python main.py
```

Then just type naturally. The system routes your request to the appropriate agent based on what you're asking for.

## Architecture

The system uses a multi-agent setup:

- **ArXiv Agent**: Paper search, download, analysis, and innovation scoring
- **Kaggle Agent**: Dataset search, download, profiling, and recommendations
- **HuggingFace Agent**: Model search, download, fine-tuning, and inference
- **Neural Network Builder**: Training MLP, LSTM, CNN, Transformer models with hyperparameter tuning
- **Literature Review Agent**: Automated literature reviews with paper analysis and synthesis
- **Writing Assistant**: Citation suggestions, note export, and writing help
- **Search Agent**: Web search for current information
- **Memory Palace**: Persistent knowledge graph that remembers your research across sessions
- **Orchestrator**: Routes requests to the right agent based on natural language understanding

All processing happens locally. The only external calls are to arXiv, Kaggle, and HuggingFace APIs for searching and downloading resources.

## Project Structure

```
├── main.py                     # Entry point and configuration
├── agents/
│   ├── orchestrator.py         # Routes requests to appropriate agents
│   ├── arxiv_agent.py          # Paper search and analysis
│   ├── kaggle_agent.py         # Dataset management
│   ├── huggingface_agent.py    # Model fine-tuning and inference
│   ├── nn_builder_agent.py     # Neural network training
│   ├── literature_review.py    # Literature review generation
│   ├── writing_assistant.py    # Citation and writing help
│   ├── search_agent.py         # Web search
│   ├── memory.py               # Knowledge graph persistence
│   └── llm.py                  # Local LLM interface
├── memory_palace.json          # Persisted knowledge graph
├── arxiv_papers/               # Downloaded PDFs
├── kaggle_datasets/            # Downloaded datasets
├── hf_cache/                   # Cached HuggingFace models
├── finetuned_models/           # Your fine-tuned models
├── saved_models/               # Trained neural networks
└── writing_output/             # Exported notes and citations
```

## Limitations

- The local LLM quality depends on what you're running. For best results, use a capable model.
- Large dataset training can be slow without a GPU.
- Using a powerful LLM on LM Studio improves quality of research but can slow down your computer depending on your specs. 
- HuggingFace fine-tuning requires sufficient VRAM for the model you're training. 
- Some Kaggle datasets require accepting competition rules on their website first.


## Acknowledgments

- arXiv for open access to academic papers
- Kaggle for dataset hosting
- HuggingFace for the transformers library and model hub
- LM Studio for the backend 
- OpenAI for gpt-oss-20b
- The open source ML community for PyTorch, scikit-learn, Optuna, and countless other tools that make this possible
>>>>>>> 426ea0b (Literature review, Kaggle dataset profiler, fine-tuning capabilites, writing assistant.)
