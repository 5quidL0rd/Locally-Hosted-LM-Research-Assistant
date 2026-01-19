"""
agents/orchestrator.py - AI-driven conversational agent

Routes natural language to appropriate agents using LM reasoning.
Seamlessly integrates with LocalLLM, memory, arXiv, Kaggle, web search,
HuggingFace, and Neural Network Builder.
"""

import re
import json
import os
from typing import Optional, Dict, Any


# ============================================
# TOOL CONTRACT
# ============================================

AGENT_TOOLS = {
    "arxiv_search": "Search academic papers on arXiv",
    "arxiv_search_scored": "Search papers and score them on innovation (1-10)",
    "arxiv_download": "Download and analyze an arXiv paper with innovation scoring",
    "arxiv_score": "Score a specific paper on innovation/uniqueness/impact",
    "kaggle_search": "Search Kaggle datasets (improved with better query handling)",
    "kaggle_download": "Download a Kaggle dataset",
    "kaggle_analyze": "Analyze a downloaded Kaggle dataset and produce plots",
    "kaggle_popular": "Get most popular Kaggle datasets",
    "kaggle_recommend": "Get dataset recommendations for a specific task type",
    "hf_model_search": "Search HuggingFace models",
    "hf_model_download": "Download a HuggingFace model",
    "hf_inference": "Run inference with a HuggingFace model",
    "hf_finetune_image": "Finetune an image classifier on a folder of images",
    "hf_finetune_text": "Finetune a text model on a CSV dataset",
    "hf_finetune_kaggle": "Download Kaggle dataset and finetune HuggingFace model on it",
    "hf_predict_image": "Predict image class using a finetuned model",
    "hf_finetuned_inference": "Run inference with a finetuned model",
    "web_search": "Search the web for current information",
    "memory": "Query or visualize long-term memory",
    "memory_export": "Export memory/research notes to markdown or BibTeX",
    "chat": "General conversation or reasoning",
    "nn_train": "Train neural networks (MLP, LSTM, CNN, Transformer) on tabular CSV data",
    "nn_predict": "Make predictions with a trained neural network model",
    "nn_list": "List all saved neural network models",
    "nn_load": "Load a specific saved neural network model",
    "nn_tune": "Automatically tune hyperparameters for a neural network model using Optuna",
    "profile_dataset": "Generate comprehensive dataset profile with statistics and recommendations",
    "literature_review": "Generate automated literature review on a research topic",
    "suggest_citations": "Suggest relevant citations for a claim or statement",
    "check_citations": "Check text for statements that need citations",
    "write_methodology": "Help write a methodology section for a paper",
    "format_citation": "Format a citation in APA, MLA, Chicago, or BibTeX style",
    "export_citations": "Export all citations from memory in specified format"
}


class ConversationalOrchestrator:
    """
    Interprets natural language and routes to appropriate agents
    using LM reasoning as the primary decision engine.
    """

    def __init__(self, llm, arxiv, kaggle, search, huggingface, memory, nn_builder,
                 data_profiler=None, literature_review=None, writing_assistant=None):
        self.llm = llm
        self.arxiv = arxiv
        self.kaggle = kaggle
        self.search = search
        self.huggingface = huggingface
        self.memory = memory
        self.nn_builder = nn_builder

        # New agents
        self.data_profiler = data_profiler
        self.literature_review = literature_review
        self.writing_assistant = writing_assistant

        # State tracking
        self.active_dataset = None
        self.active_dataset_name = None
        self.last_nn_results = None

        # Search result caches
        self.last_arxiv_results = []
        self.last_kaggle_results = []
        self.last_hf_model_results = []
        self.last_search_results = []

    # ============================================
    # LM ROUTING
    # ============================================

    def _llm_route(self, user_input: str) -> dict:
        """
        Ask the LLM which tool to use.
        Returns JSON like: {"tool":"<tool_name>", "query":"...", "params":{}}
        """
        tool_list = "\n".join(f"- {name}: {desc}" for name, desc in AGENT_TOOLS.items())

        # Add context about active dataset
        context = ""
        if self.active_dataset:
            context = f"\nCurrent active dataset: {self.active_dataset_name or self.active_dataset}"

        prompt = f"""You are an expert research assistant that decides which tool to use.

Available tools:
{tool_list}
{context}

Return ONLY valid JSON with this schema:
{{
  "tool": "<tool_name>",
  "query": "<search query or null>",
  "params": {{}}
}}

Rules:
- For neural network training requests (train, build, compare models, MLP, LSTM, CNN, Transformer), use "nn_train"
- For predictions with saved models, use "nn_predict"
- For listing saved models, use "nn_list"
- If user mentions a file path ending in .csv, extract it for nn_train
- For paper searches, use "arxiv_search"
- For dataset searches, use "kaggle_search"
- For general conversation, use "chat"

User input: "{user_input}"
"""

        response = self.llm.query(prompt, temperature=0.0, max_tokens=400)
        try:
            return json.loads(response)
        except Exception:
            return {"tool": "chat", "query": user_input}

    # ============================================
    # MAIN PROCESS
    # ============================================

    def process(self, user_input: str) -> str:
        """Main entry: route user input to appropriate handler"""

        user_lower = user_input.lower()

        # Fast-path: Memory/research summary requests
        if self._is_memory_request(user_lower):
            return self._handle_memory(user_input)

        # Fast-path: Image finetuning commands
        if self._is_finetune_image_request(user_lower):
            return self._handle_finetune_image(user_input)

        # Fast-path: Direct neural network commands (tabular data only)
        if self._is_nn_request(user_lower):
            return self._route_nn_request(user_input, user_lower)

        # Fast-path: Download commands with numbers
        if user_lower.startswith("download") and any(c.isdigit() for c in user_input):
            return self._handle_download_by_index(user_input)

        # Fast-path: Dataset profiling
        if self._is_profile_request(user_lower):
            return self._handle_profile_dataset(user_input)

        # Fast-path: Literature review
        if self._is_literature_review_request(user_lower):
            return self._handle_literature_review(user_input)

        # Fast-path: Writing/citation assistance
        if self._is_writing_request(user_lower):
            return self._route_writing_request(user_input, user_lower)

        # Fast-path: Memory export
        if self._is_memory_export_request(user_lower):
            return self._handle_memory_export(user_input, user_lower)

        # Fast-path: Model fine-tuning requests
        if self._is_finetune_text_request(user_lower):
            return self._handle_finetune_text(user_input)

        # Fast-path: Kaggle + HuggingFace fine-tuning
        if self._is_finetune_kaggle_request(user_lower):
            return self._handle_finetune_kaggle(user_input)

        # Fast-path: Paper scoring request
        if self._is_paper_score_request(user_lower):
            return self._handle_paper_score(user_input)

        try:
            action = self._llm_route(user_input)
            tool = action.get("tool", "chat")
            query = action.get("query", user_input)
            params = action.get("params", {})

            # Fallback routing for common misclassifications
            tool = self._fallback_routing(tool, user_lower)

            # Route to handler
            return self._dispatch(tool, query, params, user_input)

        except Exception:
            import traceback
            traceback.print_exc()
            return self._handle_conversation(user_input)

    def _is_nn_request(self, user_lower: str) -> bool:
        """Check if this is a neural network related request"""

        # Skip if this is clearly a search/download request for other services
        skip_keywords = ['search', 'find', 'download', 'kaggle', 'arxiv', 'paper',
                        'huggingface', 'hf', 'web', 'memory', 'dataset']
        if any(kw in user_lower for kw in skip_keywords):
            # But allow if explicitly asking to train
            if not any(train_kw in user_lower for train_kw in ['train', 'build nn', 'create nn']):
                return False

        # Explicit NN commands
        nn_explicit = [
            'train', 'neural network', 'mlp', 'lstm', 'cnn',
            'list models', 'saved models', 'load model',
            'build nn', 'create nn', 'train on'
        ]

        # Must have explicit NN keyword
        return any(kw in user_lower for kw in nn_explicit)

    def _route_nn_request(self, user_input: str, user_lower: str) -> str:
        """Route neural network specific requests"""

        # List models
        if 'list' in user_lower and 'model' in user_lower:
            return self._handle_nn_list()

        # Load model
        if 'load' in user_lower and 'model' in user_lower:
            return self._handle_nn_load(user_input)

        # Predict with model
        if 'predict' in user_lower:
            return self._handle_nn_predict(user_input)

        # Training request
        return self._handle_nn_train(user_input)

    def _fallback_routing(self, tool: str, user_lower: str) -> str:
        """Apply fallback routing rules if LLM misclassifies"""

        if tool == "chat":
            if any(k in user_lower for k in ["paper", "papers", "arxiv"]):
                return "arxiv_download" if "download" in user_lower else "arxiv_search"
            if any(k in user_lower for k in ["kaggle", "dataset"]):
                return "kaggle_download" if "download" in user_lower else "kaggle_search"
            if any(k in user_lower for k in ["huggingface", "hf", "model"]):
                if "download" in user_lower:
                    return "hf_model_download"
                return "hf_model_search"
            if any(k in user_lower for k in ["search web", "news", "latest"]):
                return "web_search"

        return tool

    def _dispatch(self, tool: str, query: str, _params: dict, user_input: str) -> str:
        """Dispatch to appropriate handler"""

        handlers = {
            "arxiv_search": lambda: self._handle_arxiv_search(query),
            "arxiv_search_scored": lambda: self._handle_arxiv_search_scored(query),
            "arxiv_download": lambda: self._handle_arxiv_download(query),
            "arxiv_score": lambda: self._handle_paper_score(user_input),
            "kaggle_search": lambda: self._handle_kaggle_search(query),
            "kaggle_download": lambda: self._handle_kaggle_download(query),
            "kaggle_analyze": lambda: self._handle_kaggle_analyze(query),
            "kaggle_popular": lambda: self._handle_kaggle_popular(),
            "kaggle_recommend": lambda: self._handle_kaggle_recommend(user_input),
            "hf_model_search": lambda: self._handle_hf_model_search(query),
            "hf_model_download": lambda: self._handle_hf_model_download(query),
            "hf_inference": lambda: self._handle_hf_inference(query),
            "hf_finetune_image": lambda: self._handle_finetune_image(user_input),
            "hf_finetune_text": lambda: self._handle_finetune_text(user_input),
            "hf_finetune_kaggle": lambda: self._handle_finetune_kaggle(user_input),
            "hf_predict_image": lambda: self._handle_predict_image(user_input),
            "hf_finetuned_inference": lambda: self._handle_finetuned_inference(user_input),
            "web_search": lambda: self._handle_web_search(query),
            "memory": lambda: self._handle_memory(query),
            "memory_export": lambda: self._handle_memory_export(user_input, user_input.lower()),
            "nn_train": lambda: self._handle_nn_train(user_input),
            "nn_predict": lambda: self._handle_nn_predict(user_input),
            "nn_list": lambda: self._handle_nn_list(),
            "nn_load": lambda: self._handle_nn_load(query),
            "nn_tune": lambda: self._handle_nn_tune(user_input),
            "profile_dataset": lambda: self._handle_profile_dataset(user_input),
            "literature_review": lambda: self._handle_literature_review(user_input),
            "suggest_citations": lambda: self._handle_suggest_citations(user_input),
            "check_citations": lambda: self._handle_check_citations(user_input),
            "write_methodology": lambda: self._handle_write_methodology(user_input),
            "format_citation": lambda: self._handle_format_citation(user_input),
            "export_citations": lambda: self._handle_export_citations(user_input),
            "chat": lambda: self._handle_conversation(user_input),
        }

        handler = handlers.get(tool, lambda: self._handle_conversation(user_input))
        return handler()

    def _handle_download_by_index(self, query: str) -> str:
        """Handle 'download N' commands based on last search results"""
        if self.last_kaggle_results:
            return self._handle_kaggle_download(query)
        elif self.last_arxiv_results:
            return self._handle_arxiv_download(query)
        elif self.last_hf_model_results:
            return self._handle_hf_model_download(query)
        return "No recent search results. Please search first."

    # ============================================
    # NEURAL NETWORK HANDLERS
    # ============================================

    def _handle_nn_train(self, user_input: str) -> str:
        """Handle neural network training requests"""

        # Extract dataset path from input
        dataset_path = self._extract_file_path(user_input)

        if not dataset_path:
            dataset_path = self.active_dataset

        if not dataset_path:
            return ("No dataset specified. Please either:\n"
                   "1. Provide a file path: 'train on C:\\path\\to\\data.csv'\n"
                   "2. Download a Kaggle dataset first\n"
                   "3. Specify which dataset to use")

        if not os.path.exists(dataset_path):
            return f"Dataset not found: `{dataset_path}`"

        # If it's a directory, look for CSV files inside
        if os.path.isdir(dataset_path):
            csv_files = []
            for root, dirs, files in os.walk(dataset_path):
                for f in files:
                    if f.lower().endswith('.csv'):
                        csv_files.append(os.path.join(root, f))

            if csv_files:
                dataset_path = csv_files[0]
                print(f"[NNBuilder] Found CSV: {dataset_path}")
            else:
                # Check for image files
                image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
                has_images = any(
                    f.lower().endswith(image_extensions)
                    for root, dirs, files in os.walk(dataset_path)
                    for f in files
                )
                if has_images:
                    return (f"This appears to be an image dataset.\n"
                           f"For image classification, use: 'finetune image classifier on this'\n"
                           f"The neural network builder works with tabular CSV data.")
                else:
                    return f"No CSV files found in `{dataset_path}`"

        # Build experiment spec from natural language
        spec = self._build_nn_spec(user_input, dataset_path)

        # Show what we're about to do
        print(self._format_nn_config(spec))

        try:
            results = self.nn_builder.run_experiment(spec)

            if "error" in results:
                return f"Training Error: {results['error']}"

            self.last_nn_results = results
            return self._format_nn_results(results)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Training failed: {str(e)}"

    def _handle_nn_predict(self, user_input: str) -> str:
        """Handle prediction requests"""

        # Extract model name
        model_name = self._extract_model_name(user_input)

        if not model_name:
            # Try to use the winner from last training
            if self.last_nn_results and 'saved_models' in self.last_nn_results:
                winner = self.last_nn_results.get('winner')
                model_name = self.last_nn_results['saved_models'].get(winner)

        if not model_name:
            models = self.nn_builder.list_models()
            if models:
                return ("Please specify which model to use. Available models:\n" +
                       "\n".join(f"  - {m['name']}" for m in models))
            return "No saved models found. Please train a model first."

        # Extract data path
        data_path = self._extract_file_path(user_input)
        if not data_path:
            data_path = self.active_dataset

        if not data_path:
            return "Please specify data for prediction."

        try:
            predictions = self.nn_builder.predict(model_name, data_path)
            return f"Predictions using `{model_name}`:\n{predictions}"
        except Exception as e:
            return f"Prediction failed: {str(e)}"

    def _handle_nn_list(self) -> str:
        """List all saved neural network models"""
        models = self.nn_builder.list_models()

        if not models:
            return "No saved models found. Train a model first!"

        response = "Saved Neural Network Models:\n" + "=" * 60 + "\n\n"

        for m in models:
            response += f"Name: {m['name']}\n"
            response += f"  Type: {m['model_type'].upper()}\n"
            response += f"  Task: {m['task_type']}\n"

            if 'metrics' in m:
                key_metric = 'accuracy' if m['task_type'] == 'classification' else 'r2'
                if key_metric in m['metrics']:
                    response += f"  {key_metric.upper()}: {m['metrics'][key_metric]:.4f}\n"

            response += f"  Created: {m.get('created', 'unknown')}\n\n"

        response += "=" * 60 + "\n"
        response += "Use 'predict with <model_name>' to make predictions"

        return response

    def _handle_nn_load(self, query: str) -> str:
        """Load a specific model"""
        model_name = self._extract_model_name(query)

        if not model_name:
            return "Please specify which model to load."

        try:
            _, config, _ = self.nn_builder.load_model(model_name)
            return (f"Loaded model: `{model_name}`\n"
                   f"  Type: {config.model_type.upper()}\n"
                   f"  Task: {config.task_type}\n"
                   f"  Input features: {config.input_size}\n"
                   f"Ready for predictions!")
        except Exception as e:
            return f"Failed to load model: {str(e)}"

    def _extract_file_path(self, user_input: str) -> Optional[str]:
        """Extract file path from user input"""

        # Windows absolute path with .csv
        match = re.search(r'[A-Za-z]:\\[^\s]+\.csv', user_input, re.IGNORECASE)
        if match:
            path = match.group(0)
            # Clean up any trailing punctuation
            path = path.rstrip('.,;:!?')
            if os.path.exists(path):
                return path

        # Windows absolute path (directory, no extension)
        match = re.search(r'[A-Za-z]:\\(?:[^\\/:*?"<>|\r\n\s]+\\)*[^\\/:*?"<>|\r\n\s]+',
                         user_input, re.IGNORECASE)
        if match:
            path = match.group(0).rstrip('.,;:!?')
            if os.path.exists(path):
                return path

        # Unix path
        match = re.search(r'/(?:[^/\0\s]+/)*[^/\0\s]+\.csv', user_input)
        if match:
            path = match.group(0).rstrip('.,;:!?')
            if os.path.exists(path):
                return path

        # Relative path with .csv
        match = re.search(r'[\w\-./\\]+\.csv', user_input, re.IGNORECASE)
        if match:
            path = match.group(0).rstrip('.,;:!?')
            if os.path.exists(path):
                return path

        return None

    def _extract_model_name(self, user_input: str) -> Optional[str]:
        """Extract model name from user input"""

        # Check for timestamp pattern (model names)
        match = re.search(r'\d{8}_\d{6}_\w+', user_input)
        if match:
            return match.group(0)

        # Check against saved models
        models = self.nn_builder.registry.list_models()
        for m in models:
            if m['name'].lower() in user_input.lower():
                return m['name']

        return None

    def _build_nn_spec(self, user_input: str, dataset_path: str) -> Dict[str, Any]:
        """Build experiment specification from natural language"""
        user_lower = user_input.lower()

        spec = {
            "dataset": {"path": dataset_path},
            "models": [],
            "epochs": 50,
            "batch_size": 32,
            "sequence_length": 20,
            "save_models": True
        }

        # Extract target column
        target_patterns = [
            r'predict\s+["\']?(\w+)["\']?',
            r'target\s+(?:is|column)?\s*["\']?(\w+)["\']?',
            r'for\s+["\']?(\w+)["\']?\s+prediction'
        ]

        for pattern in target_patterns:
            match = re.search(pattern, user_lower)
            if match:
                spec["dataset"]["target_column"] = match.group(1)
                break

        # Extract models
        if 'mlp' in user_lower or 'perceptron' in user_lower:
            spec["models"].append("mlp")
        if 'lstm' in user_lower or 'recurrent' in user_lower:
            spec["models"].append("lstm")
        if 'cnn' in user_lower or 'convolutional' in user_lower:
            spec["models"].append("cnn")
        if 'transformer' in user_lower or 'attention' in user_lower:
            spec["models"].append("transformer")
        if 'all' in user_lower and 'model' in user_lower:
            spec["models"] = ["mlp", "lstm", "cnn", "transformer"]

        # Default models
        if not spec["models"]:
            spec["models"] = ["mlp", "lstm"]

        # Extract hyperparameters
        epoch_match = re.search(r'(\d+)\s*epoch', user_lower)
        if epoch_match:
            spec["epochs"] = int(epoch_match.group(1))

        batch_match = re.search(r'batch\s*(?:size)?\s*(\d+)', user_lower)
        if batch_match:
            spec["batch_size"] = int(batch_match.group(1))

        seq_match = re.search(r'(?:sequence|window)\s*(?:length)?\s*(\d+)', user_lower)
        if seq_match:
            spec["sequence_length"] = int(seq_match.group(1))

        return spec

    def _format_nn_config(self, spec: Dict[str, Any]) -> str:
        """Format configuration message"""
        msg = "\n" + "=" * 70 + "\n"
        msg += "NEURAL NETWORK TRAINING\n"
        msg += "=" * 70 + "\n\n"

        msg += f"Dataset: {spec['dataset']['path']}\n"

        if 'target_column' in spec['dataset']:
            msg += f"Target: {spec['dataset']['target_column']}\n"
        else:
            msg += "Target: Auto-detect\n"

        msg += f"Models: {', '.join(m.upper() for m in spec['models'])}\n"
        msg += f"Epochs: {spec['epochs']}\n"
        msg += f"Batch Size: {spec['batch_size']}\n"

        if any(m in spec['models'] for m in ['lstm', 'cnn', 'transformer']):
            msg += f"Sequence Length: {spec['sequence_length']}\n"

        msg += "\n" + "=" * 70 + "\n"

        return msg

    def _format_nn_results(self, results: Dict[str, Any]) -> str:
        """Format training results"""
        msg = "\n" + "=" * 70 + "\n"
        msg += "TRAINING COMPLETE\n"
        msg += "=" * 70 + "\n\n"

        msg += f"Task Type: {results['task_type'].capitalize()}\n"
        msg += f"Winner: {results['winner'].upper()}\n"
        msg += f"Best Score: {results['winner_score']:.4f}\n\n"

        msg += "Final Metrics:\n"
        msg += "-" * 40 + "\n"

        for model_name, metrics in results['results'].items():
            msg += f"\n{model_name.upper()}:\n"
            for metric, value in metrics.items():
                msg += f"  {metric:12s}: {value:.4f}\n"

        if 'visualization' in results:
            msg += f"\nVisualization: {results['visualization']}\n"

        if 'saved_models' in results:
            msg += "\nSaved Models:\n"
            for model_type, name in results['saved_models'].items():
                msg += f"  {model_type}: {name}\n"

        msg += "\n" + "=" * 70 + "\n"
        msg += "Use 'list models' to see all saved models\n"
        msg += "Use 'predict with <model_name>' to make predictions"

        return msg

    # ============================================
    # ARXIV HANDLERS
    # ============================================

    def _handle_arxiv_search(self, query: str) -> str:
        papers = self.arxiv.search_papers(query, max_results=5)
        if not papers or (isinstance(papers, list) and papers and 'error' in papers[0]):
            return "No papers found. Try rephrasing your query."

        self.last_arxiv_results = papers
        response = "Found papers:\n\n"

        for i, paper in enumerate(papers, 1):
            response += f"{i}. **{paper['title']}**\n"
            response += f"   Authors: {', '.join(paper['authors'][:3])}\n"
            response += f"   ID: `{paper['arxiv_id']}`\n"
            response += f"   Summary: {paper['summary'][:150]}...\n\n"

        response += "Say 'download 1' or 'download <arxiv_id>' to analyze a paper."
        return response

    def _handle_arxiv_download(self, query: str) -> str:
        if not query:
            return "Need an arXiv ID or result number."

        # Direct arXiv ID
        matches = re.findall(r'\d{4}\.\d{4,5}(?:v\d+)?', query)
        if matches:
            outputs = []
            for arxiv_id in matches:
                analysis = self.arxiv.analyze_paper(arxiv_id)
                outputs.append(f"**Paper Analysis ({arxiv_id}):**\n\n{analysis}")
            return "\n\n".join(outputs)

        # Numeric index
        nums = re.findall(r'\b(\d+)\b', query)
        if nums and self.last_arxiv_results:
            outputs = []
            for num in nums:
                idx = int(num) - 1
                if 0 <= idx < len(self.last_arxiv_results):
                    arxiv_id = self.last_arxiv_results[idx]['arxiv_id']
                    analysis = self.arxiv.analyze_paper(arxiv_id)
                    outputs.append(f"**{self.last_arxiv_results[idx]['title']}**\n\n{analysis}")
            return "\n\n".join(outputs) if outputs else "Index out of range."

        return "Need a valid arXiv ID or result number."

    # ============================================
    # KAGGLE HANDLERS
    # ============================================

    def _handle_kaggle_search(self, query: str) -> str:
        datasets = self.kaggle.search_datasets(query, max_results=8)
        if not datasets or (isinstance(datasets, list) and datasets and 'error' in datasets[0]):
            return f"No datasets found for '{query}'."

        self.last_kaggle_results = datasets
        response = f"Found {len(datasets)} datasets:\n\n"

        for i, ds in enumerate(datasets, 1):
            response += f"{i}. **{ds['title']}**\n"
            response += f"   Ref: `{ds['ref']}`\n"
            response += f"   Downloads: {ds['download_count']:,}\n\n"

        response += "Say 'download 1' or use the dataset ref to download."
        return response

    def _handle_kaggle_download(self, query: str) -> str:
        if not query:
            return "Need a dataset reference or result number."

        # Extract dataset ref
        match = re.search(r'[\w-]+/[\w-]+', query)
        if match:
            dataset_ref = match.group(0)
        else:
            num_match = re.search(r'\b(\d+)\b', query)
            if num_match and self.last_kaggle_results:
                idx = int(num_match.group(1)) - 1
                if 0 <= idx < len(self.last_kaggle_results):
                    dataset_ref = self.last_kaggle_results[idx]['ref']
                else:
                    return "Index out of range."
            else:
                return "Need a valid dataset reference or number."

        result = self.kaggle.download_dataset(dataset_ref)
        if not result:
            return f"Download failed for `{dataset_ref}`."

        # Set as active dataset
        files = result.get("files", [])
        csv_files = [f for f in files if f.lower().endswith(".csv")]

        # Categorize files
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')
        image_files = [f for f in files if f.lower().endswith(image_extensions)]

        self.active_dataset_name = dataset_ref
        dataset_type = "unknown"

        if csv_files:
            self.active_dataset = csv_files[0]
            dataset_type = "tabular"
        elif image_files:
            self.active_dataset = result.get("path")
            dataset_type = "images"
        else:
            self.active_dataset = result.get("path")

        file_list = [f.split('/')[-1].split('\\')[-1] for f in files[:5]]

        response = f"Downloaded: `{dataset_ref}`\n"
        response += f"Files: {', '.join(file_list)}\n"
        response += f"Dataset type: {dataset_type}\n"
        response += f"\nActive dataset: `{self.active_dataset}`\n"

        if dataset_type == "tabular":
            response += "\nYou can now say 'train on this dataset' to build neural networks!"
        elif dataset_type == "images":
            response += f"\nThis is an image dataset with {len(image_files)} images."
            response += "\nFor image classification, say 'finetune image classifier on this'"
        else:
            response += "\nExplore the files to see what's available."

        return response

    def _handle_kaggle_analyze(self, query: str) -> str:
        match = re.search(r'[\w-]+/[\w-]+', query)
        if match:
            dataset_ref = match.group(0)
        else:
            num_match = re.search(r'\b(\d+)\b', query)
            if num_match and self.last_kaggle_results:
                idx = int(num_match.group(1)) - 1
                if 0 <= idx < len(self.last_kaggle_results):
                    dataset_ref = self.last_kaggle_results[idx]['ref']
                else:
                    return "Index out of range."
            else:
                return "Specify which dataset to analyze."

        analysis = self.kaggle.analyze_dataset(dataset_ref)
        if not analysis or 'error' in analysis:
            return f"Analysis failed: {analysis.get('error') if isinstance(analysis, dict) else analysis}"

        summary = analysis.get('analysis', '')[:1000]
        plots = analysis.get('plots', [])

        return f"Analysis complete for `{dataset_ref}`.\n\n{summary}\n\nPlots: {', '.join(plots) if plots else 'none'}"

    # ============================================
    # HUGGINGFACE HANDLERS
    # ============================================

    def _handle_hf_model_search(self, query: str) -> str:
        models = self.huggingface.search_models(query, limit=5)
        if not models or (isinstance(models, list) and models and 'error' in models[0]):
            return "No models found."

        self.last_hf_model_results = models
        response = f"Found {len(models)} models:\n\n"

        for i, m in enumerate(models, 1):
            response += f"{i}. {m['id']}\n"
            response += f"   Downloads: {m.get('downloads', 0):,}\n"
            response += f"   Task: {m.get('pipeline_tag', 'unknown')}\n\n"

        return response

    def _handle_hf_model_download(self, query: str) -> str:
        if not query:
            return "Specify a model ID or index."

        # Extract model ID
        model_id = None
        repo_match = re.search(r"[\w\-_]+/[\w\-\._]+", query)
        if repo_match:
            model_id = repo_match.group(0)

        if not model_id:
            num_match = re.search(r'\b(\d+)\b', query)
            if num_match and self.last_hf_model_results:
                idx = int(num_match.group(1)) - 1
                if 0 <= idx < len(self.last_hf_model_results):
                    model_id = self.last_hf_model_results[idx]['id']

        if not model_id:
            return "Could not determine model ID."

        path = self.huggingface.download_model(model_id)
        if not path:
            return f"Download failed for {model_id}."

        return f"Downloaded: {model_id}"

    def _handle_hf_inference(self, query: str) -> str:
        match = re.search(r'with\s+([\w-]+(?:/[\w-]+)?)\s*:\s*(.+)', query)
        if not match:
            return "Format: 'Generate with MODEL_ID: prompt'"

        model_id, prompt = match.group(1), match.group(2)
        result = self.huggingface.run_inference(model_id, prompt, max_length=150)
        return f"**Generated text:**\n\n{result}"

    # ============================================
    # IMAGE FINETUNING HANDLERS
    # ============================================

    def _is_finetune_image_request(self, user_lower: str) -> bool:
        """Check if this is an image finetuning request"""
        finetune_keywords = [
            'finetune image', 'fine-tune image', 'fine tune image',
            'train image classifier', 'image classification',
            'finetune classifier', 'finetune on images'
        ]
        return any(kw in user_lower for kw in finetune_keywords)

    def _handle_finetune_image(self, user_input: str) -> str:
        """Handle image classifier finetuning requests"""

        # Extract image folder path
        image_folder = self._extract_folder_path(user_input)

        if not image_folder:
            image_folder = self.active_dataset

        if not image_folder:
            return ("No image folder specified. Please either:\n"
                   "1. Provide a folder path: 'finetune image classifier on C:\\path\\to\\images'\n"
                   "2. Download a Kaggle image dataset first\n"
                   "The folder should contain subfolders for each class (e.g., cats/, dogs/)")

        if not os.path.exists(image_folder):
            return f"Folder not found: `{image_folder}`"

        if not os.path.isdir(image_folder):
            return f"Expected a folder, not a file: `{image_folder}`"

        # Check for class subdirectories
        subdirs = [d for d in os.listdir(image_folder) if os.path.isdir(os.path.join(image_folder, d))]
        if not subdirs:
            return (f"No class subdirectories found in `{image_folder}`.\n"
                   "Expected structure:\n"
                   "  image_folder/\n"
                   "    class1/\n"
                   "      img1.jpg\n"
                   "    class2/\n"
                   "      img2.jpg")

        # Extract parameters from natural language
        user_lower = user_input.lower()

        epochs = 5
        epoch_match = re.search(r'(\d+)\s*epoch', user_lower)
        if epoch_match:
            epochs = int(epoch_match.group(1))

        batch_size = 16
        batch_match = re.search(r'batch\s*(?:size)?\s*(\d+)', user_lower)
        if batch_match:
            batch_size = int(batch_match.group(1))

        # Default base model
        base_model = "google/vit-base-patch16-224"
        if 'resnet' in user_lower:
            base_model = "microsoft/resnet-50"
        elif 'efficientnet' in user_lower:
            base_model = "google/efficientnet-b0"

        print(f"\n{'='*60}")
        print("IMAGE CLASSIFIER FINETUNING")
        print(f"{'='*60}")
        print(f"Folder: {image_folder}")
        print(f"Classes: {subdirs}")
        print(f"Base Model: {base_model}")
        print(f"Epochs: {epochs}")
        print(f"Batch Size: {batch_size}")
        print(f"{'='*60}\n")

        try:
            result = self.huggingface.finetune_image_classifier(
                image_folder=image_folder,
                base_model=base_model,
                epochs=epochs,
                batch_size=batch_size
            )

            if "error" in result:
                return f"Finetuning Error: {result['error']}"

            return self._format_finetune_results(result)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Finetuning failed: {str(e)}"

    def _extract_folder_path(self, user_input: str) -> Optional[str]:
        """Extract folder path from user input"""

        # Windows path
        match = re.search(r'[A-Za-z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*',
                         user_input, re.IGNORECASE)
        if match and os.path.isdir(match.group(0)):
            return match.group(0)

        # Unix path
        match = re.search(r'/(?:[^/\0]+/)*[^/\0]+', user_input)
        if match and os.path.isdir(match.group(0)):
            return match.group(0)

        # Relative path
        match = re.search(r'\.?/?[\w\-./]+', user_input, re.IGNORECASE)
        if match and os.path.isdir(match.group(0)):
            return match.group(0)

        return None

    def _format_finetune_results(self, result: dict) -> str:
        """Format finetuning results"""
        msg = "\n" + "=" * 60 + "\n"
        msg += "FINETUNING COMPLETE\n"
        msg += "=" * 60 + "\n\n"

        msg += f"Final Accuracy: {result.get('accuracy', 0):.4f}\n"
        msg += f"Classes: {', '.join(result.get('classes', []))}\n"
        msg += f"Training Samples: {result.get('train_samples', 0)}\n"
        msg += f"Validation Samples: {result.get('val_samples', 0)}\n"

        if result.get('model_path'):
            msg += f"\nModel saved to: `{result['model_path']}`\n"

        if result.get('visualization'):
            msg += f"Visualization: `{result['visualization']}`\n"

        msg += "\n" + "=" * 60 + "\n"
        msg += "Use 'predict image <path> with <model_path>' to classify new images"

        return msg

    def _handle_predict_image(self, user_input: str) -> str:
        """Handle image prediction requests"""

        # Extract image path
        image_path = self._extract_file_path(user_input)
        if not image_path:
            return "Please specify an image path to classify."

        # Extract model path
        model_path = None

        # Look for finetuned_models path
        match = re.search(r'finetuned_models[\\\/][\w_]+[\\\/]final', user_input)
        if match:
            model_path = match.group(0)

        if not model_path:
            # List available models
            models = self.huggingface.list_finetuned_models()
            if models:
                model_path = models[-1]['path']  # Use most recent
                print(f"Using most recent model: {model_path}")
            else:
                return "No finetuned models found. Finetune a model first!"

        result = self.huggingface.predict_image(model_path, image_path)

        if "error" in result:
            return f"Prediction Error: {result['error']}"

        response = f"**Prediction for `{image_path}`:**\n\n"
        response += f"Class: {result['predicted_class']}\n"
        response += f"Confidence: {result['confidence']:.2%}\n\n"

        if result.get('all_probs'):
            response += "All probabilities:\n"
            for cls, prob in sorted(result['all_probs'].items(), key=lambda x: -x[1]):
                response += f"  {cls}: {prob:.2%}\n"

        return response

    # ============================================
    # WEB SEARCH HANDLER
    # ============================================

    def _handle_web_search(self, query: str) -> str:
        results = self.search.run_task(query, max_results=5)
        if not results or (isinstance(results, list) and results and 'error' in results[0]):
            return "Web search failed."

        self.last_search_results = results
        response = "Search results:\n\n"

        for i, r in enumerate(results, 1):
            response += f"{i}. {r['title']}\n   {r['url']}\n\n"

        return response

    # ============================================
    # MEMORY HANDLER
    # ============================================

    def _is_memory_request(self, user_lower: str) -> bool:
        """Check if this is a memory/research summary request"""
        memory_keywords = [
            'what have i research', 'what did i research', 'my research',
            'what have i studied', 'what did i study',
            'what have i looked', 'what did i look',
            'what have i explored', 'what did i explore',
            'what have i done', 'what did i do',
            'show my research', 'show my work', 'show my progress',
            'research history', 'research summary', 'my history',
            'knowledge graph', 'show graph', 'visualize',
            'what papers', 'which papers', 'papers i',
            'show memory', 'my memory', 'recall'
        ]
        return any(kw in user_lower for kw in memory_keywords)

    def _handle_memory(self, query: str) -> str:
        query_lower = query.lower()

        # Check if user wants to see what they've researched
        research_keywords = ['research', 'studied', 'looked at', 'explored',
                           'worked on', 'done', 'history', 'summary', 'what have i']
        wants_summary = any(kw in query_lower for kw in research_keywords)

        # Check if user wants visualization
        wants_viz = 'visualiz' in query_lower or 'graph' in query_lower or 'show' in query_lower

        # Get summary stats
        stats = self.memory.get_summary_stats()

        if stats['total_nodes'] == 0:
            return "Your knowledge graph is empty. Start by searching papers, downloading datasets, or running experiments!"

        # Build response
        response = "\n" + "=" * 60 + "\n"
        response += "YOUR RESEARCH SUMMARY\n"
        response += "=" * 60 + "\n\n"

        response += f"**Total Items:** {stats['total_nodes']}\n"
        response += f"**Connections:** {stats['total_edges']}\n\n"

        # Show breakdown by type
        if stats.get('nodes_by_type'):
            response += "**By Category:**\n"
            type_labels = {
                'arxiv_paper': 'Papers',
                'kaggle_dataset': 'Datasets',
                'nn_experiment': 'NN Experiments',
                'hf_model': 'HuggingFace Models',
                'web_search': 'Web Searches',
                'lit_review': 'Literature Reviews'
            }
            for node_type, count in stats['nodes_by_type'].items():
                label = type_labels.get(node_type, node_type)
                response += f"  - {label}: {count}\n"

        # Show recent items
        recent = self.memory.get_recent_nodes(limit=5)
        if recent:
            response += "\n**Recent Activity:**\n"
            for item in recent:
                node_data = item.get('data', {})
                if isinstance(node_data, dict):
                    title = node_data.get('title', item.get('node', 'Unknown'))[:60]
                else:
                    title = str(item.get('node', 'Unknown'))[:60]
                response += f"  - {title}\n"

        # Show papers with high innovation scores
        papers = [n for n, d in self.memory.graph.nodes(data=True)
                 if d.get('type') == 'arxiv_paper']
        if papers:
            response += f"\n**Papers Analyzed:** {len(papers)}\n"

            # Get top scored papers
            scored_papers = []
            for node in papers:
                data = self.memory.graph.nodes[node].get('data', {})
                analysis = data.get('analysis', {})
                score = analysis.get('innovation_score', 0)
                if score:
                    scored_papers.append((data.get('title', node), score))

            if scored_papers:
                scored_papers.sort(key=lambda x: x[1], reverse=True)
                response += "\n**Top Innovative Papers:**\n"
                for title, score in scored_papers[:3]:
                    response += f"  - [{score}/10] {title[:50]}...\n"

        response += "\n" + "=" * 60 + "\n"

        # Generate and open visualization
        if wants_viz or wants_summary:
            path = self.memory.visualize(open_browser=True)
            if path:
                response += f"\nKnowledge graph opened in browser!\n"
                response += f"Saved to: `{path}`\n"

        return response

    # ============================================
    # CONVERSATION HANDLER
    # ============================================

    def _handle_conversation(self, query: str) -> str:
        prompt = f"""You are a helpful research assistant.
User: "{query}"
Respond concisely and helpfully."""

        return self.llm.query(prompt, max_tokens=300, temperature=0.7)

    # ============================================
    # DATA PROFILING HANDLERS
    # ============================================

    def _is_profile_request(self, user_lower: str) -> bool:
        """Check if this is a dataset profiling request"""
        profile_keywords = [
            'profile', 'analyze data', 'data quality', 'dataset statistics',
            'describe dataset', 'data report', 'inspect data', 'data summary'
        ]
        return any(kw in user_lower for kw in profile_keywords)

    def _handle_profile_dataset(self, user_input: str) -> str:
        """Handle dataset profiling requests"""
        if not self.data_profiler:
            return "Data profiler not initialized. Please restart the assistant."

        # Extract dataset path
        dataset_path = self._extract_file_path(user_input)

        if not dataset_path:
            dataset_path = self.active_dataset

        if not dataset_path:
            return ("No dataset specified. Please either:\n"
                   "1. Provide a file path: 'profile C:\\path\\to\\data.csv'\n"
                   "2. Download a Kaggle dataset first")

        if not os.path.exists(dataset_path):
            return f"Dataset not found: `{dataset_path}`"

        try:
            profile = self.data_profiler.profile_dataset(dataset_path)

            if "error" in profile:
                return f"Profiling Error: {profile['error']}"

            # Format response
            response = f"\n{'='*60}\n"
            response += "DATASET PROFILE COMPLETE\n"
            response += f"{'='*60}\n\n"

            response += f"**File:** `{profile['file_name']}`\n"
            response += f"**Quality Score:** {profile['quality_score']:.1f}/100\n\n"

            stats = profile['basic_stats']
            response += f"**Shape:** {stats['rows']:,} rows x {stats['columns']} columns\n"
            response += f"**Missing:** {stats['total_missing_pct']}%\n"
            response += f"**Duplicates:** {stats['duplicate_pct']}%\n\n"

            target = profile['target_analysis']
            response += f"**Task Type:** {target.get('task_type', 'unknown')}\n"

            if target.get('is_imbalanced'):
                response += f"**Warning:** Class imbalance detected (ratio: {target.get('imbalance_ratio', 0):.1f}x)\n"

            # Top recommendations
            if profile.get('recommendations'):
                response += "\n**Top Recommendations:**\n"
                for rec in profile['recommendations'][:3]:
                    response += f"  - [{rec['priority'].upper()}] {rec['issue']}\n"

            if profile.get('report_path'):
                response += f"\n**Full Report:** `{profile['report_path']}`\n"

            return response

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Profiling failed: {str(e)}"

    # ============================================
    # LITERATURE REVIEW HANDLERS
    # ============================================

    def _is_literature_review_request(self, user_lower: str) -> bool:
        """Check if this is a literature review request"""
        review_keywords = [
            'literature review', 'lit review', 'review papers',
            'survey papers', 'research survey', 'paper survey',
            'systematic review', 'generate review'
        ]
        return any(kw in user_lower for kw in review_keywords)

    def _handle_literature_review(self, user_input: str) -> str:
        """Handle literature review requests"""
        if not self.literature_review:
            return "Literature review agent not initialized. Please restart the assistant."

        # Extract topic from input
        topic = self._extract_topic(user_input)

        if not topic:
            return ("Please specify a research topic. Example:\n"
                   "'Generate literature review on transformer architectures for NLP'")

        # Extract parameters
        user_lower = user_input.lower()

        max_papers = 10
        papers_match = re.search(r'(\d+)\s*papers?', user_lower)
        if papers_match:
            max_papers = min(int(papers_match.group(1)), 20)

        depth = "standard"
        if 'quick' in user_lower or 'fast' in user_lower:
            depth = "quick"
        elif 'deep' in user_lower or 'thorough' in user_lower:
            depth = "deep"

        try:
            review = self.literature_review.generate_review(
                research_question=topic,
                max_papers=max_papers,
                analyze_depth=depth
            )

            response = f"\n{'='*60}\n"
            response += "LITERATURE REVIEW COMPLETE\n"
            response += f"{'='*60}\n\n"

            response += f"**Topic:** {topic}\n"
            response += f"**Papers Analyzed:** {len(review.get('papers', []))}\n"
            response += f"**Themes Identified:** {len(review.get('themes', []))}\n"
            response += f"**Research Gaps:** {len(review.get('gaps', []))}\n\n"

            if review.get('key_findings'):
                response += "**Key Findings:**\n"
                for finding in review['key_findings'][:5]:
                    response += f"  - {finding}\n"

            if review.get('gaps'):
                response += "\n**Research Gaps:**\n"
                for gap in review['gaps'][:3]:
                    if isinstance(gap, dict):
                        response += f"  - {gap.get('gap', gap)}\n"
                    else:
                        response += f"  - {gap}\n"

            response += f"\n**Full Report:** `{review.get('report_path', 'N/A')}`\n"
            response += f"**Citations:** `{review.get('bibtex_path', 'N/A')}`\n"

            return response

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Literature review failed: {str(e)}"

    def _extract_topic(self, user_input: str) -> Optional[str]:
        """Extract research topic from user input"""
        # Remove common prefixes
        patterns = [
            r'(?:generate|create|do|write|make)\s+(?:a\s+)?(?:literature|lit)?\s*review\s+(?:on|about|for)\s+(.+)',
            r'(?:literature|lit)\s+review\s+(?:on|about|for)\s+(.+)',
            r'review\s+papers?\s+(?:on|about|for)\s+(.+)',
            r'survey\s+(?:on|about|for)\s+(.+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Fallback: take everything after key phrases
        for phrase in ['on ', 'about ', 'for ', 'review ']:
            if phrase in user_input.lower():
                idx = user_input.lower().find(phrase) + len(phrase)
                return user_input[idx:].strip()

        return None

    # ============================================
    # WRITING ASSISTANT HANDLERS
    # ============================================

    def _is_writing_request(self, user_lower: str) -> bool:
        """Check if this is a writing assistance request"""
        writing_keywords = [
            'suggest citation', 'cite this', 'need citation',
            'check citation', 'missing citation',
            'write methodology', 'methodology section',
            'format citation', 'bibtex', 'apa format', 'mla format',
            'export citation', 'generate citation'
        ]
        return any(kw in user_lower for kw in writing_keywords)

    def _route_writing_request(self, user_input: str, user_lower: str) -> str:
        """Route to appropriate writing handler"""
        if 'suggest' in user_lower and 'citation' in user_lower:
            return self._handle_suggest_citations(user_input)
        elif 'check' in user_lower and 'citation' in user_lower:
            return self._handle_check_citations(user_input)
        elif 'methodology' in user_lower:
            return self._handle_write_methodology(user_input)
        elif 'format' in user_lower and 'citation' in user_lower:
            return self._handle_format_citation(user_input)
        elif 'export' in user_lower and 'citation' in user_lower:
            return self._handle_export_citations(user_input)
        else:
            return self._handle_suggest_citations(user_input)

    def _handle_suggest_citations(self, user_input: str) -> str:
        """Handle citation suggestion requests"""
        if not self.writing_assistant:
            return "Writing assistant not initialized. Please restart the assistant."

        # Extract the claim to cite
        claim = self._extract_claim(user_input)

        if not claim:
            return ("Please specify the claim you need citations for. Example:\n"
                   "'Suggest citations for: Transformer models outperform RNNs on long sequences'")

        try:
            suggestions = self.writing_assistant.suggest_citations(claim)

            if suggestions and 'message' in suggestions[0]:
                return suggestions[0]['message']

            response = f"**Citation suggestions for:** \"{claim}\"\n\n"
            for s in suggestions:
                response += f"**{s['rank']}. {s['title']}**\n"
                response += f"   ArXiv: `{s['arxiv_id']}`\n"
                response += f"   Authors: {', '.join(s['authors'][:3])}\n"
                response += f"   Reason: {s['reason']}\n\n"

            return response

        except Exception as e:
            return f"Citation suggestion failed: {str(e)}"

    def _extract_claim(self, user_input: str) -> Optional[str]:
        """Extract claim from user input"""
        patterns = [
            r'suggest\s+citation[s]?\s+for[:\s]+(.+)',
            r'cite[:\s]+(.+)',
            r'citation[s]?\s+for[:\s]+(.+)',
            r'need\s+citation[s]?\s+for[:\s]+(.+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Fallback: take text after colon
        if ':' in user_input:
            return user_input.split(':', 1)[1].strip()

        return None

    def _handle_check_citations(self, user_input: str) -> str:
        """Handle citation checking requests"""
        if not self.writing_assistant:
            return "Writing assistant not initialized. Please restart the assistant."

        # Extract text to check
        text = self._extract_text_content(user_input)

        if not text or len(text) < 50:
            return ("Please provide the text to check for needed citations. Example:\n"
                   "'Check citations: [paste your text here]'")

        try:
            claims = self.writing_assistant.check_citations_needed(text)

            if 'error' in claims[0]:
                return f"Error: {claims[0]['error']}"

            response = f"**Found {len(claims)} statements that may need citations:**\n\n"

            for i, claim in enumerate(claims, 1):
                response += f"**{i}. {claim['claim'][:100]}...**\n"
                response += f"   Type: {claim.get('citation_type', 'N/A')}\n"
                response += f"   Reason: {claim.get('reason', 'N/A')}\n"

                if claim.get('suggested_citations'):
                    response += "   Suggested:\n"
                    for s in claim['suggested_citations'][:2]:
                        if isinstance(s, dict) and 'title' in s:
                            response += f"     - {s['title']} ({s['arxiv_id']})\n"
                response += "\n"

            return response

        except Exception as e:
            return f"Citation check failed: {str(e)}"

    def _extract_text_content(self, user_input: str) -> Optional[str]:
        """Extract text content from user input"""
        if ':' in user_input:
            return user_input.split(':', 1)[1].strip()
        return user_input

    def _handle_write_methodology(self, user_input: str) -> str:
        """Handle methodology writing requests"""
        if not self.writing_assistant:
            return "Writing assistant not initialized. Please restart the assistant."

        # Extract experiment description
        description = self._extract_text_content(user_input)

        if not description or len(description) < 30:
            return ("Please describe your experiment/method. Example:\n"
                   "'Write methodology: We trained a CNN on MNIST dataset using Adam optimizer...'")

        try:
            methodology = self.writing_assistant.generate_methodology_section(description)
            return f"**Generated Methodology Section:**\n\n{methodology}"

        except Exception as e:
            return f"Methodology generation failed: {str(e)}"

    def _handle_format_citation(self, user_input: str) -> str:
        """Handle citation formatting requests"""
        if not self.writing_assistant:
            return "Writing assistant not initialized. Please restart the assistant."

        # Extract arxiv ID
        arxiv_match = re.search(r'\d{4}\.\d{4,5}(?:v\d+)?', user_input)
        if not arxiv_match:
            return "Please provide an ArXiv ID. Example: 'Format citation 2301.00001 as APA'"

        arxiv_id = arxiv_match.group(0)

        # Detect style
        user_lower = user_input.lower()
        if 'bibtex' in user_lower or 'bib' in user_lower:
            style = 'bibtex'
        elif 'mla' in user_lower:
            style = 'mla'
        elif 'chicago' in user_lower:
            style = 'chicago'
        else:
            style = 'apa'

        try:
            citation = self.writing_assistant.format_citation(arxiv_id, style)
            return f"**{style.upper()} Citation:**\n\n```\n{citation}\n```"

        except Exception as e:
            return f"Citation formatting failed: {str(e)}"

    def _handle_export_citations(self, user_input: str) -> str:
        """Handle citation export requests"""
        if not self.writing_assistant:
            return "Writing assistant not initialized. Please restart the assistant."

        # Detect style
        user_lower = user_input.lower()
        if 'apa' in user_lower:
            style = 'apa'
        elif 'mla' in user_lower:
            style = 'mla'
        else:
            style = 'bibtex'

        try:
            path = self.writing_assistant.export_all_citations(style)
            return f"**Citations exported:** `{path}`"

        except Exception as e:
            return f"Citation export failed: {str(e)}"

    # ============================================
    # MEMORY EXPORT HANDLERS
    # ============================================

    def _is_memory_export_request(self, user_lower: str) -> bool:
        """Check if this is a memory export request"""
        export_keywords = [
            'export memory', 'export research', 'export notes',
            'save memory', 'memory to markdown', 'export knowledge',
            'research notes', 'export bibtex'
        ]
        return any(kw in user_lower for kw in export_keywords)

    def _handle_memory_export(self, user_input: str, user_lower: str) -> str:
        """Handle memory export requests"""
        if 'bibtex' in user_lower or 'bib' in user_lower:
            path = self.memory.export_citations_bibtex()
            if path:
                return f"**BibTeX citations exported:** `{path}`"
            return "No papers found to export."

        else:
            path = self.memory.export_to_markdown()
            if path:
                stats = self.memory.get_summary_stats()
                response = f"**Research notes exported:** `{path}`\n\n"
                response += f"**Contents:**\n"
                response += f"  - Total items: {stats['total_nodes']}\n"
                response += f"  - Connections: {stats['total_edges']}\n"
                for node_type, count in stats.get('nodes_by_type', {}).items():
                    response += f"  - {node_type}: {count}\n"
                return response
            return "Export failed."

    # ============================================
    # HYPERPARAMETER TUNING HANDLER
    # ============================================

    def _handle_nn_tune(self, user_input: str) -> str:
        """Handle hyperparameter tuning requests"""
        # Extract dataset path
        dataset_path = self._extract_file_path(user_input)

        if not dataset_path:
            dataset_path = self.active_dataset

        if not dataset_path:
            return ("No dataset specified. Please either:\n"
                   "1. Provide a file path: 'tune MLP on C:\\path\\to\\data.csv'\n"
                   "2. Download a Kaggle dataset first")

        if not os.path.exists(dataset_path):
            return f"Dataset not found: `{dataset_path}`"

        # Extract model type
        user_lower = user_input.lower()
        model_type = 'mlp'  # default
        if 'lstm' in user_lower:
            model_type = 'lstm'
        elif 'cnn' in user_lower:
            model_type = 'cnn'
        elif 'transformer' in user_lower:
            model_type = 'transformer'

        # Extract trial count
        n_trials = 30
        trials_match = re.search(r'(\d+)\s*trials?', user_lower)
        if trials_match:
            n_trials = int(trials_match.group(1))

        try:
            result = self.nn_builder.tune_hyperparameters(
                dataset_path=dataset_path,
                model_type=model_type,
                n_trials=n_trials
            )

            if "error" in result:
                return f"Tuning Error: {result['error']}"

            response = f"\n{'='*60}\n"
            response += "HYPERPARAMETER TUNING COMPLETE\n"
            response += f"{'='*60}\n\n"

            response += f"**Model:** {model_type.upper()}\n"
            response += f"**Trials:** {result.get('completed_trials', n_trials)}\n"
            response += f"**Best {result.get('metric_name', 'score')}:** {result.get('best_metric', 0):.4f}\n\n"

            response += "**Best Parameters:**\n"
            for k, v in result.get('best_params', {}).items():
                response += f"  - {k}: {v}\n"

            if result.get('visualization'):
                response += f"\n**Visualization:** `{result['visualization']}`\n"

            if result.get('saved_model'):
                response += f"**Saved Model:** `{result['saved_model']}`\n"

            return response

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Tuning failed: {str(e)}"

    # ============================================
    # PAPER SCORING HANDLERS
    # ============================================

    def _is_paper_score_request(self, user_lower: str) -> bool:
        """Check if this is a paper scoring request"""
        score_keywords = [
            'score paper', 'rate paper', 'score this paper',
            'how innovative', 'innovation score', 'rate the paper',
            'evaluate paper', 'paper rating'
        ]
        return any(kw in user_lower for kw in score_keywords)

    def _handle_paper_score(self, user_input: str) -> str:
        """Handle paper scoring requests"""
        # Extract arxiv ID
        arxiv_match = re.search(r'\d{4}\.\d{4,5}(?:v\d+)?', user_input)
        if not arxiv_match:
            return "Please provide an ArXiv ID. Example: 'Score paper 2301.00001'"

        arxiv_id = arxiv_match.group(0)

        try:
            score_info = self.arxiv.score_paper(arxiv_id)

            if "error" in score_info:
                return f"Scoring failed: {score_info['error']}"

            return self.arxiv._format_score(score_info)

        except Exception as e:
            return f"Scoring failed: {str(e)}"

    def _handle_arxiv_search_scored(self, query: str) -> str:
        """Search papers with innovation scores"""
        papers = self.arxiv.search_and_score(query, max_results=5)

        if not papers or (isinstance(papers, list) and papers and 'error' in papers[0]):
            return "No papers found. Try rephrasing your query."

        self.last_arxiv_results = papers
        response = "Found papers (sorted by innovation score):\n\n"

        for i, paper in enumerate(papers, 1):
            quick_score = paper.get('quick_score', {})
            score = quick_score.get('overall_score', '?')
            reason = quick_score.get('reason', '')

            score_bar = "█" * int(score) + "░" * (10 - int(score)) if isinstance(score, (int, float)) else "?"

            response += f"{i}. **{paper['title']}**\n"
            response += f"   Innovation: {score}/10 [{score_bar}]\n"
            if reason:
                response += f"   *{reason}*\n"
            response += f"   Authors: {', '.join(paper['authors'][:3])}\n"
            response += f"   ID: `{paper['arxiv_id']}`\n\n"

        response += "Say 'download 1' to get full analysis with detailed scoring."
        return response

    # ============================================
    # KAGGLE ENHANCED HANDLERS
    # ============================================

    def _handle_kaggle_popular(self) -> str:
        """Get popular Kaggle datasets"""
        datasets = self.kaggle.get_popular_datasets(max_results=10)

        if not datasets or (isinstance(datasets, list) and datasets and 'error' in datasets[0]):
            return "Failed to get popular datasets."

        self.last_kaggle_results = datasets
        response = "Most Popular Kaggle Datasets:\n\n"

        for i, ds in enumerate(datasets, 1):
            response += f"{i}. **{ds['title']}**\n"
            response += f"   Ref: `{ds['ref']}`\n"
            response += f"   Downloads: {ds['download_count']:,}\n\n"

        response += "Say 'download 1' to download a dataset."
        return response

    def _handle_kaggle_recommend(self, user_input: str) -> str:
        """Get dataset recommendations by task type"""
        user_lower = user_input.lower()

        # Detect task type
        task_type = 'classification'  # default
        if 'regression' in user_lower:
            task_type = 'regression'
        elif 'nlp' in user_lower or 'text' in user_lower:
            task_type = 'nlp'
        elif 'image' in user_lower or 'vision' in user_lower:
            task_type = 'image'
        elif 'time' in user_lower or 'series' in user_lower:
            task_type = 'timeseries'
        elif 'cluster' in user_lower:
            task_type = 'clustering'

        datasets = self.kaggle.recommend_dataset(task_type)

        if not datasets or (isinstance(datasets, list) and datasets and 'error' in datasets[0]):
            return f"Failed to get recommendations for {task_type}."

        self.last_kaggle_results = datasets
        response = f"Recommended {task_type.upper()} Datasets:\n\n"

        for i, ds in enumerate(datasets, 1):
            response += f"{i}. **{ds['title']}**\n"
            response += f"   Ref: `{ds['ref']}`\n"
            response += f"   Downloads: {ds['download_count']:,}\n\n"

        response += "Say 'download 1' to download a dataset."
        return response

    # ============================================
    # TEXT MODEL FINE-TUNING HANDLERS
    # ============================================

    def _is_finetune_text_request(self, user_lower: str) -> bool:
        """Check if this is a text model fine-tuning request"""
        # Exclude image finetuning
        if 'image' in user_lower:
            return False

        # Check for finetune + model path or model name
        has_finetune = any(kw in user_lower for kw in ['finetune', 'fine-tune', 'fine tune'])
        has_model_indicator = any(kw in user_lower for kw in [
            'bert', 'gpt', 'roberta', 't5', 'llm', 'model', 'hf_cache', 'models--'
        ])
        has_dataset = any(kw in user_lower for kw in ['.csv', 'on c:', 'on /', 'dataset'])

        # If we have finetune + model + dataset path, it's a text finetune
        if has_finetune and has_model_indicator and has_dataset:
            return True

        # Original keywords
        finetune_keywords = [
            'finetune text', 'fine-tune text', 'finetune model on',
            'train bert', 'train gpt', 'finetune bert', 'finetune gpt',
            'finetune roberta', 'fine-tune on csv', 'finetune on dataset',
            'finetune llm', 'train llm on'
        ]
        return any(kw in user_lower for kw in finetune_keywords)

    def _is_finetune_kaggle_request(self, user_lower: str) -> bool:
        """Check if this is a Kaggle + HuggingFace fine-tuning request"""
        # Only trigger Kaggle finetune if explicitly mentioning Kaggle dataset ref
        # and NOT providing a full local path

        # If user provides full CSV path, it's NOT a Kaggle request
        if '.csv' in user_lower and ('c:\\' in user_lower or 'c:/' in user_lower or user_lower.startswith('/')):
            return False

        patterns = [
            'finetune' in user_lower and 'kaggle' in user_lower,
            'train on kaggle' in user_lower,
            'finetune with kaggle' in user_lower,
        ]
        return any(patterns)

    def _handle_finetune_text(self, user_input: str) -> str:
        """Handle text model fine-tuning requests"""
        # Extract model ID (check for cache path first)
        model_id = self._extract_hf_model_id(user_input)

        if not model_id:
            model_id = "distilbert-base-uncased"  # Default
            print(f"[FineTune] Using default model: {model_id}")
        else:
            print(f"[FineTune] Detected model: {model_id}")

        # Extract dataset path
        dataset_path = self._extract_file_path(user_input)
        if not dataset_path:
            dataset_path = self.active_dataset

        if not dataset_path:
            return ("No dataset specified. Please either:\n"
                   "1. Provide a file path: 'finetune bert on C:\\path\\to\\data.csv'\n"
                   "2. Download a Kaggle dataset first\n"
                   "3. Say 'finetune bert with kaggle kazanova/sentiment140'")

        if not os.path.exists(dataset_path):
            return f"Dataset not found: `{dataset_path}`"

        print(f"[FineTune] Dataset path: {dataset_path}")

        # Extract parameters
        user_lower = user_input.lower()

        epochs = 3
        epoch_match = re.search(r'(\d+)\s*epoch', user_lower)
        if epoch_match:
            epochs = int(epoch_match.group(1))

        batch_size = 8
        batch_match = re.search(r'batch\s*(?:size)?\s*(\d+)', user_lower)
        if batch_match:
            batch_size = int(batch_match.group(1))

        # Extract column names if specified
        text_col = None
        label_col = None

        text_match = re.search(r'text\s*(?:column)?\s*[=:]\s*["\']?(\w+)["\']?', user_lower)
        if text_match:
            text_col = text_match.group(1)

        label_match = re.search(r'label\s*(?:column)?\s*[=:]\s*["\']?(\w+)["\']?', user_lower)
        if label_match:
            label_col = label_match.group(1)

        try:
            result = self.huggingface.finetune_model(
                model_id=model_id,
                dataset_path=dataset_path,
                text_column=text_col,
                label_column=label_col,
                epochs=epochs,
                batch_size=batch_size
            )

            if "error" in result:
                return f"Fine-tuning Error: {result['error']}"

            return self._format_finetune_text_results(result)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Fine-tuning failed: {str(e)}"

    def _handle_finetune_kaggle(self, user_input: str) -> str:
        """Handle Kaggle + HuggingFace fine-tuning"""
        # Extract model ID
        model_id = self._extract_hf_model_id(user_input)

        if not model_id:
            model_id = "distilbert-base-uncased"
            print(f"[FineTune] Using default model: {model_id}")

        # Extract Kaggle dataset ref
        kaggle_ref = None
        kaggle_match = re.search(r'[\w-]+/[\w-]+', user_input)
        if kaggle_match:
            potential_ref = kaggle_match.group(0)
            # Make sure it's not the HF model
            if potential_ref != model_id:
                kaggle_ref = potential_ref

        if not kaggle_ref:
            # Check recent search results
            if self.last_kaggle_results:
                kaggle_ref = self.last_kaggle_results[0]['ref']
                print(f"[FineTune] Using last searched dataset: {kaggle_ref}")
            else:
                return ("Please specify a Kaggle dataset. Example:\n"
                       "'finetune bert with kaggle kazanova/sentiment140'")

        # Extract parameters
        user_lower = user_input.lower()

        epochs = 3
        epoch_match = re.search(r'(\d+)\s*epoch', user_lower)
        if epoch_match:
            epochs = int(epoch_match.group(1))

        try:
            result = self.huggingface.finetune_with_kaggle(
                model_id=model_id,
                kaggle_dataset_ref=kaggle_ref,
                kaggle_agent=self.kaggle,
                epochs=epochs
            )

            if "error" in result:
                return f"Fine-tuning Error: {result['error']}"

            return self._format_finetune_text_results(result)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Fine-tuning failed: {str(e)}"

    def _extract_hf_model_id(self, user_input: str) -> Optional[str]:
        """Extract HuggingFace model ID from user input"""
        # First check for local cache path (models--org--name format)
        # This handles paths like: hf_cache\models--nlpaueb--legal-bert-small-uncased
        cache_path_match = re.search(r'models--([a-zA-Z0-9_-]+)--([a-zA-Z0-9_-]+)', user_input)
        if cache_path_match:
            org = cache_path_match.group(1)
            model = cache_path_match.group(2)
            return f"{org}/{model}"

        # Check for org/model format (e.g., nlpaueb/legal-bert-small-uncased)
        # Use word boundary or space/punctuation to stop matching
        org_model_match = re.search(r'([a-zA-Z0-9_-]+)/([a-zA-Z0-9_-]+)(?=\s|$|[,;:\.])', user_input)
        if org_model_match:
            # Make sure it's not a file path component
            potential_id = f"{org_model_match.group(1)}/{org_model_match.group(2)}"
            excluded = ['users', 'kaggle', 'datasets', 'path', 'finetuned', 'c:', 'd:', 'e:']
            if not any(x in potential_id.lower() for x in excluded):
                return potential_id

        # Common model patterns (without org)
        model_patterns = [
            r'\b(distilbert-base-uncased)\b',
            r'\b(distilbert-base-cased)\b',
            r'\b(bert-base-uncased)\b',
            r'\b(bert-base-cased)\b',
            r'\b(bert-large-uncased)\b',
            r'\b(roberta-base)\b',
            r'\b(roberta-large)\b',
            r'\b(gpt2)\b',
            r'\b(gpt2-medium)\b',
            r'\b(gpt2-large)\b',
            r'\b(t5-small)\b',
            r'\b(t5-base)\b',
        ]

        user_lower = user_input.lower()

        for pattern in model_patterns:
            match = re.search(pattern, user_lower)
            if match:
                return match.group(1)

        # Generic patterns as fallback
        generic_patterns = [
            r'\b(distilbert[\w-]*)\b',
            r'\b(bert[\w-]*)\b',
            r'\b(roberta[\w-]*)\b',
        ]

        for pattern in generic_patterns:
            match = re.search(pattern, user_lower)
            if match:
                return match.group(1)

        return None

    def _format_finetune_text_results(self, result: dict) -> str:
        """Format text fine-tuning results"""
        msg = "\n" + "=" * 60 + "\n"
        msg += "FINE-TUNING COMPLETE\n"
        msg += "=" * 60 + "\n\n"

        msg += f"Base Model: {result.get('base_model', 'N/A')}\n"
        msg += f"Task Type: {result.get('task_type', 'N/A')}\n"
        msg += f"Training Samples: {result.get('train_samples', 0)}\n"
        msg += f"Validation Samples: {result.get('val_samples', 0)}\n"

        metrics = result.get('metrics', {})
        if metrics:
            msg += f"\nFinal Metrics:\n"
            for k, v in metrics.items():
                if isinstance(v, float):
                    msg += f"  {k}: {v:.4f}\n"
                else:
                    msg += f"  {k}: {v}\n"

        if result.get('model_path'):
            msg += f"\nModel saved to: `{result['model_path']}`\n"

        msg += "\n" + "=" * 60 + "\n"
        msg += "Use 'run inference with <model_path>: <text>' to test the model"

        return msg

    def _handle_finetuned_inference(self, user_input: str) -> str:
        """Handle inference with fine-tuned models"""
        # Extract model path
        model_path = None
        path_match = re.search(r'finetuned_models[\\\/][\w_-]+[\\\/]final', user_input)
        if path_match:
            model_path = path_match.group(0)

        if not model_path:
            # List available finetuned models
            models = self.huggingface.list_finetuned_models()
            if models:
                model_path = models[-1]['path']  # Use most recent
                print(f"Using most recent finetuned model: {model_path}")
            else:
                return "No finetuned models found. Finetune a model first!"

        # Extract text to classify
        text = None
        if ':' in user_input:
            text = user_input.split(':', 1)[1].strip()

        if not text:
            return "Please provide text to classify. Example: 'classify with model: This is great!'"

        try:
            result = self.huggingface.run_finetuned_inference(model_path, text)

            if "error" in result:
                return f"Inference Error: {result['error']}"

            response = f"**Input:** {text}\n\n"

            if result.get('prediction'):
                pred = result['prediction']
                response += f"**Prediction:** {pred.get('label', 'N/A')}\n"
                response += f"**Confidence:** {pred.get('score', 0):.2%}\n"
            elif result.get('generated'):
                response += f"**Generated:** {result['generated']}\n"

            return response

        except Exception as e:
            return f"Inference failed: {str(e)}"
