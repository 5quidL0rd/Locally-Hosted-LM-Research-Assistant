import os
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

try:
    from huggingface_hub import HfApi, snapshot_download, login
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("[HuggingFace] huggingface_hub not available")

try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, pipeline,
        AutoImageProcessor, AutoModelForImageClassification,
        TrainingArguments, Trainer
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("[HuggingFace] transformers not available")

try:
    from datasets import load_dataset, Dataset, DatasetDict
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("[HuggingFace] datasets not available")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[HuggingFace] torch not available")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import numpy as np
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


class HuggingFaceAgent:
    """
    Agent for interacting with HuggingFace Hub (robust version)
    """

    def __init__(self, memory_palace, api_token=None, cache_dir="hf_cache"):
        self.memory = memory_palace
        self.cache_dir = cache_dir
        self.loaded_models = {}
        self.last_download_error = None  # Set when a download fails with an explanatory message

        os.makedirs(cache_dir, exist_ok=True)

        if HF_HUB_AVAILABLE:
            self.api = HfApi(token=api_token)
            self.api_token = api_token
            if api_token:
                try:
                    login(token=api_token, add_to_git_credential=False)
                    print("[HuggingFace] Authenticated")
                except Exception as e:
                    print(f"[HuggingFace] Login failed: {e}")
            else:
                print("[HuggingFace] No token provided (rate limits apply)")
        else:
            self.api = None
            print("[HuggingFace] Hub unavailable")

    # ======================================================
    # MODEL SEARCH
    # ======================================================

    def search_models(self, query, task=None, limit=5):
        if not self.api:
            return [{"error": "HuggingFace Hub not available"}]

        try:
            print(f"[HuggingFace] Searching models (broad): '{query}'")
            print(f"[DEBUG] API call parameters: search={query}, sort='downloads', direction=-1, limit={limit * 3}")
            # token presence (do NOT print raw token)
            print(f"[DEBUG] API token present: {'yes' if getattr(self, 'api_token', None) else 'no'}")
            print(f"[DEBUG] Query being passed: {query}")

            # Preprocess the query to remove command words and extract keywords
            import re
            query_lower = query.lower()
            # remove common command/agent words
            clean_query = re.sub(r"\b(search|find|look for|look|for|huggingface|hugging face|hf|models|model|the|please)\b", "", query_lower)
            # remove punctuation and extra whitespace
            clean_query = re.sub(r"[^a-z0-9\s]", " ", clean_query)
            clean_query = re.sub(r"\s+", " ", clean_query).strip()
            # fallback: if nothing remains, try to pick last word or use original query
            if not clean_query:
                tokens = query_lower.split()
                if tokens:
                    clean_query = tokens[-1]
                else:
                    clean_query = query_lower

            print(f"[DEBUG] Cleaned query: {clean_query}")

            models = list(self.api.list_models(
                search=clean_query,
                sort="downloads",
                direction=-1,
                limit=limit * 3  # oversample, prune later
            ))

            print(f"[DEBUG] Raw models found: {len(models)}")
            print(f"[DEBUG] Query: {query}, Task: {task}")
            print(f"[DEBUG] Raw API response: {models}")

            if task:
                models = [
                    m for m in models
                    if getattr(m, "pipeline_tag", None) == task
                ]
                print(f"[DEBUG] After task filter '{task}': {len(models)}")

            results = []
            for model in models[:limit]:
                info = {
                    "id": model.id,
                    "author": getattr(model, "author", "Unknown"),
                    "downloads": getattr(model, "downloads", 0),
                    "likes": getattr(model, "likes", 0),
                    "pipeline_tag": getattr(model, "pipeline_tag", None),
                    "tags": getattr(model, "tags", []),
                }
                results.append(info)

                node_id = f"hf_model_{model.id.replace('/', '_')}"
                self.memory.add_node(node_id, "huggingface_model", info)
                self.memory.add_edge(query, node_id, "search_result")

            if not results:
                return []

            return results

        except Exception as e:
            return [{"error": f"Model search failed: {e}"}]

    # ======================================================
    # MODEL DOWNLOAD / LOAD
    # ======================================================

    def get_model_size(self, model_id):
        """Return approximate total size in bytes of model repo files, or None if unknown."""
        try:
            info = self.api.model_info(model_id)
            total = 0
            siblings = getattr(info, 'siblings', None)
            if siblings:
                for s in siblings:
                    size = getattr(s, 'size', None)
                    if size:
                        total += size
                return total if total > 0 else None
            return None
        except Exception:
            return None

    def get_model_info(self, model_id):
        """Return basic model info (pipeline_tag, library_name, tags) or an error string."""
        if not HF_HUB_AVAILABLE or not getattr(self, 'api', None):
            return "HuggingFace Hub not available"

        try:
            info = self.api.model_info(model_id)
            return {
                'id': getattr(info, 'id', model_id),
                'pipeline_tag': getattr(info, 'pipeline_tag', None),
                'library_name': getattr(info, 'library_name', None),
                'tags': getattr(info, 'tags', []),
            }
        except Exception as e:
            return f"Failed to get model info: {e}"

    def download_model(self, model_id):
        """
        Download a model from HuggingFace Hub.
        Performs a pre-check of available disk space (when possible) and stores last error message
        in `self.last_download_error` for clearer diagnostics. Returns the local path on success,
        or `None` on failure (see `get_last_download_error()` for details).
        """
        if not HF_HUB_AVAILABLE:
            self.last_download_error = "huggingface_hub not installed"
            print("[HuggingFace] Cannot download - hub not available")
            return None

        # Ensure cache dir exists
        os.makedirs(self.cache_dir, exist_ok=True)

        # Pre-download size check when available
        model_size = None
        try:
            model_size = self.get_model_size(model_id)
            if model_size:
                free = shutil.disk_usage(self.cache_dir).free
                # Add a small safety margin: 15% or +200MB, whichever is larger
                required = max(int(model_size * 1.15), model_size + 200 * 1024 * 1024)
                print(f"[HuggingFace] Model size estimate: {model_size/(1024**3):.2f} GB; free: {free/(1024**3):.2f} GB; required ~ {required/(1024**3):.2f} GB")
                if free < required:
                    msg = (
                        f"Insufficient disk space to download {model_id}. "
                        f"Estimated model size: {model_size/(1024**3):.2f} GB. "
                        f"Free space in cache ({os.path.abspath(self.cache_dir)}): {free/(1024**3):.2f} GB. "
                        f"Need roughly {required/(1024**3):.2f} GB. "
                        "Consider freeing space or setting `cache_dir` to a drive with more space."
                    )
                    print(f"[HuggingFace] Download aborted: {msg}")
                    self.last_download_error = msg
                    return None
        except Exception as e:
            print(f"[HuggingFace] Could not determine model size: {e} (continuing with download)")

        try:
            print(f"[HuggingFace] Downloading: {model_id}")
            path = snapshot_download(
                repo_id=model_id,
                cache_dir=self.cache_dir,
                resume_download=True
            )

            self.memory.add_node(
                f"hf_downloaded_{model_id.replace('/', '_')}",
                "downloaded_model",
                {"model_id": model_id, "path": path}
            )

            self.last_download_error = None
            return path

        except OSError as e:
            # Windows symlink privilege error (WinError 1314) or similar
            if (hasattr(e, 'winerror') and e.winerror == 1314) or 'required privilege' in str(e).lower():
                msg = (
                    "Download failed due to insufficient privileges for creating symlinks on Windows (WinError 1314).\n"
                    "Options:\n"
                    "  - Enable Developer Mode in Windows Settings -> For Developers, or run Python as Administrator.\n"
                    "  - Set env var `HF_HUB_DISABLE_SYMLINKS_WARNING=1` to silence warnings (does not grant privileges).\n"
                    "  - Try a smaller model or use a different machine.\n"
                    "See: https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations"
                )
                print(f"[HuggingFace] Download failed: {msg}")
                self.last_download_error = msg
                return None
            else:
                msg = f"Download failed: {e}"
                print(f"[HuggingFace] {msg}")
                self.last_download_error = msg
                return None
        except Exception as e:
            msg = f"Download failed: {e}"
            print(f"[HuggingFace] {msg}")
            self.last_download_error = msg
            return None

    def load_model(self, model_id, load_in_8bit=False):
        """Load a model into memory for inference with graceful fallbacks.

        Returns (model, tokenizer) on success, or (None, error_message) on failure.
        """
        # Lazy import check: try to import transformers if not available
        global TRANSFORMERS_AVAILABLE
        if not TRANSFORMERS_AVAILABLE:
            try:
                # Use importlib to avoid creating local variables that shadow module-level names
                import importlib
                transformers_mod = importlib.import_module("transformers")
                globals()['AutoTokenizer'] = transformers_mod.AutoTokenizer
                globals()['AutoModelForCausalLM'] = transformers_mod.AutoModelForCausalLM
                globals()['pipeline'] = transformers_mod.pipeline
                TRANSFORMERS_AVAILABLE = True
            except Exception:
                return None, "transformers not installed - run: pip install transformers torch"

        # Avoid reloading if already present
        if model_id in self.loaded_models:
            return self.loaded_models[model_id]

        final_model_id = model_id

        # Helper to attempt tokenizer/model load from a given source (id or local path)
        def _try_load_from(source_id, use_local=False):
            try:
                tok = AutoTokenizer.from_pretrained(
                    source_id,
                    cache_dir=self.cache_dir,
                    use_fast=True,
                    local_files_only=use_local
                )
            except Exception as et:
                raise RuntimeError(f"Tokenizer load failed for '{source_id}': {et}")

            try:
                mdl = AutoModelForCausalLM.from_pretrained(
                    source_id,
                    cache_dir=self.cache_dir,
                    load_in_8bit=load_in_8bit,
                    device_map="auto" if load_in_8bit else None
                )
            except Exception as em:
                # Try CPU-only fallback
                try:
                    mdl = AutoModelForCausalLM.from_pretrained(
                        source_id,
                        cache_dir=self.cache_dir,
                        device_map="cpu",
                        low_cpu_mem_usage=True,
                        local_files_only=use_local
                    )
                except Exception as e_cpu:
                    raise RuntimeError(f"Model load failed for '{source_id}': {em} | CPU fallback: {e_cpu}")

            return mdl, tok

        # Try standard remote/local load first; if that fails, try downloading then try local load
        try:
            model, tokenizer = _try_load_from(final_model_id, use_local=False)
        except Exception as e1:
            print(f"[HuggingFace] Initial load failed for '{model_id}': {e1}")

            # Attempt to download the repository to the cache and load locally
            try:
                downloaded = self.download_model(model_id)
                if downloaded:
                    try:
                        model, tokenizer = _try_load_from(downloaded, use_local=True)
                        final_model_id = downloaded
                    except Exception as e_local:
                        print(f"[HuggingFace] Local load after download failed: {e_local}")
                        model = tokenizer = None
                else:
                    model = tokenizer = None
            except Exception as e_dl:
                print(f"[HuggingFace] download attempt failed: {e_dl}")
                model = tokenizer = None

        # Final fallback to ultra-small testing model
        if model is None or tokenizer is None:
            fallback = "hf-internal-testing/tiny-random-gpt2"
            try:
                print(f"[HuggingFace] Falling back to tiny model '{fallback}' for testing")
                model, tokenizer = _try_load_from(fallback, use_local=False)
                final_model_id = fallback
            except Exception as e_fb:
                return None, f"All attempts to load model '{model_id}' failed. Last error: {e_fb}"

        # Cache and return
        self.loaded_models[model_id] = (model, tokenizer)
        # Also cache by the final loaded id for convenience
        if final_model_id != model_id:
            self.loaded_models[final_model_id] = (model, tokenizer)

        print(f"[HuggingFace] Loaded {final_model_id} (requested: {model_id})")
        return model, tokenizer

    def run_inference(self, model_id, prompt, max_length=100, temperature=0.7):
        if model_id not in self.loaded_models:
            model, err = self.load_model(model_id)
            if model is None:
                return f"Load failed: {err}"

        model, tokenizer = self.loaded_models[model_id]

        # Check that the loaded model supports text generation
        if not hasattr(model, 'generate'):
            # Try an on-the-fly pipeline fallback if transformers is available
            if TRANSFORMERS_AVAILABLE:
                try:
                    print(f"[HuggingFace] Model '{model_id}' lacks 'generate'; attempting pipeline fallback")
                    gen_pipe = pipeline("text-generation", model=model_id, device=-1)
                    out = gen_pipe(prompt, max_length=max_length, do_sample=temperature > 0, temperature=temperature)
                    # pipeline returns a list of dicts with 'generated_text'
                    text = out[0].get('generated_text') if isinstance(out, list) and len(out) and isinstance(out[0], dict) else str(out)
                    return text
                except Exception as e_pipe:
                    print(f"[HuggingFace] Pipeline fallback failed: {e_pipe}")

            # Try to get pipeline tag for better messaging
            pipeline_tag = None
            try:
                info = self.get_model_info(model_id)
                if isinstance(info, dict):
                    pipeline_tag = info.get('pipeline_tag')
            except Exception:
                pipeline_tag = None

            tag_msg = f" (pipeline: {pipeline_tag})" if pipeline_tag else ""
            return (f"Model '{model_id}' does not appear to support text-generation{tag_msg}. "
                    "Use a text-generation model such as 'gpt2' or run an appropriate pipeline, e.g. "
                    f"`pipeline('<task>', model='{model_id}')` for its task.")
        if tokenizer is None:
            return "No tokenizer available for this model; cannot run text generation."

        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id
        )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def run_time_series(self, model_id, context, prediction_length, num_samples=10, device=-1, verbose=True):
        """Run a time-series forecasting pipeline and optionally print shapes and memory.

        Parameters:
          - model_id: HF model id or local path
          - context: array-like (num_series, context_length) or (context_length,) for single series
          - prediction_length: int, how many future steps to predict
          - num_samples: number of stochastic trajectories to sample (controls memory)
          - device: -1 for CPU, integer for GPU device index
          - verbose: when True prints shapes and memory usage

        Returns: dict with keys 'forecast' (np.ndarray) and 'meta' (info dict) or {'error': str}
        """
        # Local imports to keep module lightweight when not using time-series
        try:
            import numpy as np
        except Exception:
            return {"error": "numpy not installed - pip install numpy"}

        ps_mem = None
        try:
            import psutil
            ps_mem = psutil.Process()
        except Exception:
            ps_mem = None

        def _mem_str():
            if ps_mem is None:
                return "(psutil not available)"
            return f"{ps_mem.memory_info().rss // 1024**2} MB"

        # Normalize context to numpy array of shape (num_series, context_len)
        if isinstance(context, list):
            context = np.array(context, dtype=np.float32)
        elif hasattr(context, 'values'):
            # pandas DataFrame/Series
            context = np.asarray(context.values, dtype=np.float32)
        elif isinstance(context, np.ndarray):
            context = context.astype(np.float32)
        else:
            try:
                context = np.array(context, dtype=np.float32)
            except Exception as e:
                return {"error": f"Could not convert context to numpy array: {e}"}

        if context.ndim == 1:
            context = context.reshape(1, -1)

        if verbose:
            print(f"[HuggingFace][TimeSeries] Memory before load: {_mem_str()}")
            print(f"[HuggingFace][TimeSeries] Context shape: {context.shape}, dtype={context.dtype}")
            print(f"[HuggingFace][TimeSeries] prediction_length={prediction_length}, num_samples={num_samples}, device={device}")

        # Create pipeline
        try:
            pipe = pipeline("time-series-forecasting", model=model_id, trust_remote_code=True, device=device)
        except Exception as e:
            return {"error": f"Failed to construct time-series pipeline for '{model_id}': {e}"}

        # Call predict (Chronos pipelines expose predict)
        forecast = None
        try:
            if hasattr(pipe, 'predict'):
                forecast = pipe.predict(context, prediction_length, num_samples=num_samples)
            else:
                # Some pipelines may implement __call__ or return arrays directly
                out = pipe(context, prediction_length, num_samples=num_samples)
                # Try to find a plausible forecast array
                if isinstance(out, dict) and 'forecast' in out:
                    forecast = out['forecast']
                else:
                    forecast = out
        except Exception as e:
            return {"error": f"Inference failed on time-series pipeline: {e}"}

        try:
            forecast = np.asarray(forecast)
        except Exception as e:
            return {"error": f"Could not convert forecast to numpy array: {e}", "raw": forecast}

        meta = {
            "context_shape": context.shape,
            "forecast_shape": forecast.shape,
            "memory_after": _mem_str()
        }

        if verbose:
            print(f"[HuggingFace][TimeSeries] Forecast shape: {forecast.shape}")
            # Show a brief numeric summary
            try:
                import numpy as _np
                print(f"[HuggingFace][TimeSeries] Forecast summary: mean={_np.mean(forecast):.4f}, std={_np.std(forecast):.4f}")
            except Exception:
                pass
            print(f"[HuggingFace][TimeSeries] Memory after predict: {meta['memory_after']}")

        return {"forecast": forecast, "meta": meta}

    # ======================================================
    # DATASETS
    # ======================================================

    def search_datasets(self, query, limit=5):
        if not self.api:
            return [{"error": "HuggingFace Hub not available"}]

        try:
            print(f"[HuggingFace] Searching datasets: '{query}'")

            datasets = list(self.api.list_datasets(
                search=query,
                limit=limit * 3
            ))

            print(f"[DEBUG] Raw datasets found: {len(datasets)}")

            results = []
            for ds in datasets[:limit]:
                info = {
                    "id": ds.id,
                    "author": getattr(ds, "author", "Unknown"),
                    "downloads": getattr(ds, "downloads", 0),
                    "likes": getattr(ds, "likes", 0),
                    "tags": getattr(ds, "tags", [])
                }
                results.append(info)

                node_id = f"hf_dataset_{ds.id.replace('/', '_')}"
                self.memory.add_node(node_id, "huggingface_dataset", info)
                self.memory.add_edge(query, node_id, "search_result")

            if not results:
                return [{"warning": "No datasets matched"}]

            return results

        except Exception as e:
            return [{"error": f"Dataset search failed: {e}"}]

    def load_dataset_sample(self, dataset_id, split="train", num_samples=5):
        if not DATASETS_AVAILABLE:
            return "datasets not installed"

        dataset = load_dataset(
            dataset_id,
            split=split,
            cache_dir=self.cache_dir,
            trust_remote_code=True
        )

        sample = dataset.select(range(min(num_samples, len(dataset))))

        return {
            "dataset_id": dataset_id,
            "split": split,
            "size": len(dataset),
            "features": list(dataset.features.keys()),
            "samples": [dict(x) for x in sample]
        }

    # ======================================================
    # UTIL
    # ======================================================

    def list_loaded_models(self):
        return list(self.loaded_models.keys()) or "No models loaded"

    def unload_model(self, model_id):
        if model_id in self.loaded_models:
            del self.loaded_models[model_id]
            print(f"[HuggingFace] Unloaded {model_id}")
            return True
        return False

    def get_last_download_error(self):
        """Return the last download error message (or None if last download succeeded)."""
        return getattr(self, 'last_download_error', None)

    # ======================================================
    # IMAGE CLASSIFICATION FINETUNING
    # ======================================================

    def finetune_image_classifier(
        self,
        image_folder: str,
        base_model: str = "google/vit-base-patch16-224",
        output_dir: str = "finetuned_models",
        epochs: int = 5,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        Finetune an image classification model on a folder of images.

        Expected folder structure:
            image_folder/
                class1/
                    img1.jpg
                    img2.jpg
                class2/
                    img3.jpg
                    ...

        Args:
            image_folder: Path to folder with class subfolders
            base_model: HuggingFace model to finetune (default: ViT)
            output_dir: Where to save the finetuned model
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            validation_split: Fraction for validation

        Returns:
            Dict with training results and model path
        """
        if not TRANSFORMERS_AVAILABLE:
            return {"error": "transformers not installed. Run: pip install transformers"}
        if not TORCH_AVAILABLE:
            return {"error": "torch not installed. Run: pip install torch"}
        if not PIL_AVAILABLE:
            return {"error": "PIL not installed. Run: pip install pillow"}

        print(f"\n{'='*60}")
        print("IMAGE CLASSIFIER FINETUNING")
        print(f"{'='*60}")

        image_folder = Path(image_folder)
        if not image_folder.exists():
            return {"error": f"Folder not found: {image_folder}"}

        # Discover classes from subdirectories
        class_dirs = [d for d in image_folder.iterdir() if d.is_dir()]
        if not class_dirs:
            # Check if images are directly in folder (no class structure)
            return {"error": "Expected folder structure with class subdirectories (e.g., cats/, dogs/)"}

        class_names = sorted([d.name for d in class_dirs])
        label2id = {name: i for i, name in enumerate(class_names)}
        id2label = {i: name for name, i in label2id.items()}

        print(f"Found {len(class_names)} classes: {class_names}")

        # Load images and labels
        image_paths = []
        labels = []

        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')
        for class_dir in class_dirs:
            class_label = label2id[class_dir.name]
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in image_extensions:
                    image_paths.append(str(img_path))
                    labels.append(class_label)

        if not image_paths:
            return {"error": "No images found in class subdirectories"}

        print(f"Total images: {len(image_paths)}")

        # Create train/val split
        from sklearn.model_selection import train_test_split
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels,
            test_size=validation_split,
            stratify=labels,
            random_state=42
        )

        print(f"Training samples: {len(train_paths)}")
        print(f"Validation samples: {len(val_paths)}")

        # Load image processor and model
        print(f"\nLoading base model: {base_model}")
        try:
            image_processor = AutoImageProcessor.from_pretrained(base_model)
            model = AutoModelForImageClassification.from_pretrained(
                base_model,
                num_labels=len(class_names),
                label2id=label2id,
                id2label=id2label,
                ignore_mismatched_sizes=True
            )
        except Exception as e:
            return {"error": f"Failed to load model: {e}"}

        # Create datasets
        def load_image(path):
            try:
                img = Image.open(path).convert("RGB")
                return img
            except Exception:
                return None

        def create_dataset(paths, lbls):
            data = {"image": [], "label": []}
            for p, l in zip(paths, lbls):
                img = load_image(p)
                if img is not None:
                    data["image"].append(img)
                    data["label"].append(l)
            return Dataset.from_dict(data)

        print("Creating datasets...")
        train_dataset = create_dataset(train_paths, train_labels)
        val_dataset = create_dataset(val_paths, val_labels)

        # Preprocessing function
        def preprocess(examples):
            images = examples["image"]
            inputs = image_processor(images, return_tensors="pt")
            inputs["labels"] = examples["label"]
            return inputs

        # Apply preprocessing
        train_dataset = train_dataset.map(
            lambda x: image_processor(x["image"], return_tensors="pt"),
            batched=True,
            remove_columns=["image"]
        )
        train_dataset = train_dataset.rename_column("label", "labels")

        val_dataset = val_dataset.map(
            lambda x: image_processor(x["image"], return_tensors="pt"),
            batched=True,
            remove_columns=["image"]
        )
        val_dataset = val_dataset.rename_column("label", "labels")

        # Setup training
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_output_dir = Path(output_dir) / f"image_classifier_{timestamp}"
        model_output_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(model_output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            logging_dir=str(model_output_dir / "logs"),
            logging_steps=10,
            remove_unused_columns=False,
        )

        # Metrics
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            accuracy = (predictions == labels).mean()
            return {"accuracy": accuracy}

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        # Train
        print(f"\nStarting training for {epochs} epochs...")
        print(f"{'─'*60}")

        try:
            train_result = trainer.train()
        except Exception as e:
            return {"error": f"Training failed: {e}"}

        # Evaluate
        eval_result = trainer.evaluate()

        # Save model
        trainer.save_model(str(model_output_dir / "final"))
        image_processor.save_pretrained(str(model_output_dir / "final"))

        # Save config
        config = {
            "base_model": base_model,
            "classes": class_names,
            "label2id": label2id,
            "id2label": id2label,
            "train_samples": len(train_paths),
            "val_samples": len(val_paths),
            "epochs": epochs,
            "final_accuracy": eval_result.get("eval_accuracy", 0),
            "created": timestamp
        }

        with open(model_output_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

        # Create visualization
        viz_path = None
        if PLOTTING_AVAILABLE:
            viz_path = self._plot_training_results(
                trainer.state.log_history,
                class_names,
                str(model_output_dir / "training_results.png")
            )

        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Final Accuracy: {eval_result.get('eval_accuracy', 0):.4f}")
        print(f"Model saved to: {model_output_dir / 'final'}")

        # Store in memory
        self.memory.add_node(
            f"finetuned_image_classifier_{timestamp}",
            "finetuned_model",
            config
        )

        return {
            "model_path": str(model_output_dir / "final"),
            "accuracy": eval_result.get("eval_accuracy", 0),
            "classes": class_names,
            "train_samples": len(train_paths),
            "val_samples": len(val_paths),
            "visualization": viz_path,
            "config": config
        }

    def _plot_training_results(self, log_history: list, class_names: list, save_path: str) -> str:
        """Plot training metrics"""
        if not PLOTTING_AVAILABLE:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Image Classifier Training Results', fontsize=14, fontweight='bold')

        # Extract metrics from log history
        train_losses = []
        eval_losses = []
        eval_accuracies = []
        epochs_train = []
        epochs_eval = []

        for entry in log_history:
            if 'loss' in entry and 'epoch' in entry:
                train_losses.append(entry['loss'])
                epochs_train.append(entry['epoch'])
            if 'eval_loss' in entry and 'epoch' in entry:
                eval_losses.append(entry['eval_loss'])
                eval_accuracies.append(entry.get('eval_accuracy', 0))
                epochs_eval.append(entry['epoch'])

        # Plot loss
        ax = axes[0]
        if train_losses:
            ax.plot(epochs_train, train_losses, label='Train Loss', alpha=0.7)
        if eval_losses:
            ax.plot(epochs_eval, eval_losses, label='Val Loss', marker='o')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training & Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot accuracy
        ax = axes[1]
        if eval_accuracies:
            ax.plot(epochs_eval, eval_accuracies, marker='o', color='green', linewidth=2)
            ax.fill_between(epochs_eval, eval_accuracies, alpha=0.3, color='green')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Validation Accuracy')
        ax.grid(True, alpha=0.3)
        if eval_accuracies:
            ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"[HuggingFace] Saved training plot: {save_path}")
        return save_path

    def predict_image(self, model_path: str, image_path: str) -> Dict[str, Any]:
        """
        Predict class for a single image using a finetuned model.

        Args:
            model_path: Path to finetuned model directory
            image_path: Path to image file

        Returns:
            Dict with predicted class and confidence
        """
        if not TRANSFORMERS_AVAILABLE or not PIL_AVAILABLE:
            return {"error": "Required libraries not installed"}

        model_path = Path(model_path)

        # Load config
        config_path = model_path.parent / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            id2label = {int(k): v for k, v in config.get("id2label", {}).items()}
        else:
            id2label = {}

        try:
            processor = AutoImageProcessor.from_pretrained(str(model_path))
            model = AutoModelForImageClassification.from_pretrained(str(model_path))
            model.eval()
        except Exception as e:
            return {"error": f"Failed to load model: {e}"}

        try:
            image = Image.open(image_path).convert("RGB")
            inputs = processor(image, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                pred_idx = probs.argmax().item()
                confidence = probs[0, pred_idx].item()

            pred_label = id2label.get(pred_idx, f"class_{pred_idx}")

            return {
                "predicted_class": pred_label,
                "confidence": confidence,
                "all_probs": {id2label.get(i, f"class_{i}"): p.item() for i, p in enumerate(probs[0])}
            }

        except Exception as e:
            return {"error": f"Prediction failed: {e}"}

    def list_finetuned_models(self, output_dir: str = "finetuned_models") -> List[Dict]:
        """List all finetuned models"""
        output_dir = Path(output_dir)
        if not output_dir.exists():
            return []

        models = []
        for model_dir in output_dir.iterdir():
            if model_dir.is_dir():
                config_path = model_dir / "config.json"
                if config_path.exists():
                    with open(config_path) as f:
                        config = json.load(f)
                    models.append({
                        "name": model_dir.name,
                        "path": str(model_dir / "final"),
                        **config
                    })

        return models

    def iterate_and_log_models(self, query, task=None, limit=5):
        if not self.api:
            print("[HuggingFace] API not initialized.")
            return []

        try:
            print(f"[HuggingFace] Initiating model search for query: '{query}'")
            models = self.api.list_models(search=query, sort="downloads", direction=-1, limit=limit)

            results = []
            for model in models:
                model_details = {
                    "id": model.id,
                    "author": getattr(model, "author", "Unknown"),
                    "downloads": getattr(model, "downloads", 0),
                    "tags": getattr(model, "tags", []),
                }
                print(f"[Model] ID: {model_details['id']}, Author: {model_details['author']}, Downloads: {model_details['downloads']}, Tags: {model_details['tags']}")
                results.append(model_details)

            print(f"[HuggingFace] Total models retrieved: {len(results)}")
            return results

        except Exception as e:
            print(f"[HuggingFace] Error during model iteration: {e}")
            return []

    # ======================================================
    # MODEL FINE-TUNING WITH CUSTOM DATASETS
    # ======================================================

    def finetune_model(
        self,
        model_id: str,
        dataset_path: str,
        task_type: str = "auto",
        output_dir: str = "finetuned_models",
        text_column: str = None,
        label_column: str = None,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        max_length: int = 512,
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        Fine-tune a HuggingFace model on a custom dataset (CSV or folder).

        Supports:
        - Text classification (sentiment, topic classification)
        - Sequence-to-sequence (summarization, translation)
        - Token classification (NER, POS tagging)
        - Causal LM (text generation fine-tuning)

        Args:
            model_id: HuggingFace model ID to fine-tune
            dataset_path: Path to CSV file or directory
            task_type: "text-classification", "seq2seq", "token-classification",
                      "causal-lm", or "auto" to detect from model
            output_dir: Where to save fine-tuned model
            text_column: Column name for input text (auto-detected if None)
            label_column: Column name for labels (auto-detected if None)
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            max_length: Maximum sequence length
            validation_split: Fraction for validation

        Returns:
            Dict with fine-tuning results and model path
        """
        # Check dependencies
        if not TRANSFORMERS_AVAILABLE:
            return {"error": "transformers not installed. Run: pip install transformers"}
        if not TORCH_AVAILABLE:
            return {"error": "torch not installed. Run: pip install torch"}
        if not DATASETS_AVAILABLE:
            return {"error": "datasets not installed. Run: pip install datasets"}

        print(f"\n{'='*60}")
        print("MODEL FINE-TUNING")
        print(f"{'='*60}")
        print(f"Base Model: {model_id}")
        print(f"Dataset: {dataset_path}")

        # Load and prepare dataset
        try:
            dataset_info = self._prepare_finetune_dataset(
                dataset_path, text_column, label_column, validation_split
            )
            if "error" in dataset_info:
                return dataset_info
        except Exception as e:
            return {"error": f"Failed to prepare dataset: {e}"}

        train_dataset = dataset_info["train"]
        val_dataset = dataset_info["val"]
        text_col = dataset_info["text_column"]
        label_col = dataset_info["label_column"]
        num_labels = dataset_info.get("num_labels")
        label2id = dataset_info.get("label2id", {})
        id2label = dataset_info.get("id2label", {})

        print(f"Text column: {text_col}")
        print(f"Label column: {label_col}")
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        if num_labels:
            print(f"Number of classes: {num_labels}")

        # Detect task type from model if auto
        if task_type == "auto":
            task_type = self._detect_task_type(model_id)
            print(f"Detected task type: {task_type}")

        # Load tokenizer and model
        print(f"\nLoading model: {model_id}")
        try:
            from transformers import (
                AutoTokenizer,
                AutoModelForSequenceClassification,
                AutoModelForCausalLM,
                AutoModelForSeq2SeqLM,
                DataCollatorWithPadding,
                TrainingArguments,
                Trainer
            )

            tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=self.cache_dir)

            # Set pad token if needed
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load appropriate model class
            if task_type == "text-classification":
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_id,
                    num_labels=num_labels,
                    label2id=label2id,
                    id2label=id2label,
                    cache_dir=self.cache_dir,
                    ignore_mismatched_sizes=True
                )
            elif task_type == "causal-lm":
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    cache_dir=self.cache_dir
                )
            elif task_type == "seq2seq":
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_id,
                    cache_dir=self.cache_dir
                )
            else:
                return {"error": f"Unsupported task type: {task_type}"}

        except Exception as e:
            return {"error": f"Failed to load model: {e}"}

        # Tokenize datasets
        print("Tokenizing dataset...")

        def tokenize_function(examples):
            result = tokenizer(
                examples[text_col],
                padding="max_length",
                truncation=True,
                max_length=max_length
            )
            if label_col and label_col in examples:
                # Convert string labels to integers if needed
                if label2id:
                    result["labels"] = [label2id.get(l, 0) for l in examples[label_col]]
                else:
                    result["labels"] = examples[label_col]
            return result

        train_tokenized = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        val_tokenized = val_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=val_dataset.column_names
        )

        # Set format
        train_tokenized.set_format("torch")
        val_tokenized.set_format("torch")

        # Setup training arguments
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_output_dir = Path(output_dir) / f"finetuned_{model_id.replace('/', '_')}_{timestamp}"
        model_output_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(model_output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            eval_strategy="epoch",  # renamed from evaluation_strategy in newer transformers
            save_strategy="epoch",
            load_best_model_at_end=True,
            logging_dir=str(model_output_dir / "logs"),
            logging_steps=50,
            warmup_ratio=0.1,
            fp16=TORCH_AVAILABLE and torch.cuda.is_available(),
        )

        # Metrics
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            if task_type == "text-classification":
                predictions = np.argmax(predictions, axis=-1)
                accuracy = (predictions == labels).mean()
                return {"accuracy": accuracy}
            return {}

        # Create trainer
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=val_tokenized,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics if task_type == "text-classification" else None
        )

        # Train
        print(f"\nStarting fine-tuning for {epochs} epochs...")
        print(f"{'─'*60}")

        try:
            train_result = trainer.train()
        except Exception as e:
            return {"error": f"Training failed: {e}"}

        # Evaluate
        eval_result = trainer.evaluate()

        # Save model
        final_path = model_output_dir / "final"
        trainer.save_model(str(final_path))
        tokenizer.save_pretrained(str(final_path))

        # Save config
        config = {
            "base_model": model_id,
            "task_type": task_type,
            "text_column": text_col,
            "label_column": label_col,
            "num_labels": num_labels,
            "label2id": label2id,
            "id2label": id2label,
            "epochs": epochs,
            "final_metrics": eval_result,
            "dataset_path": str(dataset_path),
            "created": timestamp
        }

        with open(model_output_dir / "finetune_config.json", 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\n{'='*60}")
        print("FINE-TUNING COMPLETE")
        print(f"{'='*60}")
        print(f"Final Metrics: {eval_result}")
        print(f"Model saved to: {final_path}")

        # Store in memory
        self.memory.add_node(
            f"finetuned_{timestamp}",
            "finetuned_model",
            config
        )

        return {
            "model_path": str(final_path),
            "base_model": model_id,
            "task_type": task_type,
            "metrics": eval_result,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "config": config
        }

    def _prepare_finetune_dataset(
        self,
        dataset_path: str,
        text_column: Optional[str],
        label_column: Optional[str],
        validation_split: float
    ) -> Dict[str, Any]:
        """Prepare dataset for fine-tuning"""
        import pandas as pd
        from datasets import Dataset

        path = Path(dataset_path)

        if path.is_file() and path.suffix.lower() == '.csv':
            # Load CSV
            df = pd.read_csv(dataset_path)
        elif path.is_dir():
            # Look for CSV files in directory
            csv_files = list(path.glob("*.csv"))
            if not csv_files:
                return {"error": f"No CSV files found in {dataset_path}"}
            df = pd.read_csv(csv_files[0])
            print(f"Using CSV file: {csv_files[0]}")
        else:
            return {"error": f"Invalid dataset path: {dataset_path}"}

        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        # Auto-detect text column
        if text_column is None:
            # Priority candidates for text columns (in order of preference)
            text_candidates = ['text', 'content', 'review', 'comment', 'message',
                              'sentence', 'body', 'description', 'title', 'question',
                              'tweet', 'post', 'article', 'summary', 'abstract']

            for col in text_candidates:
                matching = [c for c in df.columns if col.lower() in c.lower()]
                for m in matching:
                    # Verify it's actually a string column with text data
                    if df[m].dtype == 'object':
                        # Check that it contains actual text (not just short codes)
                        sample = df[m].dropna().head(5)
                        if len(sample) > 0:
                            avg_len = sample.astype(str).str.len().mean()
                            if avg_len > 10:  # At least 10 chars on average
                                text_column = m
                                break
                if text_column:
                    break

            if text_column is None:
                # Fallback: find the string column with longest average text
                best_col = None
                best_avg_len = 0
                for col in df.columns:
                    if df[col].dtype == 'object':
                        sample = df[col].dropna().head(100)
                        if len(sample) > 0:
                            avg_len = sample.astype(str).str.len().mean()
                            if avg_len > best_avg_len:
                                best_avg_len = avg_len
                                best_col = col

                if best_col and best_avg_len > 10:
                    text_column = best_col
                else:
                    return {"error": "Could not auto-detect text column. Please specify text_column."}

        # Auto-detect label column
        label2id = {}
        id2label = {}
        num_labels = None

        if label_column is None:
            label_candidates = ['label', 'target', 'class', 'category', 'sentiment',
                               'rating', 'score', 'y', 'output']
            for col in label_candidates:
                matching = [c for c in df.columns if col.lower() in c.lower() and c != text_column]
                if matching:
                    label_column = matching[0]
                    break

        if label_column and label_column in df.columns:
            # Process labels
            unique_labels = df[label_column].dropna().unique()

            if df[label_column].dtype == 'object':
                # String labels - create mapping
                label2id = {str(l): i for i, l in enumerate(sorted(unique_labels))}
                id2label = {i: str(l) for l, i in label2id.items()}
                num_labels = len(label2id)
            else:
                # Numeric labels
                num_labels = int(df[label_column].max()) + 1
                id2label = {i: str(i) for i in range(num_labels)}
                label2id = {str(i): i for i in range(num_labels)}

        # Validate
        if text_column not in df.columns:
            return {"error": f"Text column '{text_column}' not found in dataset"}

        # Clean data
        df = df.dropna(subset=[text_column])
        if label_column:
            df = df.dropna(subset=[label_column])

        # Split
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(df, test_size=validation_split, random_state=42)

        # Convert to datasets
        train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
        val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))

        return {
            "train": train_dataset,
            "val": val_dataset,
            "text_column": text_column,
            "label_column": label_column,
            "num_labels": num_labels,
            "label2id": label2id,
            "id2label": id2label
        }

    def _detect_task_type(self, model_id: str) -> str:
        """Detect task type from model info"""
        try:
            info = self.api.model_info(model_id)
            pipeline_tag = getattr(info, 'pipeline_tag', None)

            if pipeline_tag:
                if 'classification' in pipeline_tag:
                    return "text-classification"
                elif 'generation' in pipeline_tag or 'causal' in pipeline_tag:
                    return "causal-lm"
                elif 'seq2seq' in pipeline_tag or 'summarization' in pipeline_tag:
                    return "seq2seq"
        except:
            pass

        # Default based on model name
        model_lower = model_id.lower()
        if 'bert' in model_lower or 'roberta' in model_lower:
            return "text-classification"
        elif 'gpt' in model_lower or 'llama' in model_lower:
            return "causal-lm"
        elif 't5' in model_lower or 'bart' in model_lower:
            return "seq2seq"

        return "text-classification"

    def finetune_with_kaggle(
        self,
        model_id: str,
        kaggle_dataset_ref: str,
        kaggle_agent,
        task_type: str = "auto",
        text_column: str = None,
        label_column: str = None,
        epochs: int = 3,
        batch_size: int = 8
    ) -> Dict[str, Any]:
        """
        Convenience method to download a Kaggle dataset and fine-tune a model on it.

        Args:
            model_id: HuggingFace model ID
            kaggle_dataset_ref: Kaggle dataset reference (e.g., "kazanova/sentiment140")
            kaggle_agent: KaggleAgent instance for downloading
            task_type: Task type or "auto"
            text_column: Text column name (auto-detected if None)
            label_column: Label column name (auto-detected if None)
            epochs: Training epochs
            batch_size: Batch size

        Returns:
            Fine-tuning results
        """
        print(f"\n{'='*60}")
        print("KAGGLE DATASET + HUGGINGFACE FINE-TUNING")
        print(f"{'='*60}")
        print(f"Kaggle Dataset: {kaggle_dataset_ref}")
        print(f"HuggingFace Model: {model_id}")

        # Download dataset
        print("\n[1/2] Downloading Kaggle dataset...")
        download_result = kaggle_agent.download_dataset(kaggle_dataset_ref)

        if not download_result or "error" in download_result:
            return {"error": f"Failed to download dataset: {download_result}"}

        dataset_path = download_result.get("path")
        files = download_result.get("files", [])

        # Find CSV file
        csv_files = [f for f in files if f.lower().endswith('.csv')]
        if not csv_files:
            return {"error": f"No CSV files found in Kaggle dataset {kaggle_dataset_ref}"}

        csv_path = csv_files[0]
        print(f"Using CSV: {csv_path}")

        # Fine-tune
        print("\n[2/2] Fine-tuning model...")
        return self.finetune_model(
            model_id=model_id,
            dataset_path=csv_path,
            task_type=task_type,
            text_column=text_column,
            label_column=label_column,
            epochs=epochs,
            batch_size=batch_size
        )

    def run_finetuned_inference(self, model_path: str, text: str) -> Dict[str, Any]:
        """Run inference with a fine-tuned model"""
        if not TRANSFORMERS_AVAILABLE:
            return {"error": "transformers not installed"}

        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

        model_path = Path(model_path)

        # Load config
        config_path = model_path.parent / "finetune_config.json"
        config = {}
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)

        try:
            task_type = config.get("task_type", "text-classification")

            if task_type == "text-classification":
                classifier = pipeline(
                    "text-classification",
                    model=str(model_path),
                    tokenizer=str(model_path)
                )
                result = classifier(text)

                # Map back to original labels if available
                id2label = config.get("id2label", {})
                if result and id2label:
                    label_id = result[0].get("label", "").replace("LABEL_", "")
                    if label_id in id2label:
                        result[0]["label"] = id2label[label_id]

                return {
                    "input": text,
                    "prediction": result[0] if result else None,
                    "task_type": task_type
                }

            elif task_type == "causal-lm":
                generator = pipeline(
                    "text-generation",
                    model=str(model_path),
                    tokenizer=str(model_path)
                )
                result = generator(text, max_length=100, num_return_sequences=1)
                return {
                    "input": text,
                    "generated": result[0]["generated_text"] if result else None,
                    "task_type": task_type
                }

            else:
                return {"error": f"Inference not implemented for task type: {task_type}"}

        except Exception as e:
            return {"error": f"Inference failed: {e}"}
