import os
import shutil
from pathlib import Path

try:
    from huggingface_hub import HfApi, snapshot_download, login
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("[HuggingFace] huggingface_hub not available")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("[HuggingFace] transformers not available")

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("[HuggingFace] datasets not available")


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
