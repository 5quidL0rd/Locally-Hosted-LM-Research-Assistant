"""
agents/orchestrator.py - AI-driven conversational agent

This orchestrator now uses LM reasoning to decide actions,
instead of brittle keyword matching. Works seamlessly with
LocalLLM, your memory, arXiv, Kaggle, web search, and HuggingFace tools.
"""

import re
import json


# ============================================
# TOOL CONTRACT
# ============================================

AGENT_TOOLS = {
    "arxiv_search": "Search academic papers on arXiv",
    "arxiv_download": "Download and analyze an arXiv paper",
    "kaggle_search": "Search Kaggle datasets",
    "kaggle_download": "Download a Kaggle dataset",
    "kaggle_analyze": "Analyze a downloaded Kaggle dataset and produce plots",
    "hf_model_search": "Search HuggingFace models",
    "hf_model_download": "Download a HuggingFace model",
    "hf_inference": "Run inference with a HuggingFace model",
    "web_search": "Search the web for current information",
    "memory": "Query or visualize long-term memory",
    "chat": "General conversation or reasoning",
    "nn_experiment": "Design, train, and evaluate neural networks on datasets"
}


class ConversationalOrchestrator:
    """
    Interprets natural language and routes to appropriate agents
    using LM reasoning as the primary decision engine.
    """

    def __init__(self, llm, arxiv, kaggle, search, huggingface, memory, nn_builder):
        self.llm = llm
        self.arxiv = arxiv
        self.kaggle = kaggle
        self.search = search
        self.huggingface = huggingface
        self.memory = memory
       
        self.nn_builder = nn_builder
        self.active_dataset = None

        # Track conversation context and last results
        self.last_arxiv_results = []
        self.last_kaggle_results = []
        self.last_hf_model_results = []
        self.last_hf_dataset_results = []
        self.last_search_results = []

    # ============================================
    # LM ROUTING
    # ============================================

    def _llm_route(self, user_input: str) -> dict:
        """
        Ask the LLM which tool to use or to produce a short plan.
        Returns JSON like:
        {"tool":"<tool_name>|plan","query":"...", "id":"...","index":N, "plan":[{step objects}]}
        Each plan step: {"tool":"...","query":"...","id":"...","index":N}
        """
        tool_list = "\n".join(
            f"- {name}: {desc}" for name, desc in AGENT_TOOLS.items()
        )

        prompt = f"""
You are an expert research-oriented assistant that decides which tool(s) to use and how to accomplish the user's request.

Available tools:
{tool_list}

Return ONLY valid JSON. No explanation, no markdown.

Schema (preferred):
{{
  "tool": "<tool_name>|plan",
  "query": "<text or null>",
  "id": "<id or null>",
  "index": <number or null>,
  "plan": [
    {{"tool":"<tool_name>", "query":"<text or null>", "id":"<id or null>", "index":<number or null>}}
  ]
}}

Rules:
- If the request can be satisfied with one tool, return that tool and its parameters.
- If it requires multiple steps (search, download, analyze, synthesize), set "tool":"plan" and include an ordered "plan" array of steps.
- **If the user asks for factual, current, or browseable information (papers, datasets, models, web results), choose the relevant *tool* (e.g., "arxiv_search", "kaggle_search", "hf_model_search", "web_search") rather than returning "chat" and avoid inventing specific results or links.**
- For conversational queries or when truly ambiguous, use "chat".
- If the user asks to build, train, compare, or evaluate neural networks,
  return tool "nn_builder" with a JSON experiment specification as "query".
- Keep outputs compact and machine-parseable.

User input:
"{user_input}"
"""

        response = self.llm.query(prompt, temperature=0.0, max_tokens=600)
        try:
            return json.loads(response)
        except Exception:
            # If parsing fails, fall back to simplest chat action
            return {"tool": "chat", "query": user_input}
    # ============================================
    # MAIN PROCESS
    # ============================================

    def process(self, user_input):
        """Main entry: ask the LLM for a tool or plan, execute steps, then synthesize a final answer via the LLM."""

        if self.active_dataset and any(k in user_input.lower() for k in ["train", "mlp", "lstm", "predict"]):
            return self._handle_nn_builder({
                "dataset": {"path": self.active_dataset}
            })

        try:
            action = self._llm_route(user_input)

                    
            if self.active_dataset:
                u = user_input.lower()
                if any(k in u for k in [
                    "predict", "prediction", "train", "training",
                    "compare", "mlp", "lstm", "neural",
                    "regression", "classification", "evaluate"
                ]):
                    action = {
                        "tool": "nn_builder",
                        "query": {
                            "name": "implicit_experiment",
                            "dataset": {"path": self.active_dataset},
                            "architectures": [
                                {"type": "mlp"},
                                {"type": "lstm"}
                            ],
                            "training": {
                                "epochs": 30,
                                "trials": 1
                            }
                        }
                    }

            u_lower = user_input.lower().strip()

            # "download 1", "download 2", etc. - should ALWAYS download, never chat
            if u_lower.startswith("download") and any(char.isdigit() for char in user_input):
                # Check what was recently searched
                if self.last_kaggle_results and len(self.last_kaggle_results) > 0:
                    action = {"tool": "kaggle_download", "query": user_input}
                elif self.last_arxiv_results and len(self.last_arxiv_results) > 0:
                    action = {"tool": "arxiv_download", "query": user_input}
                elif self.last_hf_model_results and len(self.last_hf_model_results) > 0:
                    action = {"tool": "hf_model_download", "query": user_input}


            
           

            
            


            # Deterministic fallback: if the LLM returned 'chat' but the user's request
            # clearly asks for factual search/download (papers, datasets, models, web),
            # prefer the matching tool to avoid hallucinated content.
            tool_check = action.get("tool")
            if tool_check == "chat":
                u = (user_input or "").lower()
                if any(k in u for k in ["paper", "papers", "arxiv"]):
                    if any(k in u for k in ["download", "analyze", "download the", "download 1", "download 2"]):
                        # If user asked to download or analyze a paper -> arxiv_download
                        action = {"tool": "arxiv_download", "query": user_input}
                    else:
                        action = {"tool": "arxiv_search", "query": user_input}
                elif any(k in u for k in ["kaggle", "dataset", "datasets"]):
                    if "download" in u:
                        action = {"tool": "kaggle_download", "query": user_input}
                    else:
                        action = {"tool": "kaggle_search", "query": user_input}
                elif any(k in u for k in ["huggingface", "model", "gpt2", "gpt-2", "llama"]):
                    if "download" in u:
                        action = {"tool": "hf_model_download", "query": user_input}
                    elif "generate" in u or "run" in u or "inference" in u:
                        action = {"tool": "hf_inference", "query": user_input}
                    else:
                        action = {"tool": "hf_model_search", "query": user_input}
                elif any(k in u for k in ["search", "news", "latest", "web", "site:"]):
                    action = {"tool": "web_search", "query": user_input}

            # If LLM returned a plan, execute it step-by-step and synthesize
            plan = action.get("plan") or ([] if action.get("tool") != "plan" else [])

            if plan:
                outputs = []
                for step in plan:
                    tool = step.get("tool")
                    query = step.get("query")
                    idx = step.get("index")
                    id_ = step.get("id")
                    out = self._run_step(tool, query, id_, idx)
                    outputs.append({"step": step, "output": out})

                # Ask the LLM to synthesize the results into a concise response
                synthesis = self._synthesize(user_input, plan, outputs)
                return synthesis

            # Otherwise, single-tool actions (backwards-compatible)
            tool = action.get("tool")
            query = action.get("query")
            idx = action.get("index")
            id_ = action.get("id")
            if tool == "arxiv_search":
                return self._handle_arxiv_search(query or user_input)

            if tool == "arxiv_download":
                return self._handle_arxiv_download(id_ or str(idx) or query or user_input)

            if tool == "kaggle_search":
                return self._handle_kaggle_search(query or user_input)

            if tool == "kaggle_download":
                return self._handle_kaggle_download(id_ or str(idx) or query or user_input)

            if tool == "hf_model_search":
                return self._handle_hf_model_search(query or user_input)

            if tool == "hf_model_download":
                return self._handle_hf_model_download(id_ or query or user_input)

            if tool == "hf_inference":
                return self._handle_hf_inference(f"with {id_}: {query}")

            if tool == "web_search":
                return self._handle_web_search(query or user_input)

            if tool == "memory":
                return self._handle_memory(query or user_input)

            if tool == "chat" or not tool:
                return self._handle_conversation(user_input)
            
            if tool == "nn_experiment":
                return self._handle_nn_experiment(query or user_input)

            return "I couldn't determine the right action."

        except Exception as e:
            # On unexpected errors, log and fallback to chat
            try:
                import traceback
                traceback.print_exc()
            except Exception:
                pass
            return self._handle_conversation(user_input)

    # ============================================
    # PLAN / EXECUTION HELPERS
    # ============================================

    def _run_step(self, tool, query=None, id_=None, idx=None):
        """Execute a single plan step by dispatching to the matching handler."""
        try:
            if tool == "arxiv_search":
                return self._handle_arxiv_search(query or "")
            if tool == "arxiv_download":
                return self._handle_arxiv_download(id_ or str(idx) or query or "")
            if tool == "kaggle_search":
                return self._handle_kaggle_search(query or "")
            if tool == "kaggle_download":
                return self._handle_kaggle_download(id_ or str(idx) or query or "")
            if tool == "kaggle_analyze":
                return self._handle_kaggle_analyze(query or "")
            if tool == "hf_model_search":
                return self._handle_hf_model_search(query or "")
            if tool == "hf_model_download":
                return self._handle_hf_model_download(id_ or query or "")
            if tool == "hf_inference":
                return self._handle_hf_inference(f"with {id_}: {query}")
            if tool == "web_search":
                return self._handle_web_search(query or "")
            if tool == "memory":
                return self._handle_memory(query or "")
            if tool == "chat":
                return self._handle_conversation(query or "")
            if tool == "nn_builder":
                return self._handle_nn_builder(query)
            return f"Unknown tool: {tool}"
        except Exception as e:
            return f"Step failed ({tool}): {e}"

    def _synthesize(self, user_input, plan, outputs):
        """Ask the LLM to synthesize a concise human-facing answer from executed plan outputs."""
        summary_payload = {
            "user_input": user_input,
            "plan": plan,
            "outputs": outputs
        }

        prompt = (
            "You are a research assistant. The user asked: \"" + user_input + "\"\n\n"
            "The system executed the following plan steps and collected these outputs (JSON):\n\n"
            + json.dumps(summary_payload, indent=2) +
            "\n\nPlease synthesize a concise, clear response to the user (3-6 sentences), list the most important findings, and recommend next actions if appropriate."
        )

        try:
            return self.llm.query(prompt, temperature=0.2, max_tokens=400)
        except Exception as e:
            return "Synthesis failed: " + str(e)
    # ============================================
    # ARXIV HANDLERS
    # ============================================

    def _handle_arxiv_search(self, query):
        papers = self.arxiv.search_papers(query, max_results=5)
        if not papers or 'error' in papers[0]:
            return "Sorry, couldn't find any papers. Try rephrasing your query."
        self.last_arxiv_results = papers
        response = "Found these papers:\n\n"
        for i, paper in enumerate(papers, 1):
            response += f"{i}. **{paper['title']}**\n   Authors: {', '.join(paper['authors'][:3])}\n   ID: `{paper['arxiv_id']}`\n   Summary: {paper['summary'][:150]}...\n\n"
        response += "To download: say 'download 1' or 'download ARXIV_ID'"
        return response

    def _handle_arxiv_download(self, query):
        # Support direct IDs, numeric indexes into last search results, or 'top N' ranges, and optional 'report' flag
        if not query:
            return "Need an arXiv ID, result number, or 'top N' to download/analyze."

        u = (query or "").lower()
        do_report = 'report' in u or 'add to report' in u or 'to report' in u

        # Direct arXiv IDs (possibly multiple)
        matches = re.findall(r'\d{4}\.\d{4,5}(?:v\d+)?', query)
        if matches:
            outputs = []
            for arxiv_id in matches:
                analysis = self.arxiv.analyze_paper(arxiv_id)
                outputs.append(f"**Paper Analysis ({arxiv_id}):**\n\n{analysis}")
                if do_report and self.secretary:
                    # find paper metadata if available
                    paper_meta = next((p for p in self.last_arxiv_results if p.get('arxiv_id') == arxiv_id), {'arxiv_id': arxiv_id})
                    self.secretary.add_paper_analysis(paper_meta, analysis_text=analysis, pdf_path=None)
            return "\n\n".join(outputs)

        # 'top N'
        top_match = re.search(r'top\s*(\d+)', query, re.IGNORECASE)
        if top_match and self.last_arxiv_results:
            n = int(top_match.group(1))
            n = min(n, len(self.last_arxiv_results))
            analyses = []
            for i in range(n):
                arxiv_id = self.last_arxiv_results[i]['arxiv_id']
                analysis = self.arxiv.analyze_paper(arxiv_id)
                analyses.append(f"**{i+1}. {self.last_arxiv_results[i]['title']} ({arxiv_id})**\n{analysis}")
                if do_report and self.secretary:
                    self.secretary.add_paper_analysis(self.last_arxiv_results[i], analysis_text=analysis, pdf_path=None)
            return "\n\n".join(analyses)

        # One or more numeric indices separated by commas or spaces
        nums = re.findall(r'\b(\d+)\b', query)
        if nums and self.last_arxiv_results:
            outputs = []
            for nm in nums:
                idx = int(nm) - 1
                if 0 <= idx < len(self.last_arxiv_results):
                    arxiv_id = self.last_arxiv_results[idx]['arxiv_id']
                    analysis = self.arxiv.analyze_paper(arxiv_id)
                    outputs.append(f"**{nm}. {self.last_arxiv_results[idx]['title']} ({arxiv_id})**\n{analysis}")
                    if do_report and self.secretary:
                        self.secretary.add_paper_analysis(self.last_arxiv_results[idx], analysis_text=analysis, pdf_path=None)
                else:
                    outputs.append(f"Index {nm} out of range.")
            return "\n\n".join(outputs)

        # Fallback
        return "Need a valid arXiv ID, result number, or 'top N' to download/analyze."

    # ============================================
    # KAGGLE HANDLERS
    # ============================================

    def _handle_kaggle_search(self, query):
        datasets = self.kaggle.search_datasets(query, max_results=8)
        if not datasets or 'error' in datasets[0]:
            return f"Couldn't find datasets for '{query}'."
        self.last_kaggle_results = datasets
        response = f"Found {len(datasets)} datasets for '{query}':\n\n"
        for i, ds in enumerate(datasets, 1):
            response += f"{i}. **{ds['title']}** → `{ds['ref']}` | Downloads: {ds['download_count']:,}\n"
        response += "\nTo download: say 'download 1' or use dataset ref."
        return response

    def _handle_kaggle_download(self, query):
        # Support 'top N', indexes, explicit refs, optional 'analyze' and 'report' flags
        if not query:
            return "Need a valid Kaggle dataset ref, result number, or 'top N'."

        u = query.lower()
        top_match = re.search(r'top\s*(\d+)', query, re.IGNORECASE)
        do_analyze = 'analyze' in u or 'plot' in u or 'visualize' in u
        do_report = 'report' in u or 'add to report' in u or 'to report' in u

        # ---------- TOP N ----------
        if top_match and self.last_kaggle_results:
            n = int(top_match.group(1))
            n = min(n, len(self.last_kaggle_results))
            summary = []
            for i in range(n):
                ref = self.last_kaggle_results[i]['ref']
                res = self.kaggle.download_dataset(ref)
                if res:
                    # 🧠 STATE MEMORY
                    files = result.get("files", [])
                    csv_files = [f for f in files if f.lower().endswith(".csv")]

                    if csv_files:
                        self.active_dataset = csv_files[0]
                    else:
                        self.active_dataset = result.get("path")  # fallback

                    msg = f"Downloaded `{ref}`"
                    if do_analyze:
                        analysis = self.kaggle.analyze_dataset(ref)
                        if isinstance(analysis, dict) and 'error' in analysis:
                            msg += f"; analysis failed: {analysis['error']}"
                        else:
                            msg += "; analysis complete"
                            if do_report and self.secretary:
                                ds_meta = self.last_kaggle_results[i]
                                self.secretary.add_dataset_analysis(
                                    ds_meta,
                                    analysis_text=analysis.get('analysis'),
                                    plot_paths=analysis.get('plots')
                                )
                                msg += "; added to report"
                    summary.append(msg)
                else:
                    summary.append(f"Failed to download `{ref}`")
            return "\n".join(summary)

        # ---------- EXPLICIT REF ----------
        match = re.search(r'[\w-]+/[\w-]+', query)
        if match:
            dataset_ref = match.group(0)
        else:
            # ---------- NUMERIC INDEX ----------
            num_match = re.search(r'\b(\d+)\b', query)
            if num_match and self.last_kaggle_results:
                idx = int(num_match.group(1)) - 1
                if 0 <= idx < len(self.last_kaggle_results):
                    dataset_ref = self.last_kaggle_results[idx]['ref']
                else:
                    return "Dataset index out of range for recent results."
            else:
                return "Need a valid Kaggle dataset ref, result number, or 'top N'."

        result = self.kaggle.download_dataset(dataset_ref)
        if not result:
            return f"Download failed for `{dataset_ref}`."

        # 🧠 STATE MEMORY (THIS IS THE KEY LINE)
        self.active_dataset = result.get("path")

        msg = f"✓ **Downloaded!** Dataset: `{dataset_ref}`"
        if do_analyze:
            analysis = self.kaggle.analyze_dataset(dataset_ref)
            if isinstance(analysis, dict) and 'error' in analysis:
                msg += f"; analysis failed: {analysis['error']}"
            else:
                msg += ": analysis complete"
                if do_report and self.secretary:
                    ds_meta = next(
                        (d for d in self.last_kaggle_results if d.get('ref') == dataset_ref),
                        {'ref': dataset_ref}
                    )
                    self.secretary.add_dataset_analysis(
                        ds_meta,
                        analysis_text=analysis.get('analysis'),
                        plot_paths=analysis.get('plots')
                    )
                    msg += "; added to report"
        else:
            files = [f.split('/')[-1] for f in result['files'][:5]]
            msg += f" | Files: {', '.join(files)}"

        return msg

    def _handle_kaggle_analyze(self, query):
        # Accept a dataset ref, a path, or an index referring to last search results (e.g., '1' or 'top 1')
        if not query:
            return "Specify which dataset to analyze (ref or result number)."
        match = re.search(r'[\w-]+/[\w-]+', query)
        if match:
            dataset_ref = match.group(0)
        else:
            # If a number was provided, map to last results
            num_match = re.search(r'\b(\d+)\b', query)
            if num_match and self.last_kaggle_results:
                idx = int(num_match.group(1)) - 1
                if 0 <= idx < len(self.last_kaggle_results):
                    dataset_ref = self.last_kaggle_results[idx]['ref']
                else:
                    return "Dataset index out of range for recent results."
            else:
                dataset_ref = query

        analysis = self.kaggle.analyze_dataset(dataset_ref)
        if not analysis or 'error' in analysis:
            return f"Kaggle analysis failed: {analysis.get('error') if isinstance(analysis, dict) else analysis}"
        # Keep last dataset analysis result
        self.last_hf_dataset_results = analysis
        summary = (analysis.get('analysis') or '')
        plots = analysis.get('plots') or []
        msg = f"Analysis complete for `{dataset_ref}`.\nSummary:\n{summary[:1000]}\nPlots: {', '.join(plots) if plots else 'none'}"
        return msg

    def _handle_secretary_add_arxiv(self, query):
        # Resolve id similar to download handler
        match = re.search(r'\d{4}\.\d{4,5}(?:v\d+)?', query)
        arxiv_id = match.group(0) if match else (self.last_arxiv_results[int(query)-1]['arxiv_id'] if self.last_arxiv_results else None)
        if not arxiv_id:
            return "Need a valid arXiv ID, reference, or result number to add to the report."
        # Analyze using ArxivAgent
        analysis = self.arxiv.analyze_paper(arxiv_id)
        # Find title/url from last results if available
        paper = None
        for p in self.last_arxiv_results:
            if p.get('arxiv_id') == arxiv_id:
                paper = p
                break
        if not hasattr(self, 'secretary') or not self.secretary:
            return "Secretary not configured."
        # Use Secretary's method to append analysis
        self.secretary.add_paper_analysis(paper or {'arxiv_id': arxiv_id}, analysis_text=analysis, pdf_path=None)
        return f"Appended analysis for `{arxiv_id}` to report: {self.secretary.output_file}"

    def _handle_secretary_add_kaggle(self, query):
        # Resolve dataset ref
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
                    return "Dataset index out of range for recent results."
            else:
                return "Need a dataset reference or recent result index to add to the report."

        analysis = self.kaggle.analyze_dataset(dataset_ref)
        if not analysis or 'error' in analysis:
            return f"Kaggle analysis failed: {analysis.get('error') if isinstance(analysis, dict) else analysis}"
        if not hasattr(self, 'secretary') or not self.secretary:
            return "Secretary not configured."
        # Find dataset metadata from last search results if available
        ds_meta = None
        for d in self.last_kaggle_results:
            if d.get('ref') == dataset_ref:
                ds_meta = d
                break
        self.secretary.add_dataset_analysis(ds_meta or {'ref': dataset_ref, 'title': dataset_ref}, analysis_text=analysis.get('analysis'), plot_paths=analysis.get('plots'))
        return f"Appended Kaggle analysis for `{dataset_ref}` to report: {self.secretary.output_file}"

    # ============================================
    # HUGGINGFACE HANDLERS
    # ============================================

    def _handle_hf_model_search(self, query):
        models = self.huggingface.search_models(query, limit=5)
        if not models or 'error' in models[0]:
            return "No models found."
        # Enrich with size estimates when available
        enriched = []
        response = f"Found {len(models)} models:\n"
        for i, m in enumerate(models, 1):
            size_bytes = None
            try:
                size_bytes = self.huggingface.get_model_size(m['id'])
            except Exception:
                size_bytes = None
            size_str = self._human_size(size_bytes) if size_bytes else "unknown"
            enriched.append({**m, "size_bytes": size_bytes})
            response += f"{i}. {m['id']} | Downloads: {m.get('downloads',0):,} | Task: {m.get('pipeline_tag')} | Size: {size_str}\n"
        self.last_hf_model_results = enriched
        return response

    def _human_size(self, nbytes):
        if not nbytes:
            return "unknown"
        for unit in ['B','KB','MB','GB','TB']:
            if nbytes < 1024.0:
                return f"{nbytes:.2f}{unit}"
            nbytes /= 1024.0
        return f"{nbytes:.2f}PB"

    def _handle_hf_model_download(self, query):
        """Supports:
          - direct id or index ("1")
          - 'top N' to download several
          - 'and run: prompt' to download, load, and run a prompt against the model(s)
        """
        if not query:
            return "Specify a model ID, index, or 'top N'."

        # Check for 'top N'
        top_match = re.search(r'top\s*(\d+)', query, re.IGNORECASE)
        run_match = re.search(r'run\s*[:\-]?\s*(.+)', query, re.IGNORECASE)

        if top_match and self.last_hf_model_results:
            n = int(top_match.group(1))
            n = min(n, len(self.last_hf_model_results))
            results = []
            outputs = []
            for i in range(n):
                mid = self.last_hf_model_results[i]['id']
                print(f"[Orchestrator] About to download HF model (from last search): {mid!r} (query={query!r})")
                path = self.huggingface.download_model(mid)
                results.append((mid, bool(path)))
                if path and run_match:
                    prompt = run_match.group(1).strip()
                    # attempt to load and run
                    model_tuple = self.huggingface.load_model(mid)
                    if not model_tuple or model_tuple[0] is None:
                        outputs.append({"model": mid, "error": f"load failed: {model_tuple[1] if isinstance(model_tuple, tuple) else model_tuple}"})
                    else:
                        out = self.huggingface.run_inference(mid, prompt, max_length=150)
                        outputs.append({"model": mid, "output": out})
            ok = [r for r in results if r[1]]
            msg = f"Downloaded {len(ok)}/{len(results)} models: {', '.join(m[0] for m in ok)}" if ok else "No models downloaded."
            if outputs:
                msg += "\nOutputs:\n" + "\n---\n".join([f"{o.get('model')}: {o.get('output') or o.get('error')}" for o in outputs])
            return msg

        # Prefer explicit owner/model pattern anywhere in the query
        model_id = None
        repo_match = re.search(r"[\w\-_]+/[\w\-\._]+", query)
        if repo_match:
            model_id = repo_match.group(0)

        # Numeric index (e.g., 'download 2')
        if not model_id:
            num_match = re.search(r'\b(\d+)\b', query)
            if num_match and self.last_hf_model_results:
                idx = int(num_match.group(1)) - 1
                if 0 <= idx < len(self.last_hf_model_results):
                    model_id = self.last_hf_model_results[idx]['id']

        # Fallback: pick first reasonable token that isn't a stopword
        if not model_id:
            stopwords = set(['download','downloaded','run','search','for','top','and','the','with','model','models','hf','huggingface','download','try','please'])
            tokens = re.findall(r"[A-Za-z0-9_\-\/\.]+", query)
            for t in tokens:
                low = t.lower()
                if low in stopwords:
                    continue
                if re.fullmatch(r'\d+', t):
                    continue
                # avoid picking single-letter artifacts
                if len(t) < 2:
                    continue
                model_id = t
                break

        # Final fallback: use first recent search result
        if not model_id and self.last_hf_model_results:
            model_id = self.last_hf_model_results[0]['id']

        if not model_id:
            return "Could not determine model id. Provide a model name, index from last search, or 'top N'."

        # Perform download (log resolved id and raw query for debugging)
        print(f"[Orchestrator] Resolved model_id for download: {model_id!r} (raw query: {query!r})")
        path = self.huggingface.download_model(model_id)
        if not path:
            return f"Download failed for {model_id}: {self.huggingface.get_last_download_error() if hasattr(self.huggingface,'get_last_download_error') else ''}"
        # If user asked to run, load model and run prompt
        if run_match:
            prompt = run_match.group(1).strip()
            model_tuple = self.huggingface.load_model(model_id)
            if not model_tuple or model_tuple[0] is None:
                return f"Model downloaded but load failed: {model_tuple[1] if isinstance(model_tuple, tuple) else model_tuple}"
            output = self.huggingface.run_inference(model_id, prompt, max_length=150)
            return f"Downloaded and ran {model_id}.\nOutput:\n{output}"

        return f"✓ Downloaded {model_id}!"

    def _handle_hf_inference(self, query):
        match = re.search(r'with\s+([\w-]+(?:/[\w-]+)?)\s*:\s*(.+)', query)
        if not match:
            return "Specify model and prompt. Format: 'Generate with MODEL_ID: prompt'"
        model_id, prompt = match.group(1), match.group(2)
        result = self.huggingface.run_inference(model_id, prompt, max_length=150)
        return f"**Generated text:**\n\n{result}"

    # ============================================
    # WEB SEARCH HANDLER
    # ============================================

    def _handle_web_search(self, query):
        results = self.search.run_task(query, max_results=5)
        if not results or 'error' in results[0]:
            return "Web search failed."
        response = "Search results:\n"
        for i, r in enumerate(results, 1):
            response += f"{i}. {r['title']} | {r['url']}\n"
        return response

    # ============================================
    # MEMORY HANDLER
    # ============================================

    def _handle_memory(self, query):
        if 'visualiz' in query or 'graph' in query:
            path = self.memory.visualize()
            return f"Knowledge graph saved at `{path}`" if path else "Visualization failed."
        results = self.memory.search(query)
        if results:
            response = f"Found {len(results)} items:\n"
            for r in results[:10]:
                response += f"• {r['node']} ({r['data'].get('type', 'unknown')})\n"
            return response
        return "No memory results found."

    # ============================================
    # CONVERSATION HANDLER
    # ============================================

    def _handle_conversation(self, query):
        prompt = f"""You are a helpful research assistant.
User said: "{query}"
Respond naturally, concisely, and helpfully."""
        return self.llm.query(prompt, max_tokens=300, temperature=0.7)


    

    def _handle_nn_builder(self, query):
    # Accept dict or JSON string
        if isinstance(query, dict):
            spec = query
        else:
            try:
                spec = json.loads(query)
            except Exception as e:
                return f"Experiment spec must be valid JSON: {e}"

        # 🧠 Inject active dataset if missing
        dataset = spec.get("dataset", {})
        if "path" not in dataset:
            if not self.active_dataset:
                return "No active dataset. Download or specify a dataset first."
            dataset["path"] = self.active_dataset
            spec["dataset"] = dataset

        if not self.nn_builder:
            return "NN builder not configured."

        result = self.nn_builder.run_experiment(spec)

        if "error" in result:
            return f"Experiment failed: {result['error']}"

        return json.dumps(result, indent=2)



    
