import os
import json
import re
import requests
from collections import deque
from pathlib import Path
from datetime import datetime


class LocalLLM:
    """
    Local LLM wrapper with:
    - sliding window conversational memory
    - persistent personal profile memory
    - episodic conversation summaries
    - stable researcher personality
    """

    PROFILE_PATH = Path.home() / ".local_llm_profile.json"
    CONVERSATION_PATH = Path.home() / ".local_llm_conversations.json"

    RESEARCHER_SYSTEM_PROMPT = (
        "You are a research assistant designed to help with technical exploration, "
        "scientific reasoning, and structured problem-solving.\n\n"
        "Behavior guidelines:\n"
        "- Be concise but thorough.\n"
        "- Prefer accuracy over speculation.\n"
        "- When unsure, state uncertainty clearly.\n"
        "- Break complex ideas into clear steps.\n"
        "- Treat the user as a capable collaborator, not a novice.\n"
        "- Do not repeat known personal details unless directly relevant.\n"
        "- Maintain continuity across the conversation when possible.\n"
    )

    def __init__(
        self,
        endpoint=None,
        model=None,
        max_turns=6
    ):
        self.endpoint = endpoint or os.getenv(
            "LOCAL_LLM_ENDPOINT",
            "http://127.0.0.1:1234/v1/chat/completions"
        )
        self.model = model or os.getenv("LOCAL_LLM_MODEL", "local-model")

        self.chat_memory = deque(maxlen=max_turns * 2)
        self.profile = self._load_json(self.PROFILE_PATH, default={})
        self.conversation_log = []

    # --------------------------------------------------
    # JSON helpers
    # --------------------------------------------------

    def _load_json(self, path, default):
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                return default
        return default

    def _save_json(self, path, data):
        try:
            path.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    # --------------------------------------------------
    # Personal memory extraction
    # --------------------------------------------------

    def _extract_personal_facts(self, text: str):
        patterns = {
            "occupation": r"\b(i work as|my job is|i am a|i'm a)\s+(.+)",
            "field": r"\b(i work in|my field is)\s+(.+)",
            "education": r"\b(i study|i am studying|my major is)\s+(.+)",
            "location": r"\b(i live in|i'm from)\s+(.+)"
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                self.profile[key] = match.group(2).strip()
                self._save_json(self.PROFILE_PATH, self.profile)

    def _explicit_remember(self, text: str) -> bool:
        match = re.search(r"\bremember (that )?(.+)", text, re.IGNORECASE)
        if match:
            note = match.group(2).strip()
            self.profile.setdefault("notes", []).append(note)
            self._save_json(self.PROFILE_PATH, self.profile)
            return True
        return False

    # --------------------------------------------------
    # System message assembly
    # --------------------------------------------------

    def _system_messages(self):
        messages = [{
            "role": "system",
            "content": self.RESEARCHER_SYSTEM_PROMPT
        }]

        if self.profile:
            facts = "\n".join(f"- {k}: {v}" for k, v in self.profile.items())
            messages.append({
                "role": "system",
                "content": (
                    "Known user background information:\n"
                    f"{facts}"
                )
            })

        return messages

    # --------------------------------------------------
    # Main query
    # --------------------------------------------------

    def query(
        self,
        prompt: str,
        temperature=0.0,
        max_tokens=None,
        timeout=120
    ):
        messages = []
        messages.extend(self._system_messages())

        messages.extend(
            {"role": r, "content": c}
            for r, c in self.chat_memory
        )

        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature
        }

        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)

        try:
            resp = requests.post(
                self.endpoint,
                json=payload,
                timeout=timeout
            )
            resp.raise_for_status()
            data = resp.json()
            reply = data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(f"LLM request failed: {e}")

        # Update memories
        self.chat_memory.append(("user", prompt))
        self.chat_memory.append(("assistant", reply))
        self.conversation_log.append(("user", prompt))
        self.conversation_log.append(("assistant", reply))

        if not self._explicit_remember(prompt):
            self._extract_personal_facts(prompt)

        return reply

    # --------------------------------------------------
    # Conversation summarization on exit
    # --------------------------------------------------

    def summarize_and_store_conversation(self):
        if not self.conversation_log:
            return

        summary_prompt = (
            "Summarize the following conversation in 5–7 bullet points, "
            "focusing on topics discussed and conclusions reached.\n\n"
        )

        for role, content in self.conversation_log:
            summary_prompt += f"{role.upper()}: {content}\n"

        summary = self.query(
            summary_prompt,
            temperature=0.2,
            max_tokens=300
        )

        history = self._load_json(self.CONVERSATION_PATH, default=[])

        history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "summary": summary
        })

        self._save_json(self.CONVERSATION_PATH, history)