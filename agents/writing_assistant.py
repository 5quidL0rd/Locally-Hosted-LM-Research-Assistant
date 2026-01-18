"""
agents/writing_assistant.py - Research Writing and Citation Assistant

Provides research writing support:
- Generate BibTeX citations from memory
- Suggest citations for claims
- Help structure methodology sections
- Check for missing citations
- Format references in various styles
"""

import os
import re
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


class WritingAssistant:
    """
    Research writing assistant that helps with citations and paper structure.

    Capabilities:
    - Generate citations from stored papers
    - Suggest relevant citations for claims
    - Help write methodology sections
    - Check draft sections for needed citations
    - Export references in multiple formats
    """

    def __init__(self, llm, memory_palace, output_dir: str = "writing_output"):
        self.llm = llm
        self.memory = memory_palace
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def suggest_citations(self, claim: str, context: str = None) -> List[Dict]:
        """
        Suggest relevant citations from memory for a given claim.

        Args:
            claim: The statement that needs citation support
            context: Optional surrounding context

        Returns:
            List of suggested citations with relevance scores
        """
        # Search memory for relevant papers
        relevant_papers = self._search_relevant_papers(claim)

        if not relevant_papers:
            return [{
                "message": "No relevant papers found in memory. Try searching ArXiv first.",
                "suggestion": f'Search ArXiv: "papers about {claim[:50]}"'
            }]

        # Use LLM to rank relevance
        suggestions = self._rank_citations(claim, relevant_papers, context)

        return suggestions

    def _search_relevant_papers(self, query: str) -> List[Dict]:
        """Search memory for papers relevant to query"""
        papers = []

        # Get all papers from memory
        for node, data in self.memory.graph.nodes(data=True):
            if data.get('type') == 'arxiv_paper':
                paper_data = data.get('data', {})
                papers.append({
                    'node_id': node,
                    'title': paper_data.get('title', ''),
                    'authors': paper_data.get('authors', []),
                    'summary': paper_data.get('summary', ''),
                    'arxiv_id': paper_data.get('arxiv_id', ''),
                    'published': paper_data.get('published', ''),
                })

        if not papers:
            return []

        # Simple keyword matching for initial filtering
        query_words = set(query.lower().split())
        scored_papers = []

        for paper in papers:
            title_words = set(paper['title'].lower().split())
            summary_words = set(paper['summary'].lower().split())
            all_words = title_words | summary_words

            overlap = len(query_words & all_words)
            if overlap > 0:
                scored_papers.append((overlap, paper))

        # Sort by overlap score
        scored_papers.sort(reverse=True, key=lambda x: x[0])

        return [p[1] for p in scored_papers[:10]]

    def _rank_citations(self, claim: str, papers: List[Dict],
                       context: str = None) -> List[Dict]:
        """Use LLM to rank citation relevance"""
        paper_list = "\n".join([
            f"{i+1}. {p['title']} ({p['arxiv_id']}): {p['summary'][:150]}..."
            for i, p in enumerate(papers[:8])
        ])

        prompt = f"""Rank these papers by relevance for supporting this claim.

Claim: "{claim}"
{f'Context: {context}' if context else ''}

Available papers:
{paper_list}

Return a JSON array ranking the top 3 most relevant papers:
[
    {{"rank": 1, "paper_index": N, "reason": "why this supports the claim"}},
    {{"rank": 2, "paper_index": N, "reason": "why this supports the claim"}},
    {{"rank": 3, "paper_index": N, "reason": "why this supports the claim"}}
]

Return ONLY the JSON array."""

        try:
            response = self.llm.query(prompt, temperature=0.2, max_tokens=400)
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                rankings = json.loads(match.group())

                suggestions = []
                for r in rankings:
                    idx = r.get('paper_index', 1) - 1
                    if 0 <= idx < len(papers):
                        paper = papers[idx]
                        suggestions.append({
                            'rank': r.get('rank'),
                            'title': paper['title'],
                            'arxiv_id': paper['arxiv_id'],
                            'authors': paper['authors'],
                            'reason': r.get('reason', ''),
                            'citation_key': paper['arxiv_id'].replace('.', '_').replace('/', '_'),
                        })
                return suggestions
        except:
            pass

        # Fallback: return top papers without LLM ranking
        return [
            {
                'rank': i+1,
                'title': p['title'],
                'arxiv_id': p['arxiv_id'],
                'authors': p['authors'],
                'reason': 'Keyword match',
                'citation_key': p['arxiv_id'].replace('.', '_').replace('/', '_'),
            }
            for i, p in enumerate(papers[:3])
        ]

    def check_citations_needed(self, text: str) -> List[Dict]:
        """
        Analyze text and identify claims that need citations.

        Args:
            text: Research text to analyze

        Returns:
            List of claims that need citations with suggestions
        """
        prompt = f"""Analyze this research text and identify statements that need citations.

Text:
{text[:3000]}

For each claim that should be cited, return JSON:
[
    {{
        "claim": "the statement that needs citation",
        "reason": "why it needs a citation",
        "citation_type": "empirical/theoretical/methodological"
    }}
]

Focus on:
- Factual claims about prior work
- Statistical claims
- Methodological choices referencing standard practices
- Comparative statements

Return ONLY the JSON array."""

        try:
            response = self.llm.query(prompt, temperature=0.3, max_tokens=600)
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                claims = json.loads(match.group())

                # For each claim, try to find relevant citations
                for claim in claims:
                    suggestions = self.suggest_citations(claim['claim'])
                    claim['suggested_citations'] = suggestions[:2]

                return claims
        except:
            pass

        return [{"error": "Could not analyze text for citations"}]

    def generate_methodology_section(self, experiment_description: str,
                                    include_citations: bool = True) -> str:
        """
        Help write a methodology section.

        Args:
            experiment_description: Description of what was done
            include_citations: Whether to suggest citations

        Returns:
            Drafted methodology section
        """
        prompt = f"""Write a methodology section for a research paper based on this description.

Experiment Description:
{experiment_description}

Write in third person, past tense, academic style.
Include subsections for:
1. Data/Dataset
2. Model/Approach
3. Training/Implementation Details
4. Evaluation Metrics

Mark places where citations would be appropriate with [CITE].

Keep it concise (300-400 words)."""

        try:
            methodology = self.llm.query(prompt, temperature=0.4, max_tokens=600)

            if include_citations:
                # Find places marked for citation and suggest
                cite_spots = re.findall(r'\[CITE(?:\:([^\]]+))?\]', methodology)
                if cite_spots:
                    methodology += "\n\n---\n**Suggested Citations:**\n"
                    for topic in set(cite_spots):
                        if topic:
                            suggestions = self.suggest_citations(topic)
                            if suggestions and 'message' not in suggestions[0]:
                                methodology += f"\n*For '{topic}':*\n"
                                for s in suggestions[:2]:
                                    methodology += f"  - {s['title']} ({s['arxiv_id']})\n"

            return methodology
        except:
            return "Error generating methodology section. Please try again."

    def format_citation(self, arxiv_id: str, style: str = "apa") -> str:
        """
        Format a citation in the specified style.

        Args:
            arxiv_id: ArXiv paper ID
            style: Citation style (apa, mla, chicago, bibtex)

        Returns:
            Formatted citation string
        """
        # Find paper in memory
        paper = None
        for node, data in self.memory.graph.nodes(data=True):
            if data.get('type') == 'arxiv_paper':
                paper_data = data.get('data', {})
                if paper_data.get('arxiv_id') == arxiv_id:
                    paper = paper_data
                    break

        if not paper:
            return f"Paper {arxiv_id} not found in memory. Search and download it first."

        authors = paper.get('authors', ['Unknown'])
        title = paper.get('title', 'Unknown Title')
        year = paper.get('published', '')[:4] or '2024'

        if style.lower() == 'bibtex':
            key = arxiv_id.replace('.', '_').replace('/', '_')
            author_str = ' and '.join(authors)
            return f"""@article{{{key},
    title = {{{title}}},
    author = {{{author_str}}},
    year = {{{year}}},
    eprint = {{{arxiv_id}}},
    archivePrefix = {{arXiv}}
}}"""

        elif style.lower() == 'apa':
            if len(authors) > 2:
                author_str = f"{authors[0].split()[-1]} et al."
            elif len(authors) == 2:
                author_str = f"{authors[0].split()[-1]} & {authors[1].split()[-1]}"
            else:
                author_str = authors[0].split()[-1]
            return f"{author_str} ({year}). {title}. arXiv preprint arXiv:{arxiv_id}."

        elif style.lower() == 'mla':
            author_str = authors[0] if authors else "Unknown"
            if len(authors) > 1:
                author_str += ", et al"
            return f'{author_str}. "{title}." arXiv preprint arXiv:{arxiv_id} ({year}).'

        elif style.lower() == 'chicago':
            author_str = ", ".join(authors[:3])
            if len(authors) > 3:
                author_str += ", et al."
            return f'{author_str}. "{title}." arXiv preprint arXiv:{arxiv_id} ({year}).'

        else:
            return f"{', '.join(authors[:2])} ({year}). {title}. arXiv:{arxiv_id}"

    def export_all_citations(self, style: str = "bibtex",
                            output_path: str = None) -> str:
        """
        Export all citations from memory in specified format.

        Args:
            style: Citation style (bibtex, apa, mla)
            output_path: Output file path (auto-generated if None)

        Returns:
            Path to the generated file
        """
        citations = []

        for node, data in self.memory.graph.nodes(data=True):
            if data.get('type') == 'arxiv_paper':
                paper_data = data.get('data', {})
                arxiv_id = paper_data.get('arxiv_id')
                if arxiv_id:
                    citation = self.format_citation(arxiv_id, style)
                    citations.append(citation)

        if not citations:
            return "No papers found in memory to export."

        # Determine file extension
        ext = ".bib" if style.lower() == "bibtex" else ".txt"

        if output_path is None:
            output_path = self.output_dir / f"citations_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"

        separator = "\n\n" if style.lower() == "bibtex" else "\n"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(separator.join(citations))

        return str(output_path)

    def generate_abstract(self, paper_content: str) -> str:
        """
        Generate an abstract for a paper.

        Args:
            paper_content: The main content/findings of the paper

        Returns:
            Generated abstract
        """
        prompt = f"""Write a research paper abstract based on this content.

Content:
{paper_content[:2000]}

Write a 150-250 word abstract following this structure:
1. Background/Problem (1-2 sentences)
2. Method/Approach (2-3 sentences)
3. Results (2-3 sentences)
4. Conclusion/Implications (1-2 sentences)

Write in third person, past tense for methods/results, present tense for conclusions."""

        try:
            abstract = self.llm.query(prompt, temperature=0.4, max_tokens=400)
            return abstract
        except:
            return "Error generating abstract. Please try again."

    def improve_writing(self, text: str, focus: str = "clarity") -> str:
        """
        Suggest improvements for academic writing.

        Args:
            text: Text to improve
            focus: What to focus on (clarity, conciseness, formality, flow)

        Returns:
            Improved text with explanations
        """
        focus_instructions = {
            "clarity": "Make the text clearer and easier to understand. Simplify complex sentences.",
            "conciseness": "Remove redundant words and phrases. Make it more direct.",
            "formality": "Make the language more formal and academic. Remove colloquialisms.",
            "flow": "Improve transitions between sentences. Ensure logical progression.",
        }

        instruction = focus_instructions.get(focus, focus_instructions["clarity"])

        prompt = f"""Improve this academic text. Focus: {instruction}

Original text:
{text[:1500]}

Provide:
1. Improved version of the text
2. Brief explanation of key changes made

Format:
IMPROVED TEXT:
[improved text here]

CHANGES MADE:
[list of changes]"""

        try:
            response = self.llm.query(prompt, temperature=0.3, max_tokens=800)
            return response
        except:
            return "Error improving text. Please try again."

    def related_work_outline(self, topic: str) -> str:
        """
        Generate an outline for a related work section.

        Args:
            topic: The research topic

        Returns:
            Structured outline with paper placeholders
        """
        # Get papers from memory
        papers = []
        for node, data in self.memory.graph.nodes(data=True):
            if data.get('type') == 'arxiv_paper':
                paper_data = data.get('data', {})
                papers.append({
                    'title': paper_data.get('title', ''),
                    'arxiv_id': paper_data.get('arxiv_id', ''),
                    'summary': paper_data.get('summary', '')[:200],
                })

        paper_list = "\n".join([
            f"- {p['title']} ({p['arxiv_id']}): {p['summary']}"
            for p in papers[:15]
        ])

        prompt = f"""Create an outline for a Related Work section on "{topic}".

Available papers in memory:
{paper_list if paper_list else "No papers found - outline will use placeholders"}

Create a structured outline with:
1. 3-4 main subsections based on themes/approaches
2. Under each subsection, indicate which papers to discuss
3. Brief notes on what to say about each paper

Format as markdown outline."""

        try:
            outline = self.llm.query(prompt, temperature=0.4, max_tokens=600)
            return outline
        except:
            return "Error generating outline. Please try again."


def create_writing_assistant(llm, memory_palace, output_dir: str = "writing_output"):
    """Factory function to create WritingAssistant"""
    return WritingAssistant(llm, memory_palace, output_dir)
