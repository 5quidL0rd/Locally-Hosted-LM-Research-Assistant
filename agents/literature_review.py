"""
agents/literature_review.py - Automated Literature Review Generator

Provides comprehensive literature review automation:
- Takes a research question and searches multiple sources
- Downloads and analyzes top papers
- Extracts key findings, methods, datasets, and gaps
- Generates structured literature review document
- Creates citation graph visualization
"""

import os
import re
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict
import time

import arxiv
import fitz  # pymupdf


class LiteratureReviewAgent:
    """
    Automates the literature review process for researchers.

    Takes a research question and:
    1. Searches ArXiv for relevant papers
    2. Downloads and analyzes top N papers
    3. Extracts structured information using LLM
    4. Identifies research gaps and trends
    5. Generates a comprehensive review document
    """

    def __init__(self, llm, memory_palace, output_dir: str = "literature_reviews"):
        self.llm = llm
        self.memory = memory_palace
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.papers_dir = self.output_dir / "papers"
        self.papers_dir.mkdir(exist_ok=True)

    def generate_review(self, research_question: str,
                       max_papers: int = 10,
                       analyze_depth: str = "standard") -> Dict[str, Any]:
        """
        Generate a comprehensive literature review.

        Args:
            research_question: The main research question to explore
            max_papers: Maximum number of papers to include (default 10)
            analyze_depth: "quick", "standard", or "deep"

        Returns:
            Dictionary containing the full review and metadata
        """
        print(f"\n{'='*70}")
        print("AUTOMATED LITERATURE REVIEW")
        print(f"{'='*70}")
        print(f"Research Question: {research_question}")
        print(f"Max Papers: {max_papers}")
        print(f"Analysis Depth: {analyze_depth}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        review = {
            "research_question": research_question,
            "generated_at": timestamp,
            "parameters": {
                "max_papers": max_papers,
                "analyze_depth": analyze_depth
            },
            "papers": [],
            "themes": [],
            "methods": [],
            "gaps": [],
            "timeline": [],
        }

        # Step 1: Generate search queries
        print("\n[1/6] Generating search queries...")
        search_queries = self._generate_search_queries(research_question)
        review["search_queries"] = search_queries
        print(f"       Generated {len(search_queries)} queries")

        # Step 2: Search for papers
        print("\n[2/6] Searching ArXiv...")
        all_papers = []
        seen_ids = set()

        for query in search_queries:
            papers = self._search_arxiv(query, max_results=max_papers)
            for paper in papers:
                if paper['arxiv_id'] not in seen_ids:
                    all_papers.append(paper)
                    seen_ids.add(paper['arxiv_id'])
            time.sleep(0.5)  # Rate limiting

        # Sort by relevance (newer papers often more relevant)
        all_papers.sort(key=lambda x: x.get('published', ''), reverse=True)
        all_papers = all_papers[:max_papers]
        print(f"       Found {len(all_papers)} unique papers")

        # Step 3: Download and analyze papers
        print("\n[3/6] Downloading and analyzing papers...")
        for i, paper in enumerate(all_papers):
            print(f"       [{i+1}/{len(all_papers)}] {paper['title'][:60]}...")

            # Download
            pdf_path = self._download_paper(paper['arxiv_id'])
            paper['pdf_path'] = pdf_path

            # Extract text
            if pdf_path:
                text = self._extract_text(pdf_path)
                paper['text_length'] = len(text)

                # Analyze based on depth
                if analyze_depth == "quick":
                    paper['analysis'] = self._quick_analyze(paper, text)
                elif analyze_depth == "deep":
                    paper['analysis'] = self._deep_analyze(paper, text)
                else:
                    paper['analysis'] = self._standard_analyze(paper, text)

            review["papers"].append(paper)

            # Add to memory
            self._add_to_memory(paper, research_question)

        # Step 4: Synthesize themes and methods
        print("\n[4/6] Synthesizing themes and methods...")
        synthesis = self._synthesize_papers(review["papers"], research_question)
        review["themes"] = synthesis.get("themes", [])
        review["methods"] = synthesis.get("methods", [])
        review["datasets"] = synthesis.get("datasets", [])
        review["key_findings"] = synthesis.get("key_findings", [])

        # Step 5: Identify research gaps
        print("\n[5/6] Identifying research gaps...")
        review["gaps"] = self._identify_gaps(review["papers"], research_question)
        review["future_directions"] = self._suggest_future_directions(review)

        # Step 6: Generate report
        print("\n[6/6] Generating review document...")
        report_path = self._generate_report(review)
        review["report_path"] = report_path

        # Generate citations
        bibtex_path = self._generate_bibtex(review["papers"])
        review["bibtex_path"] = bibtex_path

        print(f"\n{'='*70}")
        print("LITERATURE REVIEW COMPLETE")
        print(f"{'='*70}")
        print(f"Papers Analyzed: {len(review['papers'])}")
        print(f"Themes Identified: {len(review['themes'])}")
        print(f"Research Gaps: {len(review['gaps'])}")
        print(f"Report: {report_path}")
        print(f"Citations: {bibtex_path}")

        return review

    def _generate_search_queries(self, research_question: str) -> List[str]:
        """Generate multiple search queries from research question"""
        prompt = f"""Given this research question, generate 3-5 specific search queries for academic paper databases.

Research Question: "{research_question}"

Return ONLY a JSON array of search query strings, no other text.
Example: ["query 1", "query 2", "query 3"]

Focus on:
- Key technical terms
- Related methodologies
- Application domains
- Alternative phrasings"""

        try:
            response = self.llm.query(prompt, temperature=0.3, max_tokens=300)
            # Extract JSON array
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                queries = json.loads(match.group())
                return queries[:5]
        except:
            pass

        # Fallback: use question directly + extract keywords
        words = research_question.lower().split()
        keywords = [w for w in words if len(w) > 4 and w not in
                   ['about', 'using', 'based', 'through', 'between']]
        return [research_question, ' '.join(keywords[:3])]

    def _search_arxiv(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search ArXiv for papers"""
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )

            papers = []
            for result in search.results():
                papers.append({
                    'title': result.title,
                    'authors': [a.name for a in result.authors],
                    'summary': result.summary,
                    'pdf_url': result.pdf_url,
                    'published': str(result.published),
                    'arxiv_id': result.entry_id.split('/')[-1],
                    'categories': result.categories,
                })
            return papers
        except Exception as e:
            print(f"       [Warning] Search failed: {e}")
            return []

    def _download_paper(self, arxiv_id: str) -> Optional[str]:
        """Download paper PDF"""
        try:
            clean_id = re.search(r'\d{4}\.\d{4,5}(?:v\d+)?', arxiv_id)
            if clean_id:
                arxiv_id = clean_id.group(0)

            paper = next(arxiv.Search(id_list=[arxiv_id]).results())
            pdf_path = str(self.papers_dir / f"{arxiv_id.replace('/', '_')}.pdf")

            if not os.path.exists(pdf_path):
                paper.download_pdf(filename=pdf_path)

            return pdf_path
        except Exception as e:
            print(f"       [Warning] Download failed for {arxiv_id}: {e}")
            return None

    def _extract_text(self, pdf_path: str, max_chars: int = 15000) -> str:
        """Extract text from PDF"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text("text")

            # Limit text length
            if len(text) > max_chars:
                half = max_chars // 2
                text = text[:half] + "\n\n[... content truncated ...]\n\n" + text[-half:]

            return text
        except Exception as e:
            return f"[Text extraction failed: {e}]"

    def _quick_analyze(self, paper: Dict, text: str) -> Dict:
        """Quick analysis - just key contribution"""
        prompt = f"""Analyze this paper briefly.

Title: {paper['title']}
Abstract: {paper['summary'][:500]}

In 2-3 sentences, state:
1. The main contribution
2. The method used"""

        try:
            response = self.llm.query(prompt, temperature=0.3, max_tokens=200, timeout=60)
            return {"summary": response, "depth": "quick"}
        except:
            return {"summary": paper['summary'][:300], "depth": "quick"}

    def _standard_analyze(self, paper: Dict, text: str) -> Dict:
        """Standard analysis with structured extraction and innovation scoring"""
        prompt = f"""Analyze this research paper and extract structured information.

Title: {paper['title']}
Abstract: {paper['summary']}

Paper text (truncated):
{text[:4000]}

Provide a JSON response with these fields:
{{
    "main_contribution": "1-2 sentence summary of key contribution",
    "methodology": "brief description of method/approach",
    "datasets_used": ["list", "of", "datasets"],
    "key_findings": ["finding 1", "finding 2"],
    "limitations": "main limitations mentioned",
    "future_work": "suggested future directions",
    "innovation_score": <1-10 rating of how innovative/unique this paper is>,
    "score_reasoning": "brief explanation of the innovation score"
}}

Innovation scoring guidelines:
- 1-3: Incremental work, minor contribution
- 4-5: Solid work but not groundbreaking
- 6-7: Good contribution with notable novel aspects
- 8-9: Significant innovation, likely influential
- 10: Exceptional paradigm-shifting work (very rare)

Return ONLY the JSON, no other text."""

        try:
            response = self.llm.query(prompt, temperature=0.2, max_tokens=600, timeout=120)
            # Extract JSON
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                analysis = json.loads(match.group())
                analysis["depth"] = "standard"

                # Ensure innovation score is valid
                if 'innovation_score' in analysis:
                    analysis['innovation_score'] = max(1, min(10, int(analysis['innovation_score'])))

                return analysis
        except Exception as e:
            pass

        return {
            "main_contribution": paper['summary'][:200],
            "methodology": "Not extracted",
            "datasets_used": [],
            "key_findings": [],
            "limitations": "Not extracted",
            "future_work": "Not extracted",
            "innovation_score": 5,
            "score_reasoning": "Score unavailable",
            "depth": "standard"
        }

    def _deep_analyze(self, paper: Dict, text: str) -> Dict:
        """Deep analysis with detailed extraction"""
        # First get standard analysis
        analysis = self._standard_analyze(paper, text)

        # Then add deeper analysis
        prompt = f"""Provide deeper analysis of this paper's methodology and implications.

Title: {paper['title']}
Initial analysis: {json.dumps(analysis, indent=2)}

Paper text (truncated):
{text[:6000]}

Add these fields (JSON format):
{{
    "methodology_details": "detailed explanation of how the method works",
    "experimental_setup": "description of experiments",
    "comparison_to_prior_work": "how does this compare to existing methods",
    "potential_applications": ["application 1", "application 2"],
    "open_questions": ["question this paper raises"],
    "reproducibility": "can this be reproduced? what's needed?"
}}

Return ONLY the JSON."""

        try:
            response = self.llm.query(prompt, temperature=0.2, max_tokens=600, timeout=180)
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                deep_analysis = json.loads(match.group())
                analysis.update(deep_analysis)
                analysis["depth"] = "deep"
        except:
            pass

        return analysis

    def _synthesize_papers(self, papers: List[Dict], research_question: str) -> Dict:
        """Synthesize findings across all papers"""
        # Prepare summaries
        paper_summaries = []
        for p in papers:
            summary = f"- {p['title']}: {p.get('analysis', {}).get('main_contribution', p['summary'][:150])}"
            paper_summaries.append(summary)

        prompt = f"""Synthesize these research papers to identify themes and patterns.

Research Question: {research_question}

Papers analyzed:
{chr(10).join(paper_summaries[:15])}

Provide a JSON response:
{{
    "themes": [
        {{"name": "theme name", "description": "what this theme covers", "paper_count": N}}
    ],
    "methods": [
        {{"name": "method name", "description": "brief description", "frequency": "common/emerging/rare"}}
    ],
    "datasets": ["dataset1", "dataset2"],
    "key_findings": ["major finding 1", "major finding 2", "major finding 3"]
}}

Return ONLY the JSON."""

        try:
            response = self.llm.query(prompt, temperature=0.3, max_tokens=800, timeout=120)
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                return json.loads(match.group())
        except:
            pass

        return {
            "themes": [],
            "methods": [],
            "datasets": [],
            "key_findings": []
        }

    def _identify_gaps(self, papers: List[Dict], research_question: str) -> List[Dict]:
        """Identify research gaps from the papers"""
        limitations = []
        future_work = []

        for p in papers:
            analysis = p.get('analysis', {})
            if analysis.get('limitations'):
                limitations.append(analysis['limitations'])
            if analysis.get('future_work'):
                future_work.append(analysis['future_work'])

        prompt = f"""Based on these papers' limitations and future work suggestions, identify research gaps.

Research Question: {research_question}

Limitations mentioned:
{chr(10).join(f'- {l}' for l in limitations[:10])}

Future work suggested:
{chr(10).join(f'- {f}' for f in future_work[:10])}

Identify 3-5 specific research gaps. Return as JSON array:
[
    {{"gap": "description of gap", "importance": "high/medium/low", "difficulty": "high/medium/low"}}
]

Return ONLY the JSON array."""

        try:
            response = self.llm.query(prompt, temperature=0.4, max_tokens=500, timeout=120)
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                return json.loads(match.group())
        except:
            pass

        return [{"gap": "Unable to extract gaps automatically", "importance": "unknown", "difficulty": "unknown"}]

    def _suggest_future_directions(self, review: Dict) -> List[str]:
        """Suggest future research directions"""
        prompt = f"""Based on this literature review, suggest 3-5 concrete future research directions.

Research Question: {review['research_question']}
Themes: {[t.get('name', t) for t in review.get('themes', [])]}
Gaps: {[g.get('gap', g) for g in review.get('gaps', [])]}

Provide specific, actionable research directions as a JSON array of strings.
Example: ["Direction 1: specific proposal", "Direction 2: specific proposal"]

Return ONLY the JSON array."""

        try:
            response = self.llm.query(prompt, temperature=0.5, max_tokens=400, timeout=120)
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                return json.loads(match.group())
        except:
            pass

        return ["Review gaps section for research opportunities"]

    def _add_to_memory(self, paper: Dict, research_question: str):
        """Add paper to memory palace"""
        try:
            node_id = f"paper_{paper['arxiv_id']}"
            self.memory.add_node(
                node_id,
                "arxiv_paper",
                {
                    'title': paper['title'],
                    'authors': paper['authors'],
                    'arxiv_id': paper['arxiv_id'],
                    'summary': paper['summary'],
                    'published': paper['published'],
                    'pdf_url': paper['pdf_url'],
                    'analysis': paper.get('analysis', {})
                }
            )
            self.memory.add_edge(
                f"lit_review_{research_question[:30]}",
                node_id,
                "includes_paper"
            )
        except:
            pass

    def _generate_report(self, review: Dict) -> str:
        """Generate markdown report"""
        lines = []

        # Title
        lines.append(f"# Literature Review")
        lines.append(f"\n## Research Question")
        lines.append(f"\n> {review['research_question']}")
        lines.append(f"\n*Generated: {review['generated_at']}*")
        lines.append(f"\n*Papers Analyzed: {len(review['papers'])}*")

        # Executive Summary
        lines.append("\n## Executive Summary\n")
        if review.get('key_findings'):
            for finding in review['key_findings'][:5]:
                lines.append(f"- {finding}")

        # Themes
        if review.get('themes'):
            lines.append("\n## Major Themes\n")
            for theme in review['themes']:
                if isinstance(theme, dict):
                    lines.append(f"### {theme.get('name', 'Theme')}")
                    lines.append(f"\n{theme.get('description', '')}")
                    if theme.get('paper_count'):
                        lines.append(f"\n*Papers: {theme['paper_count']}*\n")
                else:
                    lines.append(f"- {theme}")

        # Methods
        if review.get('methods'):
            lines.append("\n## Methodologies\n")
            lines.append("| Method | Description | Frequency |")
            lines.append("|--------|-------------|-----------|")
            for method in review['methods']:
                if isinstance(method, dict):
                    lines.append(f"| {method.get('name', 'N/A')} | {method.get('description', 'N/A')} | {method.get('frequency', 'N/A')} |")

        # Datasets
        if review.get('datasets'):
            lines.append("\n## Common Datasets\n")
            for ds in review['datasets']:
                lines.append(f"- {ds}")

        # Paper Summaries (sorted by innovation score)
        lines.append("\n## Paper Summaries\n")

        # Sort papers by innovation score
        papers_with_scores = []
        for paper in review['papers']:
            score = paper.get('analysis', {}).get('innovation_score', 5)
            papers_with_scores.append((score, paper))
        papers_with_scores.sort(key=lambda x: x[0], reverse=True)

        for i, (score, paper) in enumerate(papers_with_scores, 1):
            # Innovation score visual
            score_bar = "█" * score + "░" * (10 - score)
            score_label = {
                (1, 3): "Incremental",
                (4, 5): "Solid",
                (6, 7): "Notable",
                (8, 9): "Significant",
                (10, 10): "Exceptional"
            }
            label = next((v for (lo, hi), v in score_label.items() if lo <= score <= hi), "")

            lines.append(f"### {i}. {paper['title']}\n")
            lines.append(f"**Innovation Score:** {score}/10 [{score_bar}] *{label}*")
            lines.append(f"**Authors:** {', '.join(paper['authors'][:5])}")
            lines.append(f"**ArXiv:** [{paper['arxiv_id']}](https://arxiv.org/abs/{paper['arxiv_id']})")
            lines.append(f"**Published:** {paper['published'][:10]}")

            analysis = paper.get('analysis', {})
            if analysis.get('score_reasoning'):
                lines.append(f"\n*Score Reasoning: {analysis['score_reasoning']}*")
            if analysis.get('main_contribution'):
                lines.append(f"\n**Contribution:** {analysis['main_contribution']}")
            if analysis.get('methodology'):
                lines.append(f"\n**Method:** {analysis['methodology']}")
            if analysis.get('key_findings'):
                lines.append("\n**Key Findings:**")
                for finding in analysis['key_findings']:
                    lines.append(f"- {finding}")
            lines.append("")

        # Research Gaps
        if review.get('gaps'):
            lines.append("\n## Research Gaps\n")
            for gap in review['gaps']:
                if isinstance(gap, dict):
                    importance = gap.get('importance', 'unknown')
                    emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(importance, "⚪")
                    lines.append(f"{emoji} **{gap.get('gap', 'N/A')}**")
                    lines.append(f"   - Importance: {importance}")
                    lines.append(f"   - Difficulty: {gap.get('difficulty', 'unknown')}\n")
                else:
                    lines.append(f"- {gap}")

        # Future Directions
        if review.get('future_directions'):
            lines.append("\n## Suggested Future Directions\n")
            for i, direction in enumerate(review['future_directions'], 1):
                lines.append(f"{i}. {direction}")

        # Search queries used
        lines.append("\n## Appendix: Search Methodology\n")
        lines.append("**Search Queries Used:**")
        for q in review.get('search_queries', []):
            lines.append(f"- `{q}`")

        # Write to file
        report_path = self.output_dir / f"review_{review['generated_at']}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        return str(report_path)

    def _generate_bibtex(self, papers: List[Dict]) -> str:
        """Generate BibTeX file for all papers"""
        entries = []

        for paper in papers:
            arxiv_id = paper.get('arxiv_id', '').replace('.', '_').replace('/', '_')
            if not arxiv_id:
                continue

            authors = paper.get('authors', [])
            author_str = ' and '.join(authors) if authors else 'Unknown'
            year = paper.get('published', '')[:4] or '2024'

            entry = f"""@article{{{arxiv_id},
    title = {{{paper.get('title', 'Unknown')}}},
    author = {{{author_str}}},
    year = {{{year}}},
    eprint = {{{paper.get('arxiv_id', '')}}},
    archivePrefix = {{arXiv}},
    url = {{{paper.get('pdf_url', '')}}}
}}"""
            entries.append(entry)

        if not entries:
            return None

        bibtex_path = self.output_dir / f"citations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.bib"
        with open(bibtex_path, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(entries))

        return str(bibtex_path)

    def quick_search(self, topic: str, max_papers: int = 5) -> str:
        """Quick search and summary without full analysis"""
        papers = self._search_arxiv(topic, max_results=max_papers)

        if not papers:
            return f"No papers found for '{topic}'"

        response = f"Found {len(papers)} papers on '{topic}':\n\n"
        for i, p in enumerate(papers, 1):
            response += f"{i}. **{p['title']}**\n"
            response += f"   Authors: {', '.join(p['authors'][:3])}\n"
            response += f"   ArXiv: `{p['arxiv_id']}`\n"
            response += f"   {p['summary'][:200]}...\n\n"

        response += "\nUse 'generate literature review on <topic>' for full analysis."
        return response


def create_literature_review_agent(llm, memory_palace, output_dir: str = "literature_reviews"):
    """Factory function to create LiteratureReviewAgent"""
    return LiteratureReviewAgent(llm, memory_palace, output_dir)
