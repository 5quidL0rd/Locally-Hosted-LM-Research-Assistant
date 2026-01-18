"""
agents/arxiv_agent.py - Downloads and analyzes papers from arXiv

Includes innovation scoring (1-10) for each paper based on:
- Novelty of approach
- Uniqueness of contribution
- Potential impact
- Technical depth
"""

import os
import re
import json
from typing import Dict, Any, List, Optional
import arxiv
import fitz  # pymupdf


class ArxivAgent:
    """Downloads and analyzes papers from arXiv with innovation scoring"""

    def __init__(self, llm, memory_palace, download_dir="arxiv_papers"):
        self.llm = llm
        self.memory = memory_palace
        self.download_dir = download_dir
        os.makedirs(download_dir, exist_ok=True)
    
    def search_papers(self, query, max_results=5):
        """Search arXiv for papers"""
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            papers = []
            for result in search.results():
                paper_info = {
                    'title': result.title,
                    'authors': [a.name for a in result.authors],
                    'summary': result.summary,
                    'pdf_url': result.pdf_url,
                    'published': str(result.published),
                    'arxiv_id': result.entry_id.split('/')[-1]
                }
                papers.append(paper_info)
                
                # Add to memory palace
                self.memory.add_node(
                    f"paper_{paper_info['arxiv_id']}", 
                    "arxiv_paper", 
                    paper_info
                )
                self.memory.add_edge(query, f"paper_{paper_info['arxiv_id']}", "search_result")
            
            return papers
        except Exception as e:
            return [{"error": f"arXiv search failed: {e}"}]
    
    def download_paper(self, arxiv_id):
        """Download PDF of a paper"""
        try:
            # Clean the arxiv_id - extract just the ID
            match = re.search(r'\d{4}\.\d{4,5}(?:v\d+)?', arxiv_id)
            if match:
                clean_id = match.group(0)
            else:
                clean_id = arxiv_id
            
            print(f"[arXiv] Using ID: {clean_id}")
            paper = next(arxiv.Search(id_list=[clean_id]).results())
            pdf_path = os.path.join(self.download_dir, f"{clean_id.replace('/', '_')}.pdf")
            paper.download_pdf(filename=pdf_path)
            print(f"[arXiv] Downloaded: {pdf_path}")
            return pdf_path
        except Exception as e:
            print(f"[arXiv] Download failed: {e}")
            return None
    
    def analyze_paper(self, arxiv_id: str, include_score: bool = True) -> str:
        """Download and analyze a paper with optional innovation scoring"""
        pdf_path = self.download_paper(arxiv_id)
        if not pdf_path:
            return "Failed to download paper"

        # Extract text
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text")

        # Limit text to avoid crashing the LLM
        max_chars = 6000
        if len(text) > max_chars:
            half = max_chars // 2
            text = text[:half] + "\n\n[... middle section omitted ...]\n\n" + text[-half:]

        # Summarize with LLM
        prompt = f"""Analyze this research paper and provide:
1. Main contribution/findings (2-3 sentences)
2. Methodology used
3. Key limitations
4. Potential applications

Keep your response concise (under 300 words).

Paper text:
{text}
"""
        try:
            analysis = self.llm.query(prompt, max_tokens=400, timeout=180)

            # Get innovation score if requested
            score_info = None
            if include_score:
                score_info = self.score_paper(arxiv_id, text)

            # Store analysis in memory
            analysis_data = {"analysis": analysis, "pdf_path": pdf_path}
            if score_info:
                analysis_data["innovation_score"] = score_info

            self.memory.add_node(f"analysis_{arxiv_id}", "paper_analysis", analysis_data)
            self.memory.add_edge(f"paper_{arxiv_id}", f"analysis_{arxiv_id}", "analyzed_as")

            # Format response with score if available
            if score_info:
                score_section = self._format_score(score_info)
                return f"{analysis}\n\n{score_section}"

            return analysis
        except RuntimeError as e:
            error_msg = str(e)
            if "crashed" in error_msg.lower() or "exit code" in error_msg.lower():
                return "LLM crashed (paper too large or model overloaded). Try a different paper or restart LM Studio."
            return f"Analysis failed: {e}"

    def score_paper(self, arxiv_id: str, text: str = None) -> Dict[str, Any]:
        """
        Score a paper on innovation, uniqueness, and interest (1-10 scale).

        Returns dict with:
        - overall_score: 1-10 overall rating
        - innovation: 1-10 how novel is the approach
        - uniqueness: 1-10 how unique is the contribution
        - impact: 1-10 potential impact on the field
        - technical_depth: 1-10 rigor and depth of work
        - reasoning: explanation of scores
        """
        # Get text if not provided
        if text is None:
            pdf_path = self.download_paper(arxiv_id)
            if not pdf_path:
                return {"error": "Could not download paper"}

            doc = fitz.open(pdf_path)
            text = ""
            for page in doc[:5]:  # First 5 pages for scoring
                text += page.get_text("text")

        # Truncate for scoring
        if len(text) > 4000:
            text = text[:2000] + "\n...\n" + text[-2000:]

        prompt = f"""You are an expert research evaluator. Score this paper on a scale of 1-10 for each criterion.

Paper text (truncated):
{text}

Provide your evaluation as a JSON object with these fields:
{{
    "overall_score": <1-10>,
    "innovation": <1-10>,
    "uniqueness": <1-10>,
    "impact": <1-10>,
    "technical_depth": <1-10>,
    "reasoning": "<2-3 sentences explaining the scores>"
}}

Scoring guidelines:
- 1-3: Incremental work, minor contribution, limited novelty
- 4-5: Solid work but not groundbreaking
- 6-7: Good contribution with notable novel aspects
- 8-9: Significant innovation, likely to influence the field
- 10: Exceptional, paradigm-shifting work (very rare)

Be objective and critical. Most papers score 4-7.

Return ONLY the JSON object, no other text."""

        try:
            response = self.llm.query(prompt, temperature=0.2, max_tokens=400, timeout=120)

            # Extract JSON
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                scores = json.loads(match.group())

                # Validate scores
                for key in ['overall_score', 'innovation', 'uniqueness', 'impact', 'technical_depth']:
                    if key in scores:
                        scores[key] = max(1, min(10, int(scores[key])))

                scores['arxiv_id'] = arxiv_id
                return scores

        except Exception as e:
            print(f"[ArXiv] Scoring failed: {e}")

        # Fallback: return neutral scores
        return {
            "arxiv_id": arxiv_id,
            "overall_score": 5,
            "innovation": 5,
            "uniqueness": 5,
            "impact": 5,
            "technical_depth": 5,
            "reasoning": "Could not generate detailed scores."
        }

    def _format_score(self, score_info: Dict[str, Any]) -> str:
        """Format score information for display"""
        overall = score_info.get('overall_score', 5)

        # Visual rating bar
        filled = "█" * overall
        empty = "░" * (10 - overall)
        rating_bar = f"[{filled}{empty}]"

        output = f"""
{'='*50}
INNOVATION SCORE: {overall}/10 {rating_bar}
{'='*50}
  Innovation:      {score_info.get('innovation', 5)}/10
  Uniqueness:      {score_info.get('uniqueness', 5)}/10
  Potential Impact:{score_info.get('impact', 5)}/10
  Technical Depth: {score_info.get('technical_depth', 5)}/10

{score_info.get('reasoning', '')}
{'='*50}"""

        return output

    def search_and_score(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search papers and return with innovation scores"""
        papers = self.search_papers(query, max_results)

        if not papers or (isinstance(papers, list) and papers and 'error' in papers[0]):
            return papers

        scored_papers = []
        for paper in papers:
            # Quick score based on abstract only
            abstract = paper.get('summary', '')
            if abstract:
                quick_score = self._quick_score_abstract(abstract)
                paper['quick_score'] = quick_score

            scored_papers.append(paper)

        # Sort by quick score descending
        scored_papers.sort(key=lambda x: x.get('quick_score', {}).get('overall_score', 5), reverse=True)

        return scored_papers

    def _quick_score_abstract(self, abstract: str) -> Dict[str, Any]:
        """Quick scoring based on abstract only (faster than full paper)"""
        prompt = f"""Rate this paper abstract on innovation (1-10). Return JSON only.

Abstract: {abstract[:1000]}

{{"overall_score": <1-10>, "reason": "<one sentence>"}}"""

        try:
            response = self.llm.query(prompt, temperature=0.2, max_tokens=100, timeout=30)
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                return json.loads(match.group())
        except:
            pass

        return {"overall_score": 5, "reason": "Score unavailable"}