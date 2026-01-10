"""
agents/arxiv_agent.py - Downloads and analyzes papers from arXiv
"""

import os
import re
import arxiv
import fitz  # pymupdf


class ArxivAgent:
    """Downloads and analyzes papers from arXiv"""
    
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
    
    def analyze_paper(self, arxiv_id):
        """Download and analyze a paper"""
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
            
            # Store analysis in memory
            self.memory.add_node(f"analysis_{arxiv_id}", "paper_analysis", 
                               {"analysis": analysis, "pdf_path": pdf_path})
            self.memory.add_edge(f"paper_{arxiv_id}", f"analysis_{arxiv_id}", "analyzed_as")
            
            return analysis
        except RuntimeError as e:
            error_msg = str(e)
            if "crashed" in error_msg.lower() or "exit code" in error_msg.lower():
                return "LLM crashed (paper too large or model overloaded). Try a different paper or restart LM Studio."
            return f"Analysis failed: {e}"