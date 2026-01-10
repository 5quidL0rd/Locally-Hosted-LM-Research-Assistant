from ddgs import DDGS
import re

class SearchAgent:
    """Performs structured web searches."""

    def run_task(self, query, max_results=5):
        """General-purpose web search via DuckDuckGo."""
        results = []

        try:
            with DDGS() as ddgs:
                raw = list(ddgs.text(query, max_results=max_results * 2))

                for r in raw:
                    url = r.get("href")
                    if not url:
                        continue

                    results.append({
                        "title": r.get("title", "No title"),
                        "url": url,
                        "description": (r.get("body") or "")[:200]
                    })

                    if len(results) >= max_results:
                        break

        except Exception as e:
            return [{"error": str(e)}]

        return results
    
    def search_github(self, query, max_results=1):
        with DDGS() as ddgs:
            raw = ddgs.text(f"site:github.com {query}", max_results=max_results*2)
            return self._parse_results(raw, max_results)
    
    def search_youtube(self, query, max_results=1):
        with DDGS() as ddgs:
            raw = ddgs.text(f"site:youtube.com {query}", max_results=max_results*2)
            return self._parse_results(raw, max_results)
    
    def search_blogs(self, query, max_results=1):
        """General blog search - kept for backwards compatibility."""
        with DDGS() as ddgs:
            raw = ddgs.text(query, max_results=max_results*2)
            return self._parse_results(raw, max_results)
    
    def search_anthropic_research(self, max_results=5):
        """
        Specifically search Anthropic's research page.
        This is what the Secretary needs!
        """
        print("[SearchAgent] Searching Anthropic research...")
        results = []
        
        try:
            with DDGS() as ddgs:
                # Search for latest Anthropic research articles
                queries = [
                    "site:anthropic.com/research 2025",
                    "site:anthropic.com/research 2024",
                    "site:anthropic.com/research alignment",
                    "site:anthropic.com/research interpretability",
                ]
                
                seen_urls = set()
                for query in queries:
                    try:
                        print(f"[SearchAgent]   Trying query: {query}")
                        raw = list(ddgs.text(query, max_results=10))
                        print(f"[SearchAgent]   Got {len(raw)} raw results")
                        
                        for r in raw:
                            url = r.get('href', '')
                            # Only include actual research articles
                            if url and 'anthropic.com/research' in url and url not in seen_urls:
                                # Skip the main research page, only get articles
                                if url != "https://www.anthropic.com/research" and "/research/" in url:
                                    seen_urls.add(url)
                                    title = r.get('title', 'Anthropic Research Article')
                                    body = r.get('body', '') or r.get('description', '')
                                    results.append({
                                        "title": title,
                                        "url": url,
                                        "description": body[:200] if body else "Research from Anthropic"
                                    })
                                    print(f"[SearchAgent]     Added: {title[:50]}...")
                            
                            if len(results) >= max_results:
                                break
                    except Exception as e:
                        print(f"[SearchAgent]   Error with query '{query}': {e}")
                        continue
                    
                    if len(results) >= max_results:
                        break
        except Exception as e:
            print(f"[SearchAgent] Fatal error in search_anthropic_research: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"[SearchAgent] Found {len(results)} Anthropic research articles")
        return results
    
    def search_openai_research(self, max_results=5):
        """
        Specifically search OpenAI's research page.
        This is what the Secretary needs!
        """
        print("[SearchAgent] Searching OpenAI research...")
        results = []
        
        try:
            with DDGS() as ddgs:
                # Search for latest OpenAI research articles
                queries = [
                    "site:openai.com/index 2025",
                    "site:openai.com/index 2024", 
                    "site:openai.com/research GPT",
                    "site:openai.com/research o1",
                ]
                
                seen_urls = set()
                for query in queries:
                    try:
                        print(f"[SearchAgent]   Trying query: {query}")
                        raw = list(ddgs.text(query, max_results=10))
                        print(f"[SearchAgent]   Got {len(raw)} raw results")
                        
                        for r in raw:
                            url = r.get('href', '')
                            # Include research and index pages
                            if url and 'openai.com' in url and url not in seen_urls:
                                # Look for actual research articles
                                if any(path in url for path in ['/research/', '/index/']):
                                    seen_urls.add(url)
                                    title = r.get('title', 'OpenAI Research Article')
                                    body = r.get('body', '') or r.get('description', '')
                                    results.append({
                                        "title": title,
                                        "url": url,
                                        "description": body[:200] if body else "Research from OpenAI"
                                    })
                                    print(f"[SearchAgent]     Added: {title[:50]}...")
                            
                            if len(results) >= max_results:
                                break
                    except Exception as e:
                        print(f"[SearchAgent]   Error with query '{query}': {e}")
                        continue
                    
                    if len(results) >= max_results:
                        break
        except Exception as e:
            print(f"[SearchAgent] Fatal error in search_openai_research: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"[SearchAgent] Found {len(results)} OpenAI research articles")
        return results