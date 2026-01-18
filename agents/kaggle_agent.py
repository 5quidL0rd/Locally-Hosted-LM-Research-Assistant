# agents/kaggle_agent.py

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    KAGGLE_AVAILABLE = True
except Exception as e:
    KAGGLE_AVAILABLE = False
    print(f"[Kaggle] Not available: {e}")


class KaggleAgent:
    """Finds and downloads Kaggle datasets with improved search strategies"""

    # Common dataset categories for fallback searches
    CATEGORY_KEYWORDS = {
        'classification': ['classification', 'labeled', 'categories', 'classes'],
        'regression': ['regression', 'prediction', 'forecasting', 'price'],
        'nlp': ['text', 'nlp', 'sentiment', 'language', 'tweets', 'reviews'],
        'cv': ['image', 'images', 'vision', 'photos', 'pictures', 'visual'],
        'timeseries': ['time series', 'timeseries', 'stock', 'weather', 'temporal'],
        'tabular': ['tabular', 'csv', 'structured', 'features'],
    }

    # Popular/reliable datasets as fallbacks
    FALLBACK_DATASETS = {
        'classification': [
            'uciml/iris', 'uciml/adult-census-income',
            'rashikrahmanpritom/heart-attack-analysis-prediction-dataset'
        ],
        'regression': [
            'vikrishnan/boston-house-prices', 'mirichoi0218/insurance',
            'harlfoxem/housesalesprediction'
        ],
        'nlp': [
            'kazanova/sentiment140', 'snap/amazon-fine-food-reviews',
            'crowdflower/twitter-airline-sentiment'
        ],
        'cv': [
            'moltean/fruits', 'puneet6060/intel-image-classification',
            'paultimothymooney/chest-xray-pneumonia'
        ],
        'timeseries': [
            'szrlee/stock-time-series-20050101-to-20171231',
            'rakannimer/air-passengers'
        ],
        'general': [
            'titanic', 'heptapod/titanic', 'datasnaek/youtube-new'
        ]
    }

    def __init__(self, memory_palace=None, download_dir="kaggle_datasets"):
        self.memory = memory_palace
        self.download_dir = download_dir
        os.makedirs(download_dir, exist_ok=True)

        if KAGGLE_AVAILABLE:
            try:
                self.api = KaggleApi()
                self.api.authenticate()
                print("[Kaggle] Authenticated successfully")
            except Exception as e:
                print(f"[Kaggle] Authentication failed: {e}")
                self.api = None
        else:
            self.api = None

    def _clean_query(self, query: str) -> str:
        """Clean and normalize search query"""
        # Remove common words that confuse the Kaggle API
        stopwords = [
            'dataset', 'datasets', 'data', 'find', 'search', 'kaggle',
            'download', 'get', 'for', 'the', 'a', 'an', 'with', 'about',
            'related', 'to', 'on', 'in', 'please', 'can', 'you', 'i', 'want'
        ]

        words = query.lower().split()
        cleaned = [w for w in words if w not in stopwords and len(w) > 2]

        return ' '.join(cleaned) if cleaned else query

    def _detect_category(self, query: str) -> Optional[str]:
        """Detect dataset category from query"""
        query_lower = query.lower()

        for category, keywords in self.CATEGORY_KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                return category

        return None

    def _generate_query_variations(self, query: str) -> List[str]:
        """Generate multiple query variations to improve search results"""
        cleaned = self._clean_query(query)
        variations = [cleaned]

        # Split into individual keywords
        words = cleaned.split()
        if len(words) > 1:
            # Try each word individually
            for word in words:
                if len(word) > 3:
                    variations.append(word)

            # Try pairs of words
            if len(words) >= 2:
                variations.append(f"{words[0]} {words[1]}")

        # Add synonyms for common terms
        synonyms = {
            'car': ['automobile', 'vehicle'],
            'house': ['housing', 'real estate', 'home'],
            'stock': ['finance', 'market', 'trading'],
            'health': ['medical', 'healthcare', 'disease'],
            'movie': ['film', 'cinema', 'imdb'],
            'music': ['song', 'spotify', 'audio'],
            'weather': ['climate', 'temperature', 'meteorological'],
            'sales': ['revenue', 'retail', 'e-commerce'],
        }

        for word in words:
            if word in synonyms:
                variations.extend(synonyms[word])

        return list(dict.fromkeys(variations))  # Remove duplicates, preserve order

    def search_datasets(self, query: str, max_results: int = 8) -> List[Dict[str, Any]]:
        """
        Search Kaggle datasets with improved strategies:
        1. Clean and normalize the query
        2. Try multiple query variations
        3. Sort by download count for quality
        4. Fall back to category-based suggestions
        """
        if not self.api:
            return [{"error": "Kaggle API not available. Check your kaggle.json credentials."}]

        print(f"[Kaggle] Searching for: '{query}'")
        all_results = {}

        # Generate query variations
        query_variations = self._generate_query_variations(query)
        print(f"[Kaggle] Trying variations: {query_variations[:5]}")

        # Try each variation
        for variation in query_variations[:5]:  # Limit to 5 variations
            try:
                datasets = list(self.api.dataset_list(
                    search=variation,
                    sort_by='downloadCount'  # Sort by popularity
                ))[:max_results * 2]

                for ds in datasets:
                    ref = ds.ref
                    if ref not in all_results:
                        download_count = getattr(ds, 'downloadCount', 0)
                        # Only include datasets with reasonable download counts
                        if download_count > 10:  # Filter out very obscure datasets
                            all_results[ref] = {
                                "ref": ref,
                                "title": ds.title,
                                "size": getattr(ds, 'size', 'Unknown'),
                                "download_count": download_count,
                                "vote_count": getattr(ds, 'voteCount', 0),
                                "last_updated": str(getattr(ds, 'lastUpdated', 'Unknown')),
                                "url": f"https://www.kaggle.com/datasets/{ref}",
                                "usability_rating": getattr(ds, 'usabilityRating', 0)
                            }

            except Exception as e:
                print(f"[Kaggle] Search variation '{variation}' failed: {e}")
                continue

        # Sort by download count (popularity as proxy for quality)
        results = sorted(
            all_results.values(),
            key=lambda x: (x.get('download_count', 0), x.get('usability_rating', 0)),
            reverse=True
        )[:max_results]

        # If we found results, return them
        if results:
            print(f"[Kaggle] Found {len(results)} datasets")

            # Add to memory
            if self.memory:
                for ds in results:
                    try:
                        node_id = f"kaggle_dataset_{ds['ref'].replace('/', '_')}"
                        self.memory.add_node(node_id, "kaggle_dataset", ds)
                        self.memory.add_edge(query, node_id, "search_result")
                    except:
                        pass

            return results

        # Fallback: suggest related datasets based on category
        print(f"[Kaggle] No direct results, trying category fallback...")
        category = self._detect_category(query)

        if category and category in self.FALLBACK_DATASETS:
            return self._get_fallback_datasets(category, query)

        # Ultimate fallback: popular general datasets
        return self._get_fallback_datasets('general', query)

    def _get_fallback_datasets(self, category: str, original_query: str) -> List[Dict[str, Any]]:
        """Get fallback datasets from a category"""
        fallback_refs = self.FALLBACK_DATASETS.get(category, self.FALLBACK_DATASETS['general'])
        results = []

        print(f"[Kaggle] Suggesting {category} datasets as fallback")

        for ref in fallback_refs[:5]:
            try:
                # Get actual dataset info
                datasets = list(self.api.dataset_list(search=ref))[:1]
                if datasets:
                    ds = datasets[0]
                    results.append({
                        "ref": ds.ref,
                        "title": ds.title,
                        "size": getattr(ds, 'size', 'Unknown'),
                        "download_count": getattr(ds, 'downloadCount', 0),
                        "vote_count": getattr(ds, 'voteCount', 0),
                        "last_updated": str(getattr(ds, 'lastUpdated', 'Unknown')),
                        "url": f"https://www.kaggle.com/datasets/{ds.ref}",
                        "suggested": True,
                        "suggestion_reason": f"Popular {category} dataset (no exact matches for '{original_query}')"
                    })
            except:
                continue

        if results:
            results[0]['note'] = f"No exact matches for '{original_query}'. Showing popular {category} datasets instead."

        return results if results else [{"error": f"No datasets found for '{original_query}'. Try simpler search terms."}]

    def search_by_tag(self, tag: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search datasets by tag/category"""
        if not self.api:
            return [{"error": "Kaggle API not available"}]

        try:
            datasets = list(self.api.dataset_list(tag_ids=tag))[:max_results]
            return [{
                "ref": ds.ref,
                "title": ds.title,
                "size": getattr(ds, 'size', 'Unknown'),
                "download_count": getattr(ds, 'downloadCount', 0),
                "url": f"https://www.kaggle.com/datasets/{ds.ref}"
            } for ds in datasets]
        except Exception as e:
            return [{"error": f"Tag search failed: {e}"}]

    def get_popular_datasets(self, max_results: int = 10) -> List[Dict[str, Any]]:
        """Get most popular datasets overall"""
        if not self.api:
            return [{"error": "Kaggle API not available"}]

        try:
            datasets = list(self.api.dataset_list(sort_by='downloadCount'))[:max_results]
            return [{
                "ref": ds.ref,
                "title": ds.title,
                "size": getattr(ds, 'size', 'Unknown'),
                "download_count": getattr(ds, 'downloadCount', 0),
                "vote_count": getattr(ds, 'voteCount', 0),
                "url": f"https://www.kaggle.com/datasets/{ds.ref}"
            } for ds in datasets]
        except Exception as e:
            return [{"error": f"Failed to get popular datasets: {e}"}]

    def download_dataset(self, dataset_ref: str) -> Optional[Dict[str, Any]]:
        """Download a dataset with better error handling"""
        if not self.api:
            return {"error": "Kaggle API not available"}

        try:
            dataset_dir = os.path.join(
                self.download_dir, dataset_ref.replace("/", "_")
            )
            os.makedirs(dataset_dir, exist_ok=True)

            print(f"[Kaggle] Downloading: {dataset_ref}")

            # Check if already downloaded
            existing_files = list(Path(dataset_dir).rglob("*"))
            existing_files = [f for f in existing_files if f.is_file()]

            if existing_files:
                print(f"[Kaggle] Dataset already exists with {len(existing_files)} files")
                return {
                    "dataset_ref": dataset_ref,
                    "path": dataset_dir,
                    "files": [str(f) for f in existing_files],
                    "cached": True
                }

            # Download fresh
            self.api.dataset_download_files(
                dataset_ref,
                path=dataset_dir,
                unzip=True,
                quiet=False
            )

            files = [str(p) for p in Path(dataset_dir).rglob("*") if p.is_file()]

            # Categorize files
            file_info = self._categorize_files(files)

            result = {
                "dataset_ref": dataset_ref,
                "path": dataset_dir,
                "files": files,
                "file_summary": file_info,
                "cached": False
            }

            # Add to memory
            if self.memory:
                try:
                    node_id = f"downloaded_kaggle_{dataset_ref.replace('/', '_')}"
                    self.memory.add_node(node_id, "downloaded_dataset", result)
                except:
                    pass

            return result

        except Exception as e:
            error_msg = str(e)
            print(f"[Kaggle] Download failed: {error_msg}")

            # Provide helpful error messages
            if "404" in error_msg or "not found" in error_msg.lower():
                return {"error": f"Dataset '{dataset_ref}' not found. Check the dataset reference."}
            elif "403" in error_msg or "forbidden" in error_msg.lower():
                return {"error": "Access denied. This dataset may require accepting terms on Kaggle."}
            else:
                return {"error": f"Download failed: {error_msg}"}

    def _categorize_files(self, files: List[str]) -> Dict[str, Any]:
        """Categorize downloaded files by type"""
        categories = {
            "csv": [],
            "images": [],
            "json": [],
            "text": [],
            "other": []
        }

        image_ext = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff')
        text_ext = ('.txt', '.md', '.rst')

        for f in files:
            f_lower = f.lower()
            if f_lower.endswith('.csv'):
                categories["csv"].append(f)
            elif f_lower.endswith(image_ext):
                categories["images"].append(f)
            elif f_lower.endswith('.json'):
                categories["json"].append(f)
            elif f_lower.endswith(text_ext):
                categories["text"].append(f)
            else:
                categories["other"].append(f)

        return {
            "csv_count": len(categories["csv"]),
            "image_count": len(categories["images"]),
            "json_count": len(categories["json"]),
            "primary_type": max(categories.keys(), key=lambda k: len(categories[k])) if files else "unknown",
            "csv_files": categories["csv"][:5],  # List first 5 CSVs
        }

    def recommend_dataset(self, task_type: str, domain: str = None) -> List[Dict[str, Any]]:
        """Recommend datasets based on task type and optional domain"""
        if not self.api:
            return [{"error": "Kaggle API not available"}]

        # Build search query
        queries = []

        if task_type == 'classification':
            queries = ['classification labeled', 'binary classification', 'multiclass']
        elif task_type == 'regression':
            queries = ['regression prediction', 'price prediction', 'continuous target']
        elif task_type == 'clustering':
            queries = ['clustering segmentation', 'customer segmentation']
        elif task_type == 'nlp':
            queries = ['text classification', 'sentiment analysis', 'nlp dataset']
        elif task_type == 'image':
            queries = ['image classification', 'computer vision', 'object detection']
        elif task_type == 'timeseries':
            queries = ['time series', 'forecasting', 'temporal data']
        else:
            queries = [task_type]

        if domain:
            queries = [f"{domain} {q}" for q in queries]

        # Search with first query
        return self.search_datasets(queries[0], max_results=5)
