# agents/kaggle_agent.py

import os
from pathlib import Path

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    KAGGLE_AVAILABLE = True
except Exception as e:
    KAGGLE_AVAILABLE = False
    print(f"[Kaggle] Not available: {e}")


class KaggleAgent:
    """Finds and downloads Kaggle datasets"""

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

    def search_datasets(self, query, max_results=5):
            if not self.api:
                return [{"error": "Kaggle API not available"}]

            try:
                # Fetch the list of datasets based on the query
                datasets = list(self.api.dataset_list(search=query))[:max_results]
                results = []

                for ds in datasets:
                    # We use getattr to prevent AttributeErrors if the API response structure changes
                    info = {
                        "ref": ds.ref,
                        "title": ds.title,
                        "size": getattr(ds, 'size', 'Unknown'),
                        "download_count": getattr(ds, 'downloadCount', 0),
                        "vote_count": getattr(ds, 'voteCount', 0),
                        "last_updated": str(getattr(ds, 'lastUpdated', 'Unknown')),
                        "url": f"https://www.kaggle.com/datasets/{ds.ref}"
                    }
                    results.append(info)

                return results
            except Exception as e:
                return [{"error": f"Kaggle search failed: {str(e)}"}]

    def download_dataset(self, dataset_ref):
        if not self.api:
            return None

        try:
            dataset_dir = os.path.join(
                self.download_dir, dataset_ref.replace("/", "_")
            )
            os.makedirs(dataset_dir, exist_ok=True)

            # ⚠️ Kaggle API always re-downloads unless told otherwise
            self.api.dataset_download_files(
                dataset_ref,
                path=dataset_dir,
                unzip=True,
                quiet=False
            )

            files = [str(p) for p in Path(dataset_dir).rglob("*") if p.is_file()]

            return {
                "dataset_ref": dataset_ref,
                "path": dataset_dir,
                "files": files
            }

        except Exception as e:
            print(f"[Kaggle] Download failed: {e}")
            return None
