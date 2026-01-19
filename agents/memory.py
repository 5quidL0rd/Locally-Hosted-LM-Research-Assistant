"""
agents/memory.py - Persistent knowledge graph (Memory Palace)
"""

import os
import json
from datetime import datetime
import networkx as nx
from pyvis.network import Network


class MemoryPalace:
    """
    Persistent knowledge graph that remembers research across sessions.
    """
    
    def __init__(self, storage_path="memory_palace.json"):
        self.storage_path = storage_path
        self.graph = nx.DiGraph()
        self.load()
    
    def load(self):
        """Load graph from disk"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    self.graph = nx.node_link_graph(data)
                print(f"[Memory Palace] Loaded {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
            except Exception as e:
                print(f"[Memory Palace] Failed to load: {e}")
        else:
            print("[Memory Palace] Starting fresh (no saved memory found)")
    
    def save(self):
        """Save graph to disk"""
        try:
            data = nx.node_link_data(self.graph)
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[Memory Palace] Failed to save: {e}")
    
    def add_node(self, node_id, node_type, data=None):
        """Add a concept/paper/dataset to the graph"""
        self.graph.add_node(node_id, type=node_type, data=data or {}, 
                           timestamp=datetime.now().isoformat())
        self.save()
    
    def add_edge(self, from_node, to_node, relationship):
        """Add relationship between nodes"""
        self.graph.add_edge(from_node, to_node, relationship=relationship,
                           timestamp=datetime.now().isoformat())
        self.save()
    
    def get_related(self, node_id, max_depth=2):
        """Get all nodes related to this one"""
        if node_id not in self.graph:
            return []
        
        related = []
        try:
            visited = set()
            queue = [(node_id, 0)]
            
            while queue:
                current, depth = queue.pop(0)
                if current in visited or depth > max_depth:
                    continue
                visited.add(current)
                
                if current != node_id:
                    related.append({
                        'node': current,
                        'depth': depth,
                        'data': self.graph.nodes[current]
                    })
                
                for neighbor in self.graph.neighbors(current):
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1))
        except Exception as e:
            print(f"[Memory Palace] Error getting related nodes: {e}")
        
        return related
    
    def visualize(self, output_path="knowledge_graph.html", open_browser=True):
        """Generate interactive visualization and optionally open in browser"""
        import webbrowser

        try:
            net = Network(height="750px", width="100%", directed=True,
                         bgcolor="#222222", font_color="white")

            # Add nodes with custom colors based on type
            color_map = {
                'arxiv_paper': '#4CAF50',      # Green
                'paper_analysis': '#2196F3',   # Blue
                'kaggle_dataset': '#FF9800',   # Orange
                'nn_experiment': '#9C27B0',    # Purple
                'web_search': '#00BCD4',       # Cyan
                'hf_model': '#E91E63',         # Pink
                'lit_review': '#FFEB3B',       # Yellow
            }

            for node, data in self.graph.nodes(data=True):
                node_type = data.get('type', 'unknown')
                node_data = data.get('data', {})

                # Create label and title (hover text)
                if node_type == 'arxiv_paper':
                    label = node_data.get('title', str(node))[:50] + '...' if len(node_data.get('title', str(node))) > 50 else node_data.get('title', str(node))
                    title = f"<b>{node_data.get('title', node)}</b><br>ArXiv: {node_data.get('arxiv_id', 'N/A')}<br>Authors: {', '.join(node_data.get('authors', [])[:3])}"
                elif node_type == 'kaggle_dataset':
                    label = node_data.get('title', str(node))[:40]
                    title = f"<b>{node_data.get('title', node)}</b><br>Ref: {node_data.get('ref', 'N/A')}<br>Downloads: {node_data.get('download_count', 0):,}"
                else:
                    label = str(node)[:40]
                    title = str(node)

                color = color_map.get(node_type, '#757575')
                net.add_node(node, label=label, title=title, color=color,
                           shape='dot' if node_type == 'arxiv_paper' else 'box')

            # Add edges
            for u, v, data in self.graph.edges(data=True):
                rel = data.get('relationship', 'related_to')
                net.add_edge(u, v, title=rel, label=rel[:15])

            # Configure physics for better layout
            net.set_options("""
            var options = {
              "nodes": {
                "font": {"size": 14}
              },
              "edges": {
                "color": {"inherit": true},
                "smooth": {"type": "continuous"}
              },
              "physics": {
                "forceAtlas2Based": {
                  "gravitationalConstant": -50,
                  "centralGravity": 0.01,
                  "springLength": 100,
                  "springConstant": 0.08
                },
                "minVelocity": 0.75,
                "solver": "forceAtlas2Based"
              }
            }
            """)

            # Get absolute path for browser
            abs_path = os.path.abspath(output_path)
            net.save_graph(abs_path)
            print(f"[Memory Palace] Visualization saved to {abs_path}")

            # Open in browser
            if open_browser:
                file_url = f"file:///{abs_path.replace(os.sep, '/')}"
                webbrowser.open(file_url)
                print(f"[Memory Palace] Opened in browser")

            return abs_path
        except Exception as e:
            print(f"[Memory Palace] Visualization failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def search(self, query):
        """Search for nodes matching query"""
        matches = []
        query_lower = query.lower()
        for node, data in self.graph.nodes(data=True):
            if query_lower in str(node).lower() or query_lower in str(data.get('data', {})).lower():
                matches.append({'node': node, 'data': data})
        return matches

    def get_recent_nodes(self, limit=10):
        """Get the most recently added nodes"""
        nodes_with_time = []
        for node, data in self.graph.nodes(data=True):
            timestamp = data.get('timestamp', '')
            nodes_with_time.append({
                'node': node,
                'data': data.get('data', {}),
                'type': data.get('type', 'unknown'),
                'timestamp': timestamp
            })

        # Sort by timestamp descending
        nodes_with_time.sort(key=lambda x: x['timestamp'] if x['timestamp'] else '', reverse=True)
        return nodes_with_time[:limit]

    def export_to_markdown(self, output_path: str = "research_notes.md",
                          include_sections: list = None) -> str:
        """
        Export the knowledge graph to a structured markdown document.

        Args:
            output_path: Path for the markdown file
            include_sections: List of node types to include (None = all)

        Returns:
            Path to the generated markdown file
        """
        lines = []
        lines.append("# Research Knowledge Base")
        lines.append(f"\n*Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        lines.append(f"\n**Total Items:** {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} connections\n")

        # Group nodes by type
        nodes_by_type = {}
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('type', 'unknown')
            # Ensure node_type is a string (handle malformed data)
            if not isinstance(node_type, str):
                node_type = str(node_type) if node_type is not None else 'unknown'
            if include_sections and node_type not in include_sections:
                continue
            if node_type not in nodes_by_type:
                nodes_by_type[node_type] = []
            nodes_by_type[node_type].append((node, data))

        # Section: ArXiv Papers
        if 'arxiv_paper' in nodes_by_type:
            lines.append("## ArXiv Papers\n")
            for node, data in nodes_by_type['arxiv_paper']:
                paper_data = data.get('data', {})
                lines.append(f"### {paper_data.get('title', node)}\n")
                lines.append(f"- **ArXiv ID:** `{paper_data.get('arxiv_id', 'N/A')}`")
                authors = paper_data.get('authors', [])
                if authors:
                    lines.append(f"- **Authors:** {', '.join(authors[:5])}")
                lines.append(f"- **Published:** {paper_data.get('published', 'N/A')}")
                if paper_data.get('pdf_url'):
                    lines.append(f"- **PDF:** [{paper_data['pdf_url']}]({paper_data['pdf_url']})")
                summary = paper_data.get('summary', '')
                if summary:
                    lines.append(f"\n> {summary[:500]}{'...' if len(summary) > 500 else ''}\n")
                lines.append("")

        # Section: Paper Analyses
        if 'paper_analysis' in nodes_by_type:
            lines.append("## Paper Analyses\n")
            for node, data in nodes_by_type['paper_analysis']:
                analysis_data = data.get('data', {})
                lines.append(f"### Analysis: {node}\n")
                analysis = analysis_data.get('analysis', '')
                if analysis:
                    lines.append(analysis)
                    lines.append("")
                if analysis_data.get('pdf_path'):
                    lines.append(f"*Source: `{analysis_data['pdf_path']}`*\n")

        # Section: Kaggle Datasets
        if 'kaggle_dataset' in nodes_by_type:
            lines.append("## Kaggle Datasets\n")
            lines.append("| Dataset | Reference | Downloads |")
            lines.append("|---------|-----------|-----------|")
            for node, data in nodes_by_type['kaggle_dataset']:
                ds_data = data.get('data', {})
                title = ds_data.get('title', node)
                ref = ds_data.get('ref', 'N/A')
                downloads = ds_data.get('download_count', 0)
                lines.append(f"| {title} | `{ref}` | {downloads:,} |")
            lines.append("")

        # Section: Neural Network Experiments
        if 'nn_experiment' in nodes_by_type:
            lines.append("## Neural Network Experiments\n")
            for node, data in nodes_by_type['nn_experiment']:
                exp_data = data.get('data', {})
                lines.append(f"### {node}\n")
                lines.append(f"- **Dataset:** `{exp_data.get('dataset', 'N/A')}`")
                lines.append(f"- **Task Type:** {exp_data.get('task_type', 'N/A')}")
                lines.append(f"- **Winner:** {exp_data.get('winner', 'N/A')}")
                lines.append(f"- **Best Score:** {exp_data.get('score', 'N/A')}")
                lines.append(f"- **Timestamp:** {data.get('timestamp', 'N/A')}")
                lines.append("")

        # Section: Web Search Results
        if 'web_search' in nodes_by_type:
            lines.append("## Web Searches\n")
            for node, data in nodes_by_type['web_search']:
                search_data = data.get('data', {})
                lines.append(f"- **Query:** {search_data.get('query', node)}")
                lines.append(f"  - Results: {search_data.get('result_count', 'N/A')}")
                lines.append("")

        # Section: HuggingFace Models
        if 'hf_model' in nodes_by_type:
            lines.append("## HuggingFace Models\n")
            lines.append("| Model ID | Task | Downloads |")
            lines.append("|----------|------|-----------|")
            for node, data in nodes_by_type['hf_model']:
                model_data = data.get('data', {})
                model_id = model_data.get('id', node)
                task = model_data.get('pipeline_tag', 'N/A')
                downloads = model_data.get('downloads', 0)
                lines.append(f"| `{model_id}` | {task} | {downloads:,} |")
            lines.append("")

        # Section: Other/Unknown types
        other_types = [t for t in nodes_by_type.keys()
                      if t not in ['arxiv_paper', 'paper_analysis', 'kaggle_dataset',
                                  'nn_experiment', 'web_search', 'hf_model']]
        if other_types:
            lines.append("## Other Items\n")
            for node_type in other_types:
                lines.append(f"### {node_type.replace('_', ' ').title()}\n")
                for node, data in nodes_by_type[node_type]:
                    lines.append(f"- {node}")
                lines.append("")

        # Section: Relationships
        lines.append("## Knowledge Connections\n")
        lines.append("```")
        edge_count = 0
        for u, v, data in self.graph.edges(data=True):
            if edge_count < 50:  # Limit to avoid huge files
                rel = data.get('relationship', 'related_to')
                lines.append(f"{u} --[{rel}]--> {v}")
                edge_count += 1
        if self.graph.number_of_edges() > 50:
            lines.append(f"... and {self.graph.number_of_edges() - 50} more connections")
        lines.append("```\n")

        # Write to file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            print(f"[Memory Palace] Exported to {output_path}")
            return output_path
        except Exception as e:
            print(f"[Memory Palace] Export failed: {e}")
            return None

    def export_citations_bibtex(self, output_path: str = "citations.bib") -> str:
        """
        Export ArXiv papers as BibTeX citations.

        Returns:
            Path to the generated .bib file
        """
        entries = []

        for node, data in self.graph.nodes(data=True):
            if data.get('type') != 'arxiv_paper':
                continue

            paper = data.get('data', {})
            arxiv_id = paper.get('arxiv_id', '').replace('.', '_').replace('/', '_')
            if not arxiv_id:
                continue

            title = paper.get('title', 'Unknown Title')
            authors = paper.get('authors', [])
            year = paper.get('published', '')[:4] if paper.get('published') else '2024'

            # Format authors for BibTeX
            author_str = ' and '.join(authors) if authors else 'Unknown'

            entry = f"""@article{{{arxiv_id},
    title = {{{title}}},
    author = {{{author_str}}},
    year = {{{year}}},
    eprint = {{{paper.get('arxiv_id', '')}}},
    archivePrefix = {{arXiv}},
    primaryClass = {{cs.LG}},
    url = {{{paper.get('pdf_url', '')}}}
}}"""
            entries.append(entry)

        if not entries:
            print("[Memory Palace] No ArXiv papers found for BibTeX export")
            return None

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n\n'.join(entries))
            print(f"[Memory Palace] Exported {len(entries)} citations to {output_path}")
            return output_path
        except Exception as e:
            print(f"[Memory Palace] BibTeX export failed: {e}")
            return None

    def get_summary_stats(self) -> dict:
        """Get summary statistics about the knowledge graph"""
        nodes_by_type = {}
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('type', 'unknown')
            nodes_by_type[node_type] = nodes_by_type.get(node_type, 0) + 1

        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'nodes_by_type': nodes_by_type,
        }
    



