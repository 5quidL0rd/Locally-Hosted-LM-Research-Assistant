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
    
    def visualize(self, output_path="knowledge_graph.html"):
        """Generate interactive visualization"""
        try:
            net = Network(height="750px", width="100%", directed=True)
            net.from_nx(self.graph)
            net.save_graph(output_path)
            print(f"[Memory Palace] Visualization saved to {output_path}")
            return output_path
        except Exception as e:
            print(f"[Memory Palace] Visualization failed: {e}")
            return None
    
    def search(self, query):
        """Search for nodes matching query"""
        matches = []
        query_lower = query.lower()
        for node, data in self.graph.nodes(data=True):
            if query_lower in str(node).lower() or query_lower in str(data.get('data', {})).lower():
                matches.append({'node': node, 'data': data})
        return matches
    



