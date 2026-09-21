"""
Network Graph module.

Parses SUMO road networks (.net.xml) and constructs graph representation
for pathfinding, adjacency queries, and routing optimization.
"""

from typing import Dict, List, Tuple, Optional
import xml.etree.ElementTree as ET


class NetworkGraph:
    """
    Graph representation of the SUMO road network.
    """

    def __init__(self, net_file: str):
        self.net_file = net_file
        self.edges: Dict[str, Dict[str, Any]] = {}
        self.adjacency: Dict[str, List[str]] = {}
        self.load_network()

    def load_network(self):
        """Parse .net.xml and extract edges, lanes, and connectivity."""
        tree = ET.parse(self.net_file)
        root = tree.getroot()

        for edge in root.findall("edge"):
            edge_id = edge.get("id")
            if edge_id and not edge_id.startswith(":"):  # Skip internal junction edges
                self.edges[edge_id] = {
                    "from": edge.get("from"),
                    "to": edge.get("to"),
                    "length": float(edge.find("lane").get("length", 0)) if edge.find("lane") is not None else 0.0,
                    "speed": float(edge.find("lane").get("speed", 13.89)) if edge.find("lane") is not None else 13.89,
                }
                from_node = edge.get("from")
                if from_node not in self.adjacency:
                    self.adjacency[from_node] = []
                self.adjacency[from_node].append(edge_id)
