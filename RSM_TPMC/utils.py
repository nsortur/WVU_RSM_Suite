import torch
from torch_geometric.data import Data
import numpy as np
import trimesh
from torch_geometric.datasets import GeometricShapes
from torch_geometric.transforms import SamplePoints
# Load and preprocess the dataset
dataset = GeometricShapes(root='data/GeometricShapes')

# Use a sample from the dataset
graph = dataset[30]
print(graph.pos)

# Create a Trimesh object
mesh = trimesh.Trimesh(vertices=graph.pos.numpy(), faces=graph.face.T.numpy())
mesh.fill_holes()

# Export to STL file
mesh.export('graph_mesh.stl')

print("STL file has been created: 'graph_mesh.stl'")
