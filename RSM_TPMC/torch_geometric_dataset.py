import os
import subprocess
import json
import torch
from torch_geometric.datasets import GeometricShapes
from torch_geometric.transforms import SamplePoints
import trimesh
from torch_geometric.data import Data
import numpy as np

# Load and preprocess the dataset
dataset = GeometricShapes(root='data/GeometricShapes')

# Directory to save intermediate files and results
output_dir = "tempfiles/Mesh_Files"
os.makedirs(output_dir, exist_ok=True)


def save_graph_as_stl(graph, file_path):
    """
    Convert a PyTorch Geometric graph to an STL file.
    This needs to be implemented according to your graph representation.
    """
    # Example placeholder function (to be customized):
    mesh = trimesh.Trimesh(vertices=graph.pos.numpy(), faces=graph.face.T.numpy())
    mesh.fill_holes()

    ascii_stl = trimesh.exchange.stl.export_stl_ascii(mesh)

    # Write the ASCII STL string to a file
    with open(file_path, 'w') as file:
        file.write(ascii_stl)


def calculate_drag_samples(graph, output_dir):
    # 1) Export the graph to STL
    stl_file_path = os.path.join(output_dir, f"graph_{graph.index}.stl")
    save_graph_as_stl(graph, stl_file_path)

    # 2) Run RSM+TPM
    result_file = os.path.join(output_dir, f"graph_{graph.index}_results.dat")
    subprocess.run([
        "python", "RSM_TPMC/rsm_run_script_new.py",
        "--tpm", "RSM_TPMC/tpm/build/src/tpm",
        "--input", stl_file_path,
        "--output", result_file
    ], check=True)

    # 3) Load N rows of 8 columns: first 5 = features, next 2 = orientation, last 1 = drag
    data_arr = np.loadtxt(result_file)  # shape [N, 8]

    # 4) Create one new Data object per row, repeating the 5 features for each node
    new_samples = []
    for row in data_arr:
        feat = torch.tensor(np.tile(row[:5], (graph.num_nodes, 1)), dtype=torch.float)
        orientation = torch.tensor(row[5:7], dtype=torch.float)  # shape [2]
        drag = torch.tensor([row[7]], dtype=torch.float)

        # Create a new data object sharing geometry with the original graph
        new_data = Data(
            x=feat,
            edge_index=graph.edge_index,
            pos=graph.pos if hasattr(graph, 'pos') else None
        )
        # Attach orientation and label
        new_data.orientation = orientation
        new_data.y = drag
        new_samples.append(new_data)

    return new_samples

if __name__ == "__main__":
    new_dataset = []
    for i, graph in enumerate(dataset):
        print(f"Processing graph {i+1}/{len(dataset)}...")
        graph.index = i
        try:
            samples = calculate_drag_samples(graph, output_dir)
            new_dataset.extend(samples)
            print(f"Graph {i} processed. Total samples: {len(new_dataset)}")
        except Exception as e:
            print(f"Error processing graph {i}: {e}")

        if i == 10:
            break

    torch.save(new_dataset, "geometric_shapes_with_drag.pt")
    print("Dataset with drag samples saved as 'geometric_shapes_with_drag.pt'")
