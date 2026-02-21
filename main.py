import os
import json
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import pathspec
from sklearn.cluster import HDBSCAN
from umap import UMAP
import typer

import tree_sitter
import tree_sitter_python
import tree_sitter_rust
import tree_sitter_c
import tree_sitter_cpp

EMBEDDING_API_URL = "http://localhost:7997/embeddings"
MODEL_ID = "Salesforce/SFR-Embedding-Code-400M_R"
BATCH_SIZE = 32

app = typer.Typer()

def get_parsers():
    return {
        '.py': tree_sitter.Parser(tree_sitter.Language(tree_sitter_python.language())),
        '.rs': tree_sitter.Parser(tree_sitter.Language(tree_sitter_rust.language())),
        '.c': tree_sitter.Parser(tree_sitter.Language(tree_sitter_c.language())),
        '.h': tree_sitter.Parser(tree_sitter.Language(tree_sitter_c.language())),
        '.cpp': tree_sitter.Parser(tree_sitter.Language(tree_sitter_cpp.language())),
        '.hpp': tree_sitter.Parser(tree_sitter.Language(tree_sitter_cpp.language())),
        '.cu': tree_sitter.Parser(tree_sitter.Language(tree_sitter_cpp.language())),
        '.cuh': tree_sitter.Parser(tree_sitter.Language(tree_sitter_cpp.language())),
    }

FUNCTION_NODE_TYPES = {
    'function_definition',
    'function_item',
    'function_declaration',
    'method_definition',
    'method_declaration',
}

def is_hidden(path: Path, repo_root: Path) -> bool:
    try:
        relative = path.relative_to(repo_root)
    except ValueError:
        return False
        
    for part in relative.parts:
        if part.startswith('.'):
            return True
    return False

def extract_nodes(node, source_bytes, is_root=True):
    units = []

    if not is_root and node.type in FUNCTION_NODE_TYPES:
        try:
            text = node.text.decode('utf-8', errors='replace')
        except Exception:
            text = source_bytes[node.start_byte:node.end_byte].decode('utf-8', errors='replace')
            
        units.append({
            'type': node.type,
            'start_point': [node.start_point.row, node.start_point.column],
            'end_point': [node.end_point.row, node.end_point.column],
            'start_byte': node.start_byte,
            'end_byte': node.end_byte,
            'text': text
        })

    for child in node.children:
        units.extend(extract_nodes(child, source_bytes, is_root=False))

    return units

def get_embeddings(texts):
    print(f"Requesting embeddings for {len(texts)} chunks...")
    all_embeddings = []
    
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        
        response = requests.post(EMBEDDING_API_URL, json={
            "input": batch,
            "model": MODEL_ID
        })
        
        if response.status_code == 200:
            data = response.json()
            batch_embeddings = [item['embedding'] for item in data['data']]
            all_embeddings.extend(batch_embeddings)
            print(f"  Processed [{i + len(batch)}/{len(texts)}]...")
        else:
            print(f"Error calling embedding API: {response.text}")
            all_embeddings.extend([[0.0] * 3200] * len(batch))
            
    return np.array(all_embeddings)


def reduce_and_cluster(embeddings, umap_dim: int):
    """Single UMAP projection, then HDBSCAN on the same coordinates."""
    print(f"Projecting embeddings to {umap_dim}D via UMAP (cosine)...")
    reducer = UMAP(n_components=umap_dim, metric='cosine')
    coords = reducer.fit_transform(embeddings)

    print(f"Clustering {embeddings.shape[0]} points in {umap_dim}D using HDBSCAN...")
    clusterer = HDBSCAN(min_cluster_size=3, min_samples=2, metric='euclidean')
    cluster_labels = clusterer.fit_predict(coords)
    
    unique_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"Found {unique_clusters} tight clusters (and some noise labels).")
    
    return coords, cluster_labels.tolist()


@app.command()
def main(
    repo: str = typer.Argument(..., help="Path to the repository"),
    output: str = typer.Option("clustered_index.json", help="Output JSON file path"),
    umap_dim: int = typer.Option(3, "--umap-dim", help="UMAP output dimensionality (3 = directly visualizable)"),
):
    """Parse repo into tree-sitter units & cluster them via embeddings."""
    root = Path(repo).resolve()
    if not root.exists() or not root.is_dir():
        print(f"Error: {root} is not a valid directory.")
        raise typer.Exit(1)

    gitignore_path = root / '.gitignore'
    spec = None
    if gitignore_path.exists():
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            spec = pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, f)
            
    parsers = get_parsers()
    supported_extensions = set(parsers.keys())
    
    all_files_units = []
    flattened_units = []

    for path in root.rglob('*'):
        if not path.is_file():
            continue
        if is_hidden(path, root):
            continue
        if spec:
            try:
                rel_path = str(path.relative_to(root).as_posix())
                if spec.match_file(rel_path):
                    continue
            except ValueError:
                pass
                
        ext = path.suffix.lower()
        if ext not in supported_extensions:
            continue
            
        parser = parsers[ext]
        try:
            source_bytes = path.read_bytes()
        except:
            continue
            
        try:
            tree = parser.parse(source_bytes)
            file_units = extract_nodes(tree.root_node, source_bytes, is_root=True)
            
            if file_units:
                file_map = {
                    'file': str(path.relative_to(root)),
                    'language': ext.lstrip('.'),
                    'units': file_units
                }
                all_files_units.append(file_map)
                
                for u in file_units:
                    flattened_units.append(u)
                    
        except Exception:
            pass

    print(f"Extracted {len(flattened_units)} functional nodes from {len(all_files_units)} files.")
    
    if not flattened_units:
        print("No units found to embed.")
        raise typer.Exit(1)

    # 1. Batch API Requests for Embeddings
    texts = [u['text'] for u in flattened_units]
    embeddings = get_embeddings(texts)
    
    # 2. Single UMAP projection + HDBSCAN on the same coordinates
    coords, cluster_labels = reduce_and_cluster(embeddings, umap_dim)
    
    # 3. Zip back into the structured dict payload
    for unit, emb, coord, label in zip(flattened_units, embeddings, coords, cluster_labels):
        unit['cluster_id'] = label
        unit['embedding'] = emb.tolist()
        unit['umap_coords'] = coord.tolist()
        
    # Write to output json
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(all_files_units, f, indent=2)
        
    print(f"Saved clustered results to {output}")

if __name__ == '__main__':
    app()
