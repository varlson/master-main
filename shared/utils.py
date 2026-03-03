


# ============================================
# UTILS
# ============================================

"""
DCRNN com MLflow - Versão Otimizada
Inclui: tracking de experimentos, grid search, early stopping
"""

import gdown
import numpy as np
import networkx as nx
import matplotlib.pylab as plt
from torch.utils.data import  Dataset
import pickle
from pathlib import Path
import torch
import h5py
from typing import Optional
import os
import requests
import shutil
import zipfile
import itertools


def show_graph(
    graphs,
    titles=None,
    figsize=(8, 6),
    node_size=10,
    alpha=0.1,
    fig_shape=None,
    pos=None
):
    # Caso 1: grafo único
    if not isinstance(graphs, list):
        fig, ax = plt.subplots(figsize=figsize)
    
        pos_ = nx.spring_layout(graphs, seed=42) if pos is None else pos
        
        # Usar ax para desenhar
        nx.draw_networkx_nodes(graphs, pos_, node_size=node_size, ax=ax)
        nx.draw_networkx_edges(graphs, pos_, alpha=alpha, ax=ax)
        
        # Adicionar informações do grafo
        n_nodes = graphs.number_of_nodes()
        n_edges = graphs.number_of_edges()
        ax.text(0.5, -0.1, f'Nós: {n_nodes} | Arestas: {n_edges}', 
                ha='center', transform=ax.transAxes, fontsize=9)
        
        # Ajustar layout para acomodar o texto
        plt.tight_layout()
        
        if titles is not None:
            plt.title(titles)
        
        plt.show()
        return

    if len(graphs) == 1:
        show_graph(graphs[0], titles, figsize, node_size, alpha, pos=pos)
        return

    if fig_shape is None:
        raise ValueError("fig_shape deve ser uma tupla, ex: (2, 2)")

    if titles is not None and len(graphs) != len(titles):
        raise ValueError("graphs e titles devem ter o mesmo tamanho")

    rows, cols = fig_shape

    if rows * cols != len(graphs):
        raise ValueError("fig_shape não corresponde ao número de grafos")

    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    axs = np.array(axs).reshape(rows, cols)

    # Correção: converter para array de objetos antes de reshape
    # gs = np.array(graphs, dtype=object).reshape(fig_shape)
    tls = np.array(titles, dtype=object).reshape(fig_shape) if titles is not None else None

    index=0
    for i in range(rows):
        for j in range(cols):
            g = graphs[index]
            ax = axs[i, j]
            index+=1

            pos_ = nx.spring_layout(g, seed=42) if pos is None else pos
            nx.draw_networkx_nodes(g, pos_, node_size=node_size, ax=ax)
            nx.draw_networkx_edges(g, pos_, alpha=alpha, ax=ax)

            if tls is not None:
                ax.set_title(tls[i, j])
                
            n_nodes = g.number_of_nodes()
            n_edges = g.number_of_edges()
            ax.text(0.5, -0.1, f'Nós: {n_nodes} | Arestas: {n_edges}', 
                       ha='center', transform=ax.transAxes, fontsize=9)


    plt.tight_layout()
    plt.show()


def filterdH5Data( nodesToKeep, filepath: str, key: str = "df") -> np.ndarray:
    with h5py.File(f"../dataset/h5/metr-la.h5", 'r') as f:
        if key not in f.keys():
            available_keys = list(f.keys())
            raise KeyError(f"Chave '{key}' não encontrada. Chaves disponíveis: {available_keys}")
    data = np.array(f["df"]['block0_values'])
    
    size = len([node for node in nodesToKeep if node])
    newData = np.empty((data.shape[0], size))
    
    index = 0
    for i, node_keep in enumerate(nodesToKeep):
        if node_keep:
            newData[index] = data[i][nodesToKeep]
            index += 1
    return newData

    



def save_graph_to_adjmx_nx(graph: nx.Graph, path: str) -> None:
    """
    Salva um grafo NetworkX como matriz de adjacência no formato .pkl
    Formato: tuple(data1, data2, adj_matrix) mantendo compatibilidade
    """
    # Converte grafo para matriz de adjacência numpy
    adj = nx.to_numpy_array(graph)
    
    # Salva no mesmo formato que a função original espera (tupla de 3 elementos)
    with open(path, "wb") as f:
        pickle.dump((None, None, adj), f)



def get_nodes_ids(data, key='df'):
    with h5py.File(f"../dataset/h5/{data}.h5", "r") as f:
        ids = np.array(f[key]['axis0']).astype(float)
    return ids


    
    
def save_filtered_graph(original_graph: nx.Graph, 
                        filtered_graph: nx.Graph,
                        filtered_graph_out_name: str,
                        output_path: str,
                        
                        weight: Optional[str] = 'weight') -> None:
  
    # Criar mapeamento de nós do grafo original
    node_mapping = {node: idx for idx, node in enumerate(original_graph.nodes())}
    n = len(node_mapping)
    
    # Criar matriz de adjacência do tamanho do grafo original
    adj = np.zeros((n, n))
    
    # Preencher apenas as arestas que existem no grafo filtrado
    for u, v in filtered_graph.edges():
        if u in node_mapping and v in node_mapping:
            i, j = node_mapping[u], node_mapping[v]
            edge_data = filtered_graph[u][v]
            w = edge_data.get(weight, 1.0) if weight else 1.0
            adj[i, j] = w
            
            # Se não-direcionado, preencher simetricamente
            if not filtered_graph.is_directed():
                adj[j, i] = w
    
    # Salvar com metadados
    path = Path(output_path)
    ext = path.suffix.lower()
    
    if ext == ".npz":
        np.savez_compressed(
            path,
            adj=adj,
            num_nodes_original=len(original_graph.nodes()),
            num_nodes_filtered=len(filtered_graph.nodes()),
            num_edges_original=len(original_graph.edges()),
            num_edges_filtered=len(filtered_graph.edges()),
            node_mapping=np.array(list(node_mapping.keys()), dtype=object)
        )
    elif ext == ".npy":
        np.save(path, adj)
    elif ext == ".pkl":
        with open(path, "wb") as f:
            pickle.dump(adj, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        raise ValueError(f"Formato não suportado: {ext}")
    
    nx.write_graphml(filtered_graph, f"../data/GraphML/{filtered_graph_out_name}.GraphML")
    
    print(f"  Nós originais: {len(original_graph.nodes())}")
    print(f"  Arestas originais: {len(original_graph.edges())}")
    print(f"  Nós filtrados: {len(filtered_graph.nodes())}")
    print(f"  Arestas filtradas: {len(filtered_graph.edges())}")
    print(f"  Redução: {(1 - len(filtered_graph.edges())/len(original_graph.edges()))*100:.1f}%")

def generate_graph_from_adjmx_nx(path: str, name, outputPath = "data/GraphML", ) -> nx.Graph:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pkl":
        with open(path, "rb") as f:
            data = pickle.load(f, encoding='latin1')
        if (type(data) is tuple) or (type(data) is list):
            _, _, adj = data
        else:            
            adj = data
    elif ext == ".npy":
        adj = np.load(path, allow_pickle=True)
    elif ext == ".npz":
        npz = np.load(path, allow_pickle=True)
        if "adj" in npz:
            adj = npz["adj"]
        else:
            adj = None
            for v in npz.values():
                if isinstance(v, np.ndarray) and v.ndim == 2:
                    adj = v
                    break
            if adj is None:
                raise ValueError("Não foi encontrado nenhuma matriz 2D no arquivo .npz")
    else:
        raise ValueError(f"Formato não suportado: {ext}")
    adj = np.array(adj)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("A matriz de adjacência deve ser quadrada.")
    
    graph= nx.from_numpy_array(adj)
    nx.write_graphml(graph, f"../{outputPath}/{name}.GraphML")
    return graph



def describe_network(g, name):
    print(f"{name}:")
    print(f"  Nós: {len(g.nodes())}")
    print(f"  Arestas: {len(g.edges())}")
    print(f"  Redução de arestas: {(1 - len(g.edges())/len(g.edges()))*100:.1f}%")
    print(f"O grafo é Direcionado: {g.is_directed()}")





def download_gdrive_zip(
    public_link,
    output_path=None,
    extract=True,
    overwrite=False
):


    # --- Extrair file_id ---
    if '/file/d/' in public_link:
        file_id = public_link.split('/file/d/')[1].split('/')[0]
    elif 'id=' in public_link:
        file_id = public_link.split('id=')[1].split('&')[0]
    else:
        raise ValueError("Link inválido. Use um link público do Google Drive.")

    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    if output_path is None:
        output_path = f"arquivo_{file_id}.zip"

    # --- Verificar existência do arquivo ---
    if os.path.exists(output_path):
        if not overwrite:
            print(f"✗ Arquivo já existe: {output_path}")
            print("Download cancelado (overwrite=False).")
            return output_path
        else:
            print(f"⚠ Arquivo já existe e será sobrescrito: {output_path}")

    print(f"Iniciando download de {file_id}...")

    session = requests.Session()
    response = session.get(download_url, stream=True)

    # --- Confirmação para arquivos grandes ---
    if 'confirm=' in response.text or 'download_warning' in response.text:
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                params = {'id': file_id, 'confirm': value}
                response = session.get(download_url, params=params, stream=True)
                break

    if response.status_code != 200:
        raise Exception(f"Erro ao baixar arquivo. Status: {response.status_code}")

    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    downloaded = 0

    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgresso: {percent:.1f}%", end='')

    print(f"\n✓ Download concluído: {output_path}")

    # --- Extrair ZIP ---
    if extract:
        import zipfile
        extract_path = os.path.splitext(output_path)[0]

        print(f"Extraindo para {extract_path}...")
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

        print(f"✓ Extração concluída em: {extract_path}")
        return extract_path

    return output_path





def download_gdrive_zip(
    public_link,
    output_path=None,
    extract=True,
    overwrite=False,
    h5_dir="dataset/h5",
    cleanup=True
):

    # --- Extrair file_id ---
    if '/file/d/' in public_link:
        file_id = public_link.split('/file/d/')[1].split('/')[0]
    elif 'id=' in public_link:
        file_id = public_link.split('id=')[1].split('&')[0]
    else:
        raise ValueError("Link inválido. Use um link público do Google Drive.")

    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    if output_path is None:
        output_path = f"arquivo_{file_id}.zip"

    output_path = os.path.abspath(output_path)

    # --- Verificar existência do ZIP ---
    if os.path.exists(output_path) and not overwrite:
        print(f"✗ Arquivo já existe: {output_path}")
        print("Download cancelado (overwrite=False).")
    else:
        print(f"Iniciando download de {file_id}...")
        session = requests.Session()
        response = session.get(download_url, stream=True)

        # Confirmação Google Drive (arquivos grandes)
        if 'download_warning' in response.text:
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    response = session.get(
                        download_url,
                        params={'id': file_id, 'confirm': value},
                        stream=True
                    )
                    break

        if response.status_code != 200:
            raise Exception(f"Erro ao baixar arquivo. Status: {response.status_code}")

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"✓ Download concluído: {output_path}")

    if not extract:
        return output_path

    # --- Extração ---
    extract_path = os.path.splitext(output_path)[0]
    extract_path = os.path.abspath(extract_path)

    print(f"Extraindo para {extract_path}...")
    with zipfile.ZipFile(output_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    # --- Criar pasta h5 ---
    h5_dir = os.path.abspath(h5_dir)
    os.makedirs(h5_dir, exist_ok=True)

    h5_files = []

    # --- Localizar e mover arquivos .h5 ---
    for root, _, files in os.walk(extract_path):
        for file in files:
            if file.lower().endswith(".h5"):
                src = os.path.join(root, file)
                dst = os.path.join(h5_dir, file)

                if os.path.exists(dst):
                    print(f"⚠ .h5 já existe e será sobrescrito: {dst}")

                shutil.move(src, dst)
                h5_files.append(os.path.abspath(dst))

    print(f"✓ {len(h5_files)} arquivos .h5 movidos para {h5_dir}")

    # --- Limpeza ---
    if cleanup:
        if os.path.exists(output_path):
            os.remove(output_path)
            print(f"✓ ZIP removido: {output_path}")

        if os.path.exists(extract_path):
            shutil.rmtree(extract_path)
            print(f"✓ Pasta temporária removida: {extract_path}")

    return h5_dir, h5_files




def h5ReconstructDataset(h5_file_path, dataset_name):
    with h5py.File(h5_file_path, 'r') as h5_file:
        if dataset_name not in h5_file:
            raise ValueError(f"Dataset '{dataset_name}' não encontrado no arquivo HDF5.")
        data = h5_file[dataset_name][:]
    return data


def load_graphml_backbone(filepath: Path) -> np.ndarray:
    """
    Carrega backbone de arquivo GraphML
    
    Args:
        filepath: Caminho para o arquivo .GraphML
        
    Returns:
        Matriz de adjacência do backbone
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Arquivo GraphML não encontrado: {filepath}")
    
    g = nx.read_graphml(filepath)
    backbone_adj = nx.adjacency_matrix(g).toarray()
    
    print(f"✓ Backbone carregado de {filepath.name}: shape {backbone_adj.shape}")
    return backbone_adj




def dataset_backbone_combinations(methods =["disp_fil", "nois_corr"], alpha = 0.1, percentile = 0.30 ):
    cuts = [f"alpah_filter{str(alpha).replace('.', '_')}", f"percen_filter{str(percentile).replace('.', '_')}"]
    combinations = list(itertools.product(methods, cuts))
    return combinations

