
import scanpy as sc
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import gseapy as gp
import os
from causallearn.search.ConstraintBased.PC import pc

def run_causal_pipeline(save_dir='/kaggle/working/'):
    print("🚀 Starting Automated Causal Drug Target Discovery Pipeline...")
    
    # 1. Load Data
    print("\n⏳ Step 1: Loading Dataset...")
    adata = sc.datasets.pbmc3k()
    
    # 2. Preprocessing
    print("\n🛠️ Step 2: Preprocessing and Quality Control...")
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=100)
    adata = adata[:, adata.var.highly_variable].copy()
    
    # 3. Causal Discovery
    print("\n🧠 Step 3: Running PC Causal Discovery Algorithm...")
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    gene_names = adata.var_names.tolist()
    cg = pc(X, 0.05, "fisherz")
    
    # 4. Network Construction
    print("\n🕸️ Step 4: Building Causal Network...")
    adj_matrix = cg.G.graph
    G = nx.DiGraph()
    for i in range(len(gene_names)):
        for j in range(len(gene_names)):
            if adj_matrix[i, j] == -1 and adj_matrix[j, i] == 1:
                G.add_edge(gene_names[i], gene_names[j])
                
    # 5. Identify Master Regulators
    print("\n🏆 Step 5: Identifying Master Regulators (Drug Targets)...")
    out_degrees = dict(G.out_degree())
    in_degrees = dict(G.in_degree())
    betweenness = nx.betweenness_centrality(G)
    
    df_targets = pd.DataFrame({
        'Gene': list(G.nodes()),
        'Out-Degree': [out_degrees[node] for node in G.nodes()],
        'In-Degree': [in_degrees[node] for node in G.nodes()],
        'Centrality': [round(betweenness[node], 4) for node in G.nodes()]
    }).sort_values(by=['Out-Degree', 'Centrality'], ascending=[False, False])
    
    df_targets.head(10).to_csv(os.path.join(save_dir, 'pipeline_top10_targets.csv'), index=False)
    
    top_target = df_targets.iloc[0]['Gene']
    print(f"Top Target Identified: {top_target}")
    
    # 6. Pathway Enrichment for Top Target Subnetwork
    print(f"\n🧬 Step 6: Pathway Analysis for {top_target} subnetwork...")
    affected_genes = list(G.successors(top_target))
    causing_genes = list(G.predecessors(top_target))
    subnetwork_genes = [top_target] + affected_genes + causing_genes
    
    try:
        enr = gp.enrichr(gene_list=subnetwork_genes,
                         gene_sets=['GO_Biological_Process_2021'],
                         organism='human',
                         outdir=None)
        enr.results.head(10).to_csv(os.path.join(save_dir, f'pipeline_{top_target}_pathways.csv'), index=False)
        print("✅ Pipeline Completed Successfully! All results saved.")
    except Exception as e:
        print(f"Pipeline completed with enrichment warning: {e}")

if __name__ == "__main__":
    run_causal_pipeline()
