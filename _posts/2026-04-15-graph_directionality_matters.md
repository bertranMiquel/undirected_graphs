---
layout: distill
title: Directionality is the Hidden Bottleneck in Graph Learning
description: Your blog post's abstract.
  Please add your abstract or summary here and not in the main body of your text.
  Do not include math/latex or hyperlinks.
date: 2026-01-02
future: true
htmlwidgets: true
hidden: true

# Mermaid diagrams
mermaid:
  enabled: true
  zoomable: true

# Anonymize when submitting
authors:
  - name: Anonymous

# must be the exact same name as your blogpost
bibliography: 2026-04-15-graph_directionality_matters.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Equations
  - name: Images and Figures
    subsections:
      - name: Interactive Figures
  - name: Citations
  - name: Footnotes
  - name: Code Blocks
  - name: Diagrams
  - name: Tweets
  - name: Layouts
  - name: Other Typography?

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
    .table-caption {
    font-size: 0.85em;
    color: #64748b;
    text-align: left;
    margin-top: 1.5em;
    margin-bottom: 0.5em;
    line-height: 1.5;
    font-weight: 400;
  }
  .caption {
    font-size: 0.85em;
    color: #64748b;
    text-align: center;
    margin-top: -0.5em;
    margin-bottom: 1.5em;
    line-height: 1.5;
  }
---

# **Introduction**

Graph Neural Networks (GNNs) have achieved strong performance on node classification benchmarks, particularly in **homophilic settings**, where neighboring nodes tend to share labels. However, a widely accepted narrative is that GNNs struggle in **heterophilic graphs**, where connected nodes do not necessarily share labels. This observation has motivated a seek for improvement of GNN performance in heterophilic regimes through architectural modifications and alternative aggregation schemes <d-cite key="kipf2016semi"></d-cite>, <d-cite key="velivckovic2017graph"></d-cite>, <d-cite key="rossi2024edge"></d-cite>.

By definition, the key distinction between homophilic and heterophilic datasets lies in the degree of label similarity among neighboring nodes. However, an often overlooked aspect is that these two regimes also differ structurally: many heterophilic datasets are defined as**directed**, whereas homophilic datasets are typically constructed as **undirected**.

Historically, widely used homophilic benchmarks such as Cora, CiteSeer, and PubMed were constructed as **undirected graphs**. In early work (e.g., <d-cite key="kipf2016semi"></d-cite>), citation edges were explicitly symmetrized to produce a **symmetric adjacency matrix**, enabling the use of spectral graph convolutions. This design choice simplified the problem and ensured compatibility with models relying on symmetric normalization. As a consequence, modern libraries such as PyTorch Geometric represent these graphs as **bidirectional edge lists**, even when the underlying relationships (e.g., citations) are inherently directed.

Despite the development of models capable of handling asymmetric adjacency matrices, many benchmarks continue to rely on bidirectional versions of homophilic datasets. As shown in Table 1, nearly all homophilic datasets exhibit a **bidirectionality ratio of 1.0**, indicating that every edge has a corresponding reverse edge. The main exception is OGBN-Arxiv, which retains its original directionality and exhibits a very low bidirectionality ratio (1.4%).

This persistent design choice can be attributed to several factors:

* Some datasets are naturally undirected (e.g., co-authorship and co-purchase networks) <d-cite key="shchur2018pitfalls"></d-cite>
* Others are artificially symmetrized, despite having an underlying directed structure (e.g., citation graphs)
* Even recent works, such as <d-cite key="liang2025towards"></d-cite>, explicitly enforce bidirectionality to isolate specific phenomena (e.g., long-range dependencies).

In contrast, many heterophilic datasets (e.g., Cornell, Texas, Wisconsin, Chameleon, Squirrel) are typically used in their **original directed** or asymmetric form, as their underlying relationships (e.g., hyperlinks or interactions) are inherently directional <d-cite key="pei2020geom"></d-cite>. As reported in Table 1, these datasets exhibit significantly lower bidirectionality ratios, reflecting a high degree of asymmetry.

Importantly, several works have reported poor performance of GNNs on these heterophilic benchmarks and attributed it primarily to label inconsistency, analyzing structural properties of the graphs without explicitly controlling for directionality <d-cite key="zhu2020beyond"></d-cite>, <d-cite key="ma2021homophily"></d-cite>. While recent studies such as <d-cite key="rossi2024edge"></d-cite> show that **directed GNNs (DirGNNs)** can improve performance on heterophilic datasets, a performance gap with respect to homophilic benchmarks still remains. However, these analyses do not disentangle the effects of heterophily from those of directionality, leaving open the question of which factor is the primary driver of GNN performance.

Taken together, these observations raise a fundamental question:

> Are GNNs failing because of **heterophily**, or because of **directionality**?


<div class="table-caption">
Table 1. Bidirectional properties of graph datasets. We report the dataset type (homophilic vs heterophilic), whether the graph is effectively bidirectional, and the bidirectionality ratio (fraction of edges with a reverse counterpart).
</div>

| dataset                 | type         | bidirectionality | bidirectionality_ratio |
| ----------------------- | ------------ | ---------------- | ---------------------- |
| amazon-computers        | homophilic   | True             | 1.0000                 |
| amazon-photo            | homophilic   | True             | 1.0000                 |
| citeseer_full           | homophilic   | True             | 1.0000                 |
| coauthor-cs             | homophilic   | True             | 1.0000                 |
| coauthor-phy            | homophilic   | True             | 1.0000                 |
| cora_ml                 | homophilic   | True             | 1.0000                 |
| ogbn-arxiv              | homophilic   | False            | 0.0145                 |
| pubmed                  | homophilic   | True             | 1.0000                 |
| chameleon               | heterophilic | False            | 0.2596                 |
| cornell                 | heterophilic | False            | 0.1220                 |
| directed-roman-empire   | heterophilic | False            | 0.3180                 |
| directed_amazon_ratings | heterophilic | False            | 0.3571                 |
| squirrel                | heterophilic | False            | 0.1713                 |
| texas                   | heterophilic | False            | 0.1942                 |
| wisconsin               | heterophilic | False            | 0.1964                 |


To answer this question, we perform a systematic study across widely used homophilic and heterophilic benchmarks. Specifically, we consider homophilic datasets such as Cora, CiteSeer, PubMed, Amazon, OGBN-Arxiv, and Coauthor, and heterophilic datasets including Cornell, Texas, Wisconsin, Chameleon, and Squirrel.

We explicitly control for directionality by applying two complementary transformations:

1. **Bidirectionalization.**
   Directed or asymmetric graphs are converted into fully bidirectional graphs by adding reverse edges:
   [
   (u, v) \rightarrow (u, v), (v, u)
   ]

2. **Directed graph construction.**
   Undirected graphs are transformed into directed graphs using principled orientation strategies based on:

   * temporal metadata (e.g., citation graphs: newer → older)
   * structural centrality (e.g., PageRank, degree)
   * domain-specific proxies (e.g., popularity, seniority)

   Each undirected edge is replaced by a **single directed edge**, ensuring controlled asymmetry while preserving the underlying graph structure.

After applying these transformations, we evaluate standard GNN architectures (GCN, GAT, GraphSAGE) and their directed counterparts across multiple random seeds to ensure robustness.

---

# **Results**

We evaluate the performance of GNNs of the datasets in their original form, as well as after applying the bidirectionalization and directed graph construction transformations. This allows us to isolate the effects of directionality from those of homophily.
As in <d-cite key="rossi2024edge"></d-cite>, we use the same hyperparameters for all models to ensure a fair comparison, taking the datasets and their standard splits.

{% include figure.liquid
  src="assets/img/2026-04-15-graph_directionality_matters/heterophilic_baseline_gnn_vs_directed.png"
  caption="Performance of GNNs on heterophilic datasets in their original directed form. Standard GNNs (GCN, GAT, SAGE) perform poorly, while directed GNNs (DirGCN, DirGAT, DirSAGE) show significant improvements."
%}

{% include figure.liquid
  src="assets/img/2026-04-15-graph_directionality_matters/homophilic_baseline_gnn_vs_directed.png"
  caption="Performance of GNNs on homophilic datasets in their original directed form. Standard GNNs (GCN, GAT, SAGE) perform poorly, while directed GNNs (DirGCN, DirGAT, DirSAGE) show significant improvements."
%}

<!-- [!**Figure 1.** Performance of GNNs on heterophilic datasets in their original directed form. Standard GNNs (GCN, GAT, SAGE) perform poorly, while directed GNNs (DirGCN, DirGAT, DirSAGE) show significant improvements.](assets/img/2026-04-15-graph_directionality_matters/heterophilic_baseline_gnn_vs_directed.png)

[!**Figure 2.** Performance of GNNs on homophilic datasets in their original directed form. Standard GNNs (GCN, GAT, SAGE) perform poorly, while directed GNNs (DirGCN, DirGAT, DirSAGE) show significant improvements.](assets/img/2026-04-15-graph_directionality_matters/homophilic_baseline_gnn_vs_directed.png) -->

<!-- <p align="center">
  <img src="assets/img/2026-04-15-graph_directionality_matters/heterophilic_baseline_gnn_vs_directed.png" width="500"/>
</p>

<p align="center">
  <b>Figure 1.</b> Performance of GNNs on heterophilic datasets in their original directed form. Standard GNNs (GCN, GAT, SAGE) perform poorly, while directed GNNs (DirGCN, DirGAT, DirSAGE) show significant improvements.
</p>

<p align="center">
  <img src="assets/img/2026-04-15-graph_directionality_matters/homophilic_baseline_gnn_vs_directed.png" width="500"/>
</p>

<p align="center">
  <b>Figure 2.</b> Performance of GNNs on homophilic datasets in their original directed form. Standard GNNs (GCN, GAT, SAGE) perform poorly, while directed GNNs (DirGCN, DirGAT, DirSAGE) show significant improvements.
</p> -->

In Figure 1, we observe how the performance of directed GNNs (DirGCN, DirGAT, DirSAGE) significantly outperforms standard GNNs (GCN, GAT, GraphSAGE) on heterophilic datasets in their original directed form. While in Figure 2, we see how directed GNNs have almost no difference in performance compared to standard GNNs on homophilic datasets, when the graphs is bidirectional. However, results on OGBN-Arxiv show a significant improvement when using directed GNNs.

This suggests that the poor performance of standard GNNs on these benchmarks is largely driven by their inability to handle directed graphs, rather than solely by label inconsistency.

{% include figure.liquid
  src="assets/img/2026-04-15-graph_directionality_matters/heterophilic_bidirected_gnn_vs_directed.png"
  caption="Performance of GNNs on bidirected heterophilic datasets in their original directed form. Standard GNNs (GCN, GAT) perform poorly, while directed GNNs (DirGCN, DirGAT) show significant improvements."
%}

<!-- [!**Figure 3.** Performance of GNNs on bidirected heterophilic datasets in their original directed form. Standard GNNs (GCN, GAT) perform poorly, while directed GNNs (DirGCN, DirGAT) show significant improvements.](assets/img/2026-04-15-graph_directionality_matters/heterophilic_bidirected_gnn_vs_directed.png) -->

<!-- <p align="center">
  <img src="assets/img/2026-04-15-graph_directionality_matters/heterophilic_bidirected_gnn_vs_directed.png" width="500"/>
</p>

<p align="center">
  <b>Figure 3.</b> Performance of GNNs on bidirected heterophilic datasets in their original directed form. Standard GNNs (GCN, GAT) perform poorly, while directed GNNs (DirGCN, DirGAT) show significant improvements.
</p> -->
