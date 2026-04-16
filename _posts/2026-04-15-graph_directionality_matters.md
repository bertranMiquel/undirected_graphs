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
---

# **Introduction**

<!-- difference between homophilic and heterophilic datasets -->
Graph Neural Networks (GNNs) have achieved strong performance on canonical node classification benchmarks.  However, a widely accepted narrative is that GNNs struggle in **heterophilic graphs**, where neighboring nodes do not share labels, while obtaining strong results in **homophilic graphs**, where neighbors tend to share labels. 
This has motivated a growing body of work studying how to improve GNNs in heterophilic settings, with a focus on architectural modifications and new aggregation schemes. <d-cite key="kipf2016semi"></d-cite>, <d-cite key="velivckovic2017graph"></d-cite>, <d-cite key="rossi2024edge"></d-cite>

The mainly difference between homophilic and heterophilic datasets, by definition, is the degree of label similarity among neighboring nodes. However, an often overlooked aspect is that many heterophilic datasets are also **directed**, while homophilic datasets are typically **undirected**.

Historically, many of the most widely used homophilic datasets such as Cora, CiteSeer, and PubMed, were originally constructed as **undirected graphs**. In early work (e.g., <d-cite key="kipf2016semi"></d-cite>), citation edges were explicitly **symmetrized** to produce a **symmetric adjacency matrix**, enabling the use of spectral graph convolutions. This was done to simplify the problem and ensure compatibility with the proposed GCN architecture, which relies on symmetric normalization of the adjacency matrix.
As a consequence, modern libraries (e.g., PyTorch Geometric) store these graphs as **bidirectional edge lists**, even when the underlying relation (citation) is inherently directed.

Despite new models being designed to handle non-symmetric adjacency matrices, numerous benchmarks continue to use **bidirectional versions of homophilic datasets**. This design choice has persisted for several reasons:

* Some datasets are naturally undirected (e.g., co-authorship, co-purchase) <d-cite key="shchur2018pitfalls"></d-cite>
* Others are **artificially symmetrized** despite having an underlying directed structure (e.g., citation graphs)
* Even recent works, such as <d-cite key="liang2025towards"></d-cite>, deliberately enforce **bidirectionality** to isolate other factors (e.g., long-range dependencies) while controlling for directionality.

Meanwhile, many heterophilic datasets (e.g., Cornell, Texas, Wisconsin, Chameleon, Squirrel) are often used in their **original directed form**, without symmetrization. This is partly because their relations (e.g., web links, social interactions) are inherently directed. <d-cite key="pei2020geom-gcn"></d-cite>
Several datasets reported bad performance in different architectures, relating it to the heterophilic nature of the graph, analysing different graph features, but without controlling for the directionality of the graph. <d-cite key="zhu2020beyond"></d-cite>, <d-cite key="ma2021homophily"></d-cite>

In <d-cite key="rossi2024edge"></d-cite>, they report that the performance of GNNs on heterophilic datasets is significantly worse than on homophilic datasets. However, they observe how directed GNNs (DirGNNs) can significantly improve performance on heterophilic datasets, but still underperform compared to their performance on homophilic datasets. This suggests that while directionality is a factor, it may not be the only one contributing to the performance gap. 

This raises a fundamental question:

> Are GNNs failing because of **heterophily**, or because of **directionality**?

To investigate this, we perform a systematic study across both most popular homophilic and heterophilic datasets. For homophilic datasets, we consider Cora, CiteSeer, PubMed, Amazon, OGBN-Arxiv, and Coauthor. For heterophilic datasets, we consider Cornell, Texas, Wisconsin, Chameleon, Squirrel, and others.

In this study, we want to analyse how directionality affects GNN performance no matter the initial nature of the datasets and explicitly control for directionality by applying two complementary transformations to the graphs:
1. **Bidirectionalizing graphs.** We convert directed or asymmetric graphs into fully bidirectional graphs by explicitly adding reverse edges:
[
(u, v) \rightarrow (u, v), (v, u)
]

1. **Introducing directionality** into undirected graphs. We construct directed versions of homophilic graphs using principled orientation strategies based on:

* temporal metadata (citation graphs: newer → older)
* structural centrality (PageRank, degree)
* domain-specific proxies (popularity, seniority)

Each undirected edge is replaced by a **single directed edge**, ensuring controlled asymmetry (implementation details in ).

Once the transformations are applied, we evaluate the performance of standard GNNs (GCN, GAT, GraphSAGE) and their directed variants across multiple random seeds to ensure robustness.

---

## Empirical evidence: directionality dominates

We evaluate standard GNNs (GCN, GAT, GraphSAGE) and their directed variants across 5 seeds.

### Heterophilic graphs: bidirectionality boosts performance

Across datasets, we observe **consistent improvements** when enforcing bidirectionality:

* **Chameleon (GCN)**:
  +41.6 accuracy points
* **Squirrel (GCN)**:
  +31.6 points
* **Directed Amazon Ratings (GCN)**:
  +8.7 points
* **Texas / Cornell**:
  modest but consistent gains

Even for GAT, improvements are substantial on several datasets (e.g., +27.6 on Squirrel).

Moreover:

> The performance gap between GNNs and DirGNNs **largely disappears** once graphs are bidirectional.

---

### Homophilic graphs: directionality hurts GNNs

When introducing directionality into traditionally undirected graphs:

* Standard GNNs experience a **clear drop in performance**
* Directed GNNs **recover or improve accuracy**

This indicates that:

> The success of GNNs on homophilic benchmarks is partially dependent on **bidirectional connectivity**, not solely on label similarity.

---

## Rethinking the problem

The dominant framing of graph learning separates problems into **homophilic vs heterophilic regimes**.

Our results suggest a different perspective:

> The primary challenge for GNNs is not whether neighbors share labels, but whether information can **flow symmetrically** across the graph.

Directionality introduces:

* asymmetric reachability
* reduced path multiplicity
* stronger information bottlenecks
* constrained message propagation

In contrast, bidirectional graphs artificially:

* increase connectivity
* enhance information flow
* mitigate structural bottlenecks

---

## A new perspective: beyond homophily

We argue that:

> Many failures attributed to heterophily are in fact driven by **directionality-induced constraints on message passing**.

This does not invalidate heterophily as a factor, but highlights a critical interaction:

* heterophily + directionality → worst-case scenario
* homophily + bidirectionality → best-case scenario

Crucially, current benchmarks do not disentangle these effects.

---

## Research directions

Our findings suggest three immediate directions:

### 1. Dataset design

* Construct **directed homophilic datasets**
* Construct **undirected heterophilic variants**
* Decouple label similarity from graph direction

### 2. Model development

* Improve robustness to **asymmetric adjacency**
* Design architectures that explicitly model **directional flow**

### 3. Evaluation protocols

* Treat **directionality as a controlled variable**
* Report results under both directed and undirected settings

---

## What this work does *not* claim

* We do not claim heterophily is irrelevant
* We do not claim directionality explains all failures
* We do not propose new architectures

Instead, we provide empirical evidence that:

> Directionality is a **critical, underexplored factor** that significantly impacts GNN performance and interacts with homophily.

---

## Summary

The widespread use of bidirectional graphs in benchmarks introduces a structural bias that favors existing GNN architectures.

By explicitly controlling for directionality, we show that many limitations attributed to heterophily may instead arise from the inability of GNNs to handle **directed graphs**.


---

# What is still missing (for paper)

Next steps for the position paper:

1. **Figure 1 (critical)**

   * Bar plot: baseline vs bidirected (heterophilic)
2. **Figure 2**

   * Directed vs undirected (homophilic)
3. **Table 1**

   * Full results (mean ± std)
4. **Short related work section**
5. **1-paragraph method section (your transformations)**
