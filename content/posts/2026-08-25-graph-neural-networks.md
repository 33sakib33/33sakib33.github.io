---
title: "Graph Neural Networks"
date: 2026-08-25
draft: false
categories: ["Technical"]
tags: ["gnn", "deep-learning", "graph-neural-networks"]
description: "Notes on how GNNs relate to CNNs, adjacency matrices, and batching graphs."
ShowToc: false
---

I was reading up on Graph Neural Networks(GNN) recently. The way researchers formulated the GNN architectures is really interesting.

The easiest way to understand GNNs is to compare it with CNN. Consequently, there is an architecture called GCNN or Graph Convolutional Neural Networks. GNNs and CNNs share some common motivation like, using shared parameters. Without shared parameters, CNN and GNNs both would need infeasible amount of parameters. CNNs solve this problem by exploiting the local spatial correlation of images. GNNs solve this problem using a node's adjacency matrix.

In practice, a very interesting trick from graph theory is used for graph convolutions. They represent the graph nodes and edges as an adjacency matrix and find the neighbourhood of a node using simple matrix multiplication.
Another cool trick in GNNs that I have found is in training Graphs as batches. In this case, a batch is made of a collection of disjoint graphs in a single matrix.

I'll probably share an illustrated and in depth post about this later.
