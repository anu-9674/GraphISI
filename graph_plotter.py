"""
Read graphs.json and display a graph's adjacency matrix/list in the terminal,
then plot it with matplotlib.

Usage:
  python graph_plotter.py --order 8 --wt weighted --dir directed --idx 0
  python graph_plotter.py --order 8 --all          # 2x2 grid, all categories
"""

import argparse
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.readwrite import json_graph


def load(path="./Data/graphs.json"):
    with open(path) as f:
        return json.load(f)


def reconstruct(data, order, wt, direction, idx):
    directed = direction == "directed"
    return json_graph.node_link_graph(
        data[str(order)][wt][direction][idx],
        directed=directed, multigraph=False
    )


def print_adjacency(G, weighted):
    nodes = sorted(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    mat = np.zeros((n, n))

    for u, v, d in G.edges(data=True):
        w = d.get("weight", 1) if weighted else 1
        mat[idx[u]][idx[v]] = w
        if not G.is_directed():
            mat[idx[v]][idx[u]] = w

    header = f"{'':>4}" + "".join(f"{v:>6}" for v in nodes)
    print(header)
    print("-" * len(header))
    for i, u in enumerate(nodes):
        row = f"{u:>4}" + "".join(
            f"{mat[i][j]:>6.1f}" if mat[i][j] else f"{'·':>6}"
            for j in range(n)
        )
        print(row)
    print()

    # print("Adjacency list:")
    # for u in nodes:
    #     if weighted:
    #         nbrs = ", ".join(f"{v}({G[u][v]['weight']:.1f})" for v in sorted(G[u]))
    #     else:
    #         nbrs = ", ".join(str(v) for v in sorted(G[u]))
    #     print(f"  {u}: {nbrs}")
    # print()


def plot(G, weighted, directed, title, ax=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 5))

    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx(G, pos, ax=ax, with_labels=True,
                     node_color="#4C72B0", node_size=500,
                     font_color="white", font_size=9,
                     edge_color="#444", arrows=directed,
                     **( {"connectionstyle": "arc3,rad=0.07"} if directed else {} ))

    if weighted and G.number_of_nodes() <= 20:
        labels = {(u, v): f"{d['weight']:.1f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, labels, font_size=7, ax=ax)

    ax.set_title(title, fontsize=9)
    ax.axis("off")

    if standalone:
        plt.tight_layout()
        plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",  default="./Data/graphs.json")
    p.add_argument("--order", type=int, required=True)
    p.add_argument("--wt",    choices=["weighted", "unweighted"], default="weighted")
    p.add_argument("--dir",   choices=["directed", "undirected"], default="directed")
    p.add_argument("--idx",   type=int, default=0)
    p.add_argument("--all",   action="store_true", help="Plot all 4 categories")
    args = p.parse_args()

    data = load(args.data)

    if args.all:
        categories = [
            ("weighted",   "directed"),
            ("weighted",   "undirected"),
            ("unweighted", "directed"),
            ("unweighted", "undirected"),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(11, 9))
        fig.suptitle(f"Order {args.order} — all categories (graph #{args.idx})")
        for ax, (wt, direction) in zip(axes.flat, categories):
            G = reconstruct(data, args.order, wt, direction, args.idx)
            print(f"\n-- {wt} / {direction} --")
            print_adjacency(G, wt == "weighted")
            plot(G, wt == "weighted", direction == "directed",
                 f"{wt} · {direction}", ax=ax)
        plt.tight_layout()
        plt.show()
        plt.close()
    else:
        G = reconstruct(data, args.order, args.wt, args.dir, args.idx)
        print(f"\nOrder {args.order} | {args.wt} | {args.dir} | #{args.idx}")
        print_adjacency(G, args.wt == "weighted")
        plot(G, args.wt == "weighted", args.dir == "directed",
             f"n={args.order} · {args.wt} · {args.dir} · #{args.idx}")


if __name__ == "__main__":
    main()
