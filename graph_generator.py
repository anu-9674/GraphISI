"""
Generate simple connected graphs of orders 5–30, 20 per category.
Categories: {weighted, unweighted} × {directed, undirected}
Output: graphs.json (node-link format, readable by networkx)
"""

import json
import random
import networkx as nx
from networkx.readwrite import json_graph


def random_spanning_tree(n):
    perm = list(range(n))
    random.shuffle(perm)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(1, n):
        G.add_edge(perm[i], perm[random.randint(0, i - 1)])
    return G


def random_hamiltonian_cycle(n):
    perm = list(range(n))
    random.shuffle(perm)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        G.add_edge(perm[i], perm[(i + 1) % n])
    return G


def add_edges(G, k, weighted=False):
    nodes = list(G.nodes())
    added, attempts = 0, 0
    while added < k and attempts < k * 200:
        u, v = random.sample(nodes, 2)
        if not G.has_edge(u, v):
            kw = {"weight": round(random.uniform(1, 10), 2)} if weighted else {}
            G.add_edge(u, v, **kw)
            added += 1
        attempts += 1


def make_strongly_connected(G):
    for _ in range(300):
        if nx.is_strongly_connected(G):
            break
        sccs = [list(s) for s in nx.strongly_connected_components(G)]
        s1, s2 = random.sample(sccs, 2)
        u, v = random.choice(s1), random.choice(s2)
        for src, dst in [(u, v), (v, u)]:
            if not G.has_edge(src, dst):
                G.add_edge(src, dst)


def assign_weights(G):
    H = G.copy()
    for u, v in H.edges():
        H[u][v]["weight"] = round(random.uniform(1, 10), 2)
    return H


def generate(n, count=20, seed_base=0):
    result = {
        "weighted":   {"directed": [], "undirected": []},
        "unweighted": {"directed": [], "undirected": []},
    }

    for i in range(count):
        random.seed(seed_base + i)
        extra = random.randint(0, 3)

        T = random_spanning_tree(n)
        add_edges(T, extra)
        result["unweighted"]["undirected"].append(json_graph.node_link_data(T))
        result["weighted"]["undirected"].append(json_graph.node_link_data(assign_weights(T)))

        D = random_hamiltonian_cycle(n)
        add_edges(D, extra + max(1, n // 5))
        make_strongly_connected(D)
        result["unweighted"]["directed"].append(json_graph.node_link_data(D))
        result["weighted"]["directed"].append(json_graph.node_link_data(assign_weights(D)))

    return result

def generate_bipartite(n,count=20,seed_base=0):

    result = {
        "weighted":   {"directed": [], "undirected": []},
        "unweighted": {"directed": [], "undirected": []},
    }
    for i in range(count):
        random.seed(seed_base + i)

        left_split=random.randint(n//3, 2*n//3)
        B = nx.Graph()

        for node in range(left_split):
            B.add_node(node, bipartite=0)

        for node in range(left_split, n):
            B.add_node(node, bipartite=1)
        
        for u in range(left_split):
            for v in range(left_split, n):
                if random.random() < 0.5:
                    B.add_edge(u, v)

        if B.number_of_edges() == 0:
            u = random.randrange(left_split)
            v = random.randrange(left_split, n)
            B.add_edge(u, v)

        result['unweighted']['undirected'].append(json_graph.node_link_data(B))
        result['weighted']['undirected'].append(json_graph.node_link_data(assign_weights(B)))
        
        """directed bipartite graphs"""
        D = nx.DiGraph()
        left_split=random.randint(n//3, 2*n//3)
        for node in range(left_split):
            D.add_node(node, bipartite=0)

        for node in range(left_split, n):
            D.add_node(node, bipartite=1)

        for u in range(left_split):
            for v in range(left_split, n):
                if random.random() < 0.5:
                    D.add_edge(u, v)

        result['unweighted']['directed'].append(json_graph.node_link_data(D))
        result['weighted']['directed'].append(json_graph.node_link_data(assign_weights(D)))


    return result

    
def main():
    # data = {str(n): generate(n, seed_base=n * 1000) for n in range(5, 31)}
    # with open("./Data/graphs.json", "w") as f:
    #     json.dump(data, f, indent=2)
    # print(f"Saved graphs.json - {26 * 4 * 20} graphs total")

    data={str(n): generate_bipartite(n,seed_base=n*1000) for n in range(5,31)}
    with open("./Data/bipartite_graphs.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved graphs.json - {26 * 4 * 20} graphs total")


if __name__ == "__main__":
    main()
