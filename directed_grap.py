#!/usr/bin/env python3
"""
Many-queries benchmark on ONE static graph.

Goal:
Compare query-time performance of
  - Dijkstra
  - A*
  - ALT query (with preprocessing done once)

This matches the standard use case for ALT: many shortest-path queries on the same fixed graph.

Graph model:
- Random directed acyclic graph (DAG) on nodes 0..n-1
- Edges only i->j with i<j
- Backbone edges i->i+1 ensure every pair (s,t) with s<t has a path
- Positive integer weights in [w_min, w_max]

Output:
Prints exactly the data that is typically copied into the paper:
  - n, m, edge_prob, weight range
  - number of queries Q
  - ALT parameters (k landmarks, selection method)
  - preprocessing time
  - mean ± sd query times for Dijkstra, A*, ALT(query)
  - ALT amortized time per query = pre_time/Q + mean_query_time
"""

import argparse
import heapq
import math
import random
import statistics
import time
from typing import List, Tuple

INF = float("inf")


# -----------------------------
# Graph generation (random DAG)
# -----------------------------

def generate_random_connected_dag(
    n: int,
    edge_prob: float,
    w_min: int,
    w_max: int,
    rng: random.Random,
) -> List[List[Tuple[int, int]]]:
    if n < 2:
        raise ValueError("n must be at least 2")

    adj: List[List[Tuple[int, int]]] = [[] for _ in range(n)]
    has_edge = set()

    def add_edge(u: int, v: int, w: int) -> None:
        if (u, v) in has_edge:
            return
        has_edge.add((u, v))
        adj[u].append((v, w))

    # Backbone
    for i in range(n - 1):
        add_edge(i, i + 1, rng.randint(w_min, w_max))

    # Extra edges
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in has_edge:
                continue
            if rng.random() < edge_prob:
                add_edge(i, j, rng.randint(w_min, w_max))

    for u in range(n):
        adj[u].sort(key=lambda x: x[0])
    return adj


def reverse_adjacency(adj: List[List[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
    n = len(adj)
    radj: List[List[Tuple[int, int]]] = [[] for _ in range(n)]
    for u in range(n):
        for v, w in adj[u]:
            radj[v].append((u, w))
    for v in range(n):
        radj[v].sort(key=lambda x: x[0])
    return radj


def edge_count(adj: List[List[Tuple[int, int]]]) -> int:
    return sum(len(lst) for lst in adj)


# -----------------------------
# Dijkstra distance-only
# -----------------------------

def dijkstra_distance(adj: List[List[Tuple[int, int]]], s: int, t: int) -> float:
    n = len(adj)
    dist = [INF] * n
    dist[s] = 0.0
    settled = [False] * n
    pq: List[Tuple[float, int]] = [(0.0, s)]

    while pq:
        du, u = heapq.heappop(pq)
        if settled[u]:
            continue
        settled[u] = True
        if u == t:
            return du
        for v, w in adj[u]:
            if settled[v]:
                continue
            alt = du + w
            if alt < dist[v]:
                dist[v] = alt
                heapq.heappush(pq, (alt, v))

    return INF


# -----------------------------
# A* distance-only with a simple admissible heuristic
# h_t(v) = 0 if v==t else min_out[v]
# -----------------------------

def compute_min_outgoing(adj: List[List[Tuple[int, int]]]) -> List[float]:
    n = len(adj)
    out_min = [0.0] * n
    for v in range(n):
        if adj[v]:
            out_min[v] = float(min(w for _, w in adj[v]))
        else:
            out_min[v] = 0.0
    return out_min


def astar_distance_minout(
    adj: List[List[Tuple[int, int]]],
    s: int,
    t: int,
    out_min: List[float],
) -> float:
    # admissible heuristic value
    def h(v: int) -> float:
        return 0.0 if v == t else out_min[v]

    n = len(adj)
    g = [INF] * n
    g[s] = 0.0

    pq: List[Tuple[float, float, int]] = [(h(s), 0.0, s)]  # (f, g, node)
    best_goal = INF

    while pq:
        f_u, g_u, u = heapq.heappop(pq)
        if g_u != g[u]:
            continue

        if u == t:
            best_goal = g_u

        # correct stopping rule for admissible (not necessarily consistent) heuristics
        if best_goal < INF:
            if not pq or pq[0][0] >= best_goal:
                return best_goal

        for v, w in adj[u]:
            alt = g_u + w
            if alt < g[v]:
                g[v] = alt
                heapq.heappush(pq, (alt + h(v), alt, v))

    return best_goal


# -----------------------------
# ALT preprocessing and query (distance-only)
# Directed ALT lower bounds:
#   d(v,t) >= d(L,t) - d(L,v)
#   d(v,t) >= d(v,L) - d(t,L)
# -----------------------------

def dijkstra_all(adj: List[List[Tuple[int, int]]], source: int) -> List[float]:
    n = len(adj)
    dist = [INF] * n
    dist[source] = 0.0
    pq: List[Tuple[float, int]] = [(0.0, source)]
    settled = [False] * n
    while pq:
        du, u = heapq.heappop(pq)
        if settled[u]:
            continue
        settled[u] = True
        for v, w in adj[u]:
            alt = du + w
            if alt < dist[v]:
                dist[v] = alt
                heapq.heappush(pq, (alt, v))
    return dist


def choose_landmarks_even(n: int, k: int, rng: random.Random, exclude: Tuple[int, int] = (None, None)) -> List[int]:
    s, t = exclude
    candidates = [v for v in range(n) if v not in (s, t)]
    if k <= 0 or not candidates:
        return []
    k = min(k, len(candidates))
    idxs = []
    for i in range(1, k + 1):
        pos = round(i * (len(candidates) - 1) / (k + 1))
        idxs.append(candidates[pos])
    # de-dup and fill
    out, seen = [], set()
    for v in idxs:
        if v not in seen:
            seen.add(v)
            out.append(v)
    for v in candidates:
        if len(out) >= k:
            break
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def alt_preprocess(adj: List[List[Tuple[int, int]]], landmarks: List[int]) -> Tuple[List[List[float]], List[List[float]]]:
    radj = reverse_adjacency(adj)
    dist_from = []
    dist_to = []
    for L in landmarks:
        dist_from.append(dijkstra_all(adj, L))
        dist_to.append(dijkstra_all(radj, L))  # equals distances to L in the original graph
    return dist_from, dist_to


def alt_h_value(
    v: int,
    t: int,
    dist_from: List[List[float]],
    dist_to: List[List[float]],
) -> float:
    # landmarks indexed by i; their identity isn't needed here, only the precomputed arrays
    best = 0.0
    k = len(dist_from)
    for i in range(k):
        dLv = dist_from[i][v]
        dLt = dist_from[i][t]
        if dLv < INF and dLt < INF:
            best = max(best, dLt - dLv)

        dvL = dist_to[i][v]
        dtL = dist_to[i][t]
        if dvL < INF and dtL < INF:
            best = max(best, dvL - dtL)

    return best if best > 0.0 else 0.0


def alt_query_distance(
    adj: List[List[Tuple[int, int]]],
    s: int,
    t: int,
    dist_from: List[List[float]],
    dist_to: List[List[float]],
) -> float:
    n = len(adj)
    g = [INF] * n
    g[s] = 0.0

    def h(v: int) -> float:
        return 0.0 if v == t else alt_h_value(v, t, dist_from, dist_to)

    pq: List[Tuple[float, float, int]] = [(h(s), 0.0, s)]
    best_goal = INF

    while pq:
        f_u, g_u, u = heapq.heappop(pq)
        if g_u != g[u]:
            continue

        if u == t:
            best_goal = g_u

        if best_goal < INF:
            if not pq or pq[0][0] >= best_goal:
                return best_goal

        for v, w in adj[u]:
            alt = g_u + w
            if alt < g[v]:
                g[v] = alt
                heapq.heappush(pq, (alt + h(v), alt, v))

    return best_goal


# -----------------------------
# Helpers: timing + summary
# -----------------------------

def mean_sd(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return math.nan, math.nan
    mu = statistics.mean(xs)
    sd = statistics.stdev(xs) if len(xs) >= 2 else 0.0
    return mu, sd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="Number of nodes.")
    ap.add_argument("--edge-prob", type=float, default=0.03, help="Probability for extra edges i->j (i<j).")
    ap.add_argument("--w-min", type=int, default=1, help="Minimum weight.")
    ap.add_argument("--w-max", type=int, default=20, help="Maximum weight.")
    ap.add_argument("--queries", type=int, default=2000, help="Number of shortest-path queries on the same graph.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed.")
    ap.add_argument("--alt-k", type=int, default=8, help="Number of landmarks for ALT.")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    adj = generate_random_connected_dag(
        n=args.n,
        edge_prob=args.edge_prob,
        w_min=args.w_min,
        w_max=args.w_max,
        rng=rng,
    )
    n = args.n
    m = edge_count(adj)

    # Precompute A* heuristic component
    out_min = compute_min_outgoing(adj)

    # Choose landmarks and preprocess ALT once
    s_excl, t_excl = 0, n - 1
    landmarks = choose_landmarks_even(n, args.alt_k, rng, exclude=(s_excl, t_excl))

    t0 = time.perf_counter()
    dist_from, dist_to = alt_preprocess(adj, landmarks)
    t1 = time.perf_counter()
    alt_pre_time = t1 - t0

    # Sample queries (s<t ensures reachability in this DAG model)
    queries: List[Tuple[int, int]] = []
    for _ in range(args.queries):
        s = rng.randrange(0, n - 1)
        t = rng.randrange(s + 1, n)
        queries.append((s, t))

    # Time query-only execution
    td, ta, tl = [], [], []
    tol = 1e-9

    for (s, t) in queries:
        # Dijkstra
        start = time.perf_counter()
        dd = dijkstra_distance(adj, s, t)
        td.append(time.perf_counter() - start)

        # A*
        start = time.perf_counter()
        da = astar_distance_minout(adj, s, t, out_min)
        ta.append(time.perf_counter() - start)

        # ALT (query only)
        start = time.perf_counter()
        dl = alt_query_distance(adj, s, t, dist_from, dist_to)
        tl.append(time.perf_counter() - start)

        # correctness check
        if abs(dd - da) > tol or abs(dd - dl) > tol:
            raise RuntimeError(f"Distance mismatch for query (s={s}, t={t}): d={dd}, a*={da}, alt={dl}")

    mu_d, sd_d = mean_sd(td)
    mu_a, sd_a = mean_sd(ta)
    mu_l, sd_l = mean_sd(tl)

    alt_amortized = alt_pre_time / args.queries + mu_l

    # Print paper-ready summary
    print("Static-graph many-queries benchmark")
    print("----------------------------------")
    print(f"n={n}, m={m}, edge_prob={args.edge_prob}, weights=[{args.w_min},{args.w_max}]")
    print(f"queries={args.queries}")
    print(f"ALT: k={args.alt_k}, landmarks={landmarks}")
    print()
    print(f"ALT preprocessing time: {alt_pre_time:.6g} s")
    print("Query times (mean ± sd over queries):")
    print(f"  Dijkstra: {mu_d:.6g} ± {sd_d:.6g} s")
    print(f"  A*:       {mu_a:.6g} ± {sd_a:.6g} s")
    print(f"  ALT:      {mu_l:.6g} ± {sd_l:.6g} s")
    print()
    print(f"ALT amortized per-query time (pre/Q + mean_query): {alt_amortized:.6g} s")


if __name__ == "__main__":
    main()
