#!/usr/bin/env python3
"""
Benchmark shortest-path algorithms on random directed, positively-weighted, connected acyclic graphs (DAGs).

Algorithms timed (single-source single-target):
- Exhaustive enumeration of all s->t paths (backtracking)   [exponential; intended as a baseline]
- Dijkstra (nonnegative weights)
- A* with an admissible heuristic (min outgoing edge weight)
- ALT (Landmarks + A*) with landmark-based admissible heuristic

Graph model:
- Nodes are {0,1,...,n-1}, with s=0 and t=n-1.
- Acyclicity is guaranteed by only allowing edges i -> j with i < j.
- "Connected" is ensured by including the backbone path 0->1->2->...->n-1, so every node is reachable from s
  and can reach t via the backbone.
- Edge weights are positive integers in [w_min, w_max].

Usage examples:
  python benchmark_sp.py --n 12 --trials 5 --edge-prob 0.15
  python benchmark_sp.py --n 18 --trials 3 --edge-prob 0.10 --alt-k 4
  python benchmark_sp.py --n 25 --trials 2 --edge-prob 0.06 --skip-exhaustive

Notes:
- Exhaustive enumeration becomes infeasible quickly as n or edge density increases.
- ALT has preprocessing; the script reports both preprocessing+query and query-only timings.
"""

import argparse
import heapq
import math
import random
import statistics
import time
from typing import Callable, List, Optional, Tuple


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
    """
    Returns adjacency list for a random connected DAG on nodes 0..n-1.
    Acyclic: only edges i->j with i<j.
    Connected (reachability s->all and all->t): includes backbone edges i->i+1.
    """
    if n < 2:
        raise ValueError("n must be at least 2")

    adj: List[List[Tuple[int, int]]] = [[] for _ in range(n)]
    has_edge = set()

    def add_edge(u: int, v: int, w: int) -> None:
        if (u, v) in has_edge:
            return
        has_edge.add((u, v))
        adj[u].append((v, w))

    # Backbone: guarantees reachability and a valid s->t path
    for i in range(n - 1):
        w = rng.randint(w_min, w_max)
        add_edge(i, i + 1, w)

    # Extra random edges (still respecting acyclicity i<j)
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in has_edge:
                continue
            if rng.random() < edge_prob:
                w = rng.randint(w_min, w_max)
                add_edge(i, j, w)

    # Optional: sort adjacency for reproducibility (not required)
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


# -----------------------------
# Utilities (path reconstruction)
# -----------------------------

def reconstruct_path(parent: List[int], s: int, t: int) -> Optional[List[int]]:
    if s == t:
        return [s]
    if parent[t] == -1:
        return None
    path = []
    cur = t
    while cur != -1:
        path.append(cur)
        if cur == s:
            break
        cur = parent[cur]
    if path[-1] != s:
        return None
    path.reverse()
    return path


# -----------------------------
# Algorithm 1: Exhaustive enumeration (backtracking)
# -----------------------------

def exhaustive_shortest_path_dag(
    adj: List[List[Tuple[int, int]]],
    s: int,
    t: int,
) -> Tuple[float, Optional[List[int]]]:
    """
    Enumerates all directed s->t paths and returns the minimum-cost path.
    Works for any directed graph if you include cycle-avoidance; here the graph is a DAG by construction.
    """
    n = len(adj)
    best_cost = INF
    best_parent = [-1] * n
    parent = [-1] * n

    # In a DAG, cycle-avoidance is unnecessary, but keeping a recursion stack is harmless and defensive.
    in_stack = [False] * n

    def dfs(u: int, cost_so_far: float) -> None:
        nonlocal best_cost, best_parent

        if cost_so_far >= best_cost:
            return

        if u == t:
            best_cost = cost_so_far
            best_parent = parent[:]  # snapshot to reconstruct path
            return

        in_stack[u] = True
        for v, w in adj[u]:
            if in_stack[v]:
                continue
            parent[v] = u
            dfs(v, cost_so_far + w)
            parent[v] = -1
        in_stack[u] = False

    parent[s] = -1
    dfs(s, 0.0)
    if best_cost == INF:
        return INF, None
    return best_cost, reconstruct_path(best_parent, s, t)


# -----------------------------
# Algorithm 2: Dijkstra (single-source single-target)
# -----------------------------

def dijkstra_shortest_path(
    adj: List[List[Tuple[int, int]]],
    s: int,
    t: int,
) -> Tuple[float, Optional[List[int]]]:
    n = len(adj)
    dist = [INF] * n
    parent = [-1] * n
    dist[s] = 0.0

    pq: List[Tuple[float, int]] = [(0.0, s)]
    settled = [False] * n

    while pq:
        du, u = heapq.heappop(pq)
        if settled[u]:
            continue
        settled[u] = True

        if u == t:
            break

        for v, w in adj[u]:
            if settled[v]:
                continue
            alt = du + w
            if alt < dist[v]:
                dist[v] = alt
                parent[v] = u
                heapq.heappush(pq, (alt, v))

    if dist[t] == INF:
        return INF, None
    return dist[t], reconstruct_path(parent, s, t)


# -----------------------------
# Algorithm 3: A* (admissible heuristic)
# -----------------------------

def make_min_outgoing_heuristic(adj: List[List[Tuple[int, int]]], t: int) -> List[float]:
    """
    Admissible heuristic for positive-weight directed graphs:
    h(v) = min outgoing edge cost from v, and h(t)=0.
    Reason: any path from v to t must take at least one outgoing edge (unless v=t).
    """
    n = len(adj)
    h = [0.0] * n
    for v in range(n):
        if v == t:
            h[v] = 0.0
        else:
            # By our generator, v<n-1 always has at least the backbone edge to v+1
            h[v] = min(w for _, w in adj[v]) if adj[v] else 0.0
    return h


def astar_shortest_path_admissible(
    adj: List[List[Tuple[int, int]]],
    s: int,
    t: int,
    h: List[float],
) -> Tuple[float, Optional[List[int]]]:
    """
    A* that is correct for admissible (not necessarily consistent) heuristics:
    it terminates when the smallest f-value in the open set is >= best known goal cost.
    """
    n = len(adj)
    g = [INF] * n
    parent = [-1] * n
    g[s] = 0.0

    pq: List[Tuple[float, float, int]] = [(h[s], 0.0, s)]  # (f, g, node)
    best_goal = INF

    while pq:
        f_u, g_u, u = heapq.heappop(pq)
        if g_u != g[u]:
            continue

        if u == t:
            best_goal = g_u  # best path to goal found so far

        # If a goal path is known and no node can lead to a cheaper one, stop.
        if best_goal < INF:
            if not pq or pq[0][0] >= best_goal:
                break

        for v, w in adj[u]:
            alt = g_u + w
            if alt < g[v]:
                g[v] = alt
                parent[v] = u
                heapq.heappush(pq, (alt + h[v], alt, v))

    if best_goal == INF:
        return INF, None
    return best_goal, reconstruct_path(parent, s, t)


# -----------------------------
# Algorithm 4: ALT (Landmarks + A*)
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


def choose_landmarks_even(n: int, k: int, s: int, t: int) -> List[int]:
    """
    Deterministic, simple landmark selection: spread landmarks along the topological order.
    This is not optimal in general, but it is a transparent, reproducible choice for experiments.
    """
    candidates = [v for v in range(n) if v not in (s, t)]
    if not candidates or k <= 0:
        return []
    k = min(k, len(candidates))
    # pick approximately evenly spaced indices
    idxs = []
    for i in range(1, k + 1):
        pos = round(i * (len(candidates) - 1) / (k + 1))
        idxs.append(candidates[pos])
    # remove duplicates while preserving order
    seen = set()
    out = []
    for v in idxs:
        if v not in seen:
            seen.add(v)
            out.append(v)
    # If duplicates reduced count, fill from left to right
    for v in candidates:
        if len(out) >= k:
            break
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def choose_landmarks_random(n: int, k: int, s: int, t: int, rng: random.Random) -> List[int]:
    candidates = [v for v in range(n) if v not in (s, t)]
    if not candidates or k <= 0:
        return []
    k = min(k, len(candidates))
    return rng.sample(candidates, k)


def alt_preprocess(
    adj: List[List[Tuple[int, int]]],
    landmarks: List[int],
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Precompute:
    - dist_from[i][v] = d(L_i, v)
    - dist_to[i][v]   = d(v, L_i)   (computed by running Dijkstra on the reversed graph from L_i)
    """
    radj = reverse_adjacency(adj)
    dist_from = []
    dist_to = []
    for L in landmarks:
        dist_from.append(dijkstra_all(adj, L))
        dist_to.append(dijkstra_all(radj, L))
    return dist_from, dist_to


def make_alt_heuristic(
    t: int,
    landmarks: List[int],
    dist_from: List[List[float]],
    dist_to: List[List[float]],
) -> Callable[[int], float]:
    """
    ALT admissible heuristic (directed version) using triangle-inequality lower bounds:

    For any landmark L:
      d(v,t) >= d(L,t) - d(L,v)     (if both finite)
      d(v,t) >= d(v,L) - d(t,L)     (if both finite)

    We take the maximum over landmarks and clamp below by 0.
    """
    k = len(landmarks)

    def h(v: int) -> float:
        best = 0.0
        for i in range(k):
            dLv = dist_from[i][v]
            dLt = dist_from[i][t]
            if dLv < INF and dLt < INF:
                best = max(best, dLt - dLv)

            dvL = dist_to[i][v]
            dtL = dist_to[i][t]
            if dvL < INF and dtL < INF:
                best = max(best, dvL - dtL)

        if best < 0.0:
            return 0.0
        return best

    return h


def alt_query(
    adj: List[List[Tuple[int, int]]],
    s: int,
    t: int,
    landmarks: List[int],
    dist_from: List[List[float]],
    dist_to: List[List[float]],
) -> Tuple[float, Optional[List[int]]]:
    h_func = make_alt_heuristic(t, landmarks, dist_from, dist_to)
    # Build a list heuristic for speed in this run
    n = len(adj)
    h_list = [h_func(v) for v in range(n)]
    return astar_shortest_path_admissible(adj, s, t, h_list)


# -----------------------------
# Benchmarking
# -----------------------------

def time_it(fn, *args, **kwargs):
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    end = time.perf_counter()
    return end - start, out


def benchmark_one_graph(
    adj: List[List[Tuple[int, int]]],
    s: int,
    t: int,
    alt_k: int,
    alt_landmarks_method: str,
    rng: random.Random,
    run_exhaustive: bool,
) -> dict:
    n = len(adj)
    results = {}

    # Ground truth and timing for exhaustive enumeration (optional)
    if run_exhaustive:
        te, (de, pe) = time_it(exhaustive_shortest_path_dag, adj, s, t)
        results["exhaustive_time"] = te
        results["exhaustive_dist"] = de
        results["exhaustive_path"] = pe
        truth_dist = de
    else:
        # Use Dijkstra as ground truth (always correct for positive weights)
        truth_dist = None
        results["exhaustive_time"] = math.nan
        results["exhaustive_dist"] = math.nan
        results["exhaustive_path"] = None

    # Dijkstra
    td, (dd, pd) = time_it(dijkstra_shortest_path, adj, s, t)
    results["dijkstra_time"] = td
    results["dijkstra_dist"] = dd
    results["dijkstra_path"] = pd

    if truth_dist is None:
        truth_dist = dd

    # A*
    h_astar = make_min_outgoing_heuristic(adj, t)
    ta, (da, pa) = time_it(astar_shortest_path_admissible, adj, s, t, h_astar)
    results["astar_time"] = ta
    results["astar_dist"] = da
    results["astar_path"] = pa

    # ALT (preprocessing + query, and query-only)
    if alt_k > 0:
        if alt_landmarks_method == "even":
            landmarks = choose_landmarks_even(n, alt_k, s, t)
        else:
            landmarks = choose_landmarks_random(n, alt_k, s, t, rng)

        tp, (dist_from, dist_to) = time_it(alt_preprocess, adj, landmarks)
        tq, (dl, pl) = time_it(alt_query, adj, s, t, landmarks, dist_from, dist_to)

        results["alt_landmarks"] = landmarks
        results["alt_preprocess_time"] = tp
        results["alt_query_time"] = tq
        results["alt_total_time"] = tp + tq
        results["alt_dist"] = dl
        results["alt_path"] = pl
    else:
        results["alt_landmarks"] = []
        results["alt_preprocess_time"] = math.nan
        results["alt_query_time"] = math.nan
        results["alt_total_time"] = math.nan
        results["alt_dist"] = math.nan
        results["alt_path"] = None

    # Consistency checks (distances should match)
    tol = 1e-9
    if abs(results["dijkstra_dist"] - truth_dist) > tol:
        raise RuntimeError("Dijkstra result disagrees with ground truth.")
    if abs(results["astar_dist"] - truth_dist) > tol:
        raise RuntimeError("A* result disagrees with ground truth.")
    if alt_k > 0 and abs(results["alt_dist"] - truth_dist) > tol:
        raise RuntimeError("ALT result disagrees with ground truth.")

    results["truth_dist"] = truth_dist
    return results


def summarize(values: List[float]) -> Tuple[float, float]:
    vals = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return math.nan, math.nan
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) >= 2 else 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark shortest path algorithms on random connected DAGs.")
    parser.add_argument("--n", type=int, default=12, help="Number of nodes (>=2).")
    parser.add_argument("--trials", type=int, default=5, help="Number of random graphs to test.")
    parser.add_argument("--edge-prob", type=float, default=0.12, help="Probability of an extra edge i->j (i<j).")
    parser.add_argument("--w-min", type=int, default=1, help="Minimum positive edge weight.")
    parser.add_argument("--w-max", type=int, default=20, help="Maximum edge weight.")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    parser.add_argument("--alt-k", type=int, default=2, help="Number of landmarks for ALT (0 disables ALT).")
    parser.add_argument("--alt-landmarks", choices=["even", "random"], default="even",
                        help="Landmark selection strategy for ALT.")
    parser.add_argument("--skip-exhaustive", action="store_true",
                        help="Skip exhaustive enumeration (recommended for larger n or denser graphs).")
    args = parser.parse_args()

    n = args.n
    trials = args.trials
    edge_prob = args.edge_prob
    w_min, w_max = args.w_min, args.w_max

    if not (0.0 <= edge_prob <= 1.0):
        raise ValueError("--edge-prob must be in [0,1].")
    if w_min <= 0 or w_max < w_min:
        raise ValueError("Weights must satisfy 1 <= w_min <= w_max.")
    if n < 2:
        raise ValueError("--n must be at least 2.")
    if trials < 1:
        raise ValueError("--trials must be at least 1.")

    s, t = 0, n - 1

    times_exh = []
    times_dij = []
    times_ast = []
    times_alt_pre = []
    times_alt_q = []
    times_alt_tot = []

    # For reproducibility: each trial uses an independent RNG stream
    base_rng = random.Random(args.seed)

    for k in range(trials):
        trial_seed = base_rng.randint(0, 10**9)
        rng = random.Random(trial_seed)

        adj = generate_random_connected_dag(
            n=n,
            edge_prob=edge_prob,
            w_min=w_min,
            w_max=w_max,
            rng=rng,
        )

        res = benchmark_one_graph(
            adj=adj,
            s=s,
            t=t,
            alt_k=args.alt_k,
            alt_landmarks_method=args.alt_landmarks,
            rng=rng,
            run_exhaustive=(not args.skip_exhaustive),
        )

        times_exh.append(res["exhaustive_time"])
        times_dij.append(res["dijkstra_time"])
        times_ast.append(res["astar_time"])
        times_alt_pre.append(res["alt_preprocess_time"])
        times_alt_q.append(res["alt_query_time"])
        times_alt_tot.append(res["alt_total_time"])

        print(
            f"trial={k+1}/{trials}  n={n}  truth_dist={res['truth_dist']:.6g}  "
            f"exh={res['exhaustive_time']:.6g}s  dij={res['dijkstra_time']:.6g}s  "
            f"astar={res['astar_time']:.6g}s  "
            f"alt_pre={res['alt_preprocess_time']:.6g}s  alt_q={res['alt_query_time']:.6g}s  alt_tot={res['alt_total_time']:.6g}s"
        )

    mean_exh, sd_exh = summarize(times_exh)
    mean_dij, sd_dij = summarize(times_dij)
    mean_ast, sd_ast = summarize(times_ast)
    mean_pre, sd_pre = summarize(times_alt_pre)
    mean_q, sd_q = summarize(times_alt_q)
    mean_tot, sd_tot = summarize(times_alt_tot)

    print("\nSummary (mean ± sd over trials):")
    if not args.skip_exhaustive:
        print(f"  Exhaustive: {mean_exh:.6g} ± {sd_exh:.6g} s")
    else:
        print("  Exhaustive: (skipped)")
    print(f"  Dijkstra:   {mean_dij:.6g} ± {sd_dij:.6g} s")
    print(f"  A*:         {mean_ast:.6g} ± {sd_ast:.6g} s")
    if args.alt_k > 0:
        print(f"  ALT pre:    {mean_pre:.6g} ± {sd_pre:.6g} s")
        print(f"  ALT query:  {mean_q:.6g} ± {sd_q:.6g} s")
        print(f"  ALT total:  {mean_tot:.6g} ± {sd_tot:.6g} s")
    else:
        print("  ALT: (disabled)")


if __name__ == "__main__":
    main()
