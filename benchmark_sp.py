#!/usr/bin/env python3  # Use the system’s Python 3 interpreter when run as an executable script.
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

import argparse  # Parse command-line arguments.
import heapq  # Priority queue operations for Dijkstra/A*.
import math  # NaN/inf and numeric utilities.
import random  # Random graph generation and RNG seeding.
import statistics  # Mean/stdev summaries for timings.
import time  # High-resolution timers for benchmarking.
from typing import Callable, List, Optional, Tuple  # Type annotations for clarity and tooling.


INF = float("inf")  # Sentinel value representing “infinite distance” / unreachable.


# -----------------------------  # Visual separator for readability.
# Graph generation (random DAG)  # Section header: graph generation helpers.
# -----------------------------  # Visual separator for readability.

def generate_random_connected_dag(  # Build a random connected DAG with positive weights.
    n: int,  # Number of nodes.
    edge_prob: float,  # Probability of adding each non-backbone forward edge.
    w_min: int,  # Minimum edge weight (positive).
    w_max: int,  # Maximum edge weight.
    rng: random.Random,  # RNG instance (so we can control seeds reproducibly).
) -> List[List[Tuple[int, int]]]:  # Return adjacency list: adj[u] = list of (v, w).
    """
    Returns adjacency list for a random connected DAG on nodes 0..n-1.
    Acyclic: only edges i->j with i<j.
    Connected (reachability s->all and all->t): includes backbone edges i->i+1.
    """  # Docstring explains constraints and guarantees.
    if n < 2:  # A shortest-path s->t only makes sense with at least two nodes.
        raise ValueError("n must be at least 2")  # Fail early with a clear message.

    adj: List[List[Tuple[int, int]]] = [[] for _ in range(n)]  # Initialize empty adjacency lists.
    has_edge = set()  # Track existing directed edges to avoid duplicates.

    def add_edge(u: int, v: int, w: int) -> None:  # Local helper to add edge if not already present.
        if (u, v) in has_edge:  # If edge already exists…
            return  # …do nothing (keep first weight).
        has_edge.add((u, v))  # Record presence of the edge.
        adj[u].append((v, w))  # Append neighbor and weight to adjacency list.

    # Backbone: guarantees reachability and a valid s->t path  # Explain why backbone is added.
    for i in range(n - 1):  # For each consecutive pair (i, i+1)…
        w = rng.randint(w_min, w_max)  # Sample a positive integer weight.
        add_edge(i, i + 1, w)  # Add the backbone edge i -> i+1.

    # Extra random edges (still respecting acyclicity i<j)  # Explain “forward-only” edges.
    for i in range(n):  # Iterate possible source nodes.
        for j in range(i + 1, n):  # Only consider forward targets j>i to keep a DAG.
            if (i, j) in has_edge:  # Skip if backbone or earlier addition already created it.
                continue  # Avoid duplicates.
            if rng.random() < edge_prob:  # Bernoulli trial for adding this edge.
                w = rng.randint(w_min, w_max)  # Sample its weight.
                add_edge(i, j, w)  # Add the extra edge i -> j.

    # Optional: sort adjacency for reproducibility (not required)  # Sorting helps stable outputs.
    for u in range(n):  # For each node…
        adj[u].sort(key=lambda x: x[0])  # Sort outgoing edges by neighbor id.

    return adj  # Return the generated adjacency list.


def reverse_adjacency(adj: List[List[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:  # Build reversed graph.
    n = len(adj)  # Number of nodes in the graph.
    radj: List[List[Tuple[int, int]]] = [[] for _ in range(n)]  # Initialize reverse adjacency lists.
    for u in range(n):  # For each node u…
        for v, w in adj[u]:  # For each outgoing edge u -> v with weight w…
            radj[v].append((u, w))  # Add reversed edge v -> u with same weight.
    for v in range(n):  # For each node in the reversed graph…
        radj[v].sort(key=lambda x: x[0])  # Sort for reproducibility.
    return radj  # Return reversed adjacency list.


# -----------------------------  # Visual separator for readability.
# Utilities (path reconstruction)  # Section header: reconstructing paths from parents.
# -----------------------------  # Visual separator for readability.

def reconstruct_path(parent: List[int], s: int, t: int) -> Optional[List[int]]:  # Rebuild s->t path from parent pointers.
    if s == t:  # Trivial case: start equals target.
        return [s]  # Path is just the single node.
    if parent[t] == -1:  # If t has no parent, it was never reached.
        return None  # No path exists (or wasn’t found).
    path = []  # Accumulate nodes from t back to s.
    cur = t  # Start backtracking from the target.
    while cur != -1:  # Follow parent pointers until sentinel -1.
        path.append(cur)  # Record current node in reverse order.
        if cur == s:  # If we reached the start…
            break  # …stop backtracking.
        cur = parent[cur]  # Move one step toward the start.
    if path[-1] != s:  # If we exited without actually reaching s…
        return None  # …then there is no valid s->t chain.
    path.reverse()  # Reverse to produce s->…->t order.
    return path  # Return the reconstructed path.


# -----------------------------  # Visual separator for readability.
# Algorithm 1: Exhaustive enumeration (backtracking)  # Section header: exponential baseline.
# -----------------------------  # Visual separator for readability.

def exhaustive_shortest_path_dag(  # Compute shortest s->t path by enumerating all s->t paths (DAG-friendly).
    adj: List[List[Tuple[int, int]]],  # Adjacency list of the graph.
    s: int,  # Source node.
    t: int,  # Target node.
) -> Tuple[float, Optional[List[int]]]:  # Return (distance, path) where path may be None.
    """
    Enumerates all directed s->t paths and returns the minimum-cost path.
    Works for any directed graph if you include cycle-avoidance; here the graph is a DAG by construction.
    """  # Docstring clarifies intent and complexity.
    n = len(adj)  # Number of nodes.
    best_cost = INF  # Best path cost found so far.
    best_parent = [-1] * n  # Snapshot of parent pointers for the best path.
    parent = [-1] * n  # Current DFS parent pointers for the path being explored.

    # In a DAG, cycle-avoidance is unnecessary, but keeping a recursion stack is harmless and defensive.  # Rationale.
    in_stack = [False] * n  # Marks nodes currently on the recursion stack.

    def dfs(u: int, cost_so_far: float) -> None:  # Depth-first search over all s->t paths.
        nonlocal best_cost, best_parent  # Allow inner function to update outer variables.

        if cost_so_far >= best_cost:  # Prune: current partial path already worse than best.
            return  # Stop exploring this branch.

        if u == t:  # If we reached the target…
            best_cost = cost_so_far  # …record the cost.
            best_parent = parent[:]  # …snapshot parents to reconstruct the path later.
            return  # Done for this path.

        in_stack[u] = True  # Mark u as currently in recursion stack.
        for v, w in adj[u]:  # Try each outgoing edge u -> v.
            if in_stack[v]:  # Defensive: avoid cycles (shouldn’t happen in a DAG).
                continue  # Skip edges that would create a back-edge in the recursion.
            parent[v] = u  # Set parent pointer for v in current DFS path.
            dfs(v, cost_so_far + w)  # Recurse with accumulated cost.
            parent[v] = -1  # Undo parent pointer (backtrack) after returning.
        in_stack[u] = False  # Unmark u when leaving this recursion frame.

    parent[s] = -1  # Source has no parent by convention.
    dfs(s, 0.0)  # Start DFS at source with zero initial cost.
    if best_cost == INF:  # If no path was found…
        return INF, None  # …report unreachable.
    return best_cost, reconstruct_path(best_parent, s, t)  # Return best cost and its reconstructed path.


# -----------------------------  # Visual separator for readability.
# Algorithm 2: Dijkstra (single-source single-target)  # Section header: standard baseline for nonnegative weights.
# -----------------------------  # Visual separator for readability.

def dijkstra_shortest_path(  # Compute shortest s->t path using Dijkstra’s algorithm.
    adj: List[List[Tuple[int, int]]],  # Adjacency list of the graph.
    s: int,  # Source node.
    t: int,  # Target node.
) -> Tuple[float, Optional[List[int]]]:  # Return (distance, path) where path may be None.
    n = len(adj)  # Number of nodes.
    dist = [INF] * n  # Distance labels initialized to infinity.
    parent = [-1] * n  # Parent pointers to reconstruct the shortest path.
    dist[s] = 0.0  # Distance to the source is zero.

    pq: List[Tuple[float, int]] = [(0.0, s)]  # Min-heap of (distance, node) for extracting next closest node.
    settled = [False] * n  # Whether we have finalized the shortest distance for each node.

    while pq:  # Continue until no reachable nodes remain in the heap.
        du, u = heapq.heappop(pq)  # Extract the currently smallest tentative distance node.
        if settled[u]:  # If u was already finalized via an earlier heap entry…
            continue  # …skip this stale entry.
        settled[u] = True  # Finalize u (its dist is now permanent).

        if u == t:  # Early exit: once target is settled, its distance is optimal.
            break  # Stop searching.

        for v, w in adj[u]:  # Relax all outgoing edges u -> v.
            if settled[v]:  # If v is already finalized…
                continue  # …no need to relax it.
            alt = du + w  # Candidate distance via u.
            if alt < dist[v]:  # If we found an improvement…
                dist[v] = alt  # …update best known distance.
                parent[v] = u  # …record how we got to v.
                heapq.heappush(pq, (alt, v))  # …push updated label into heap.

    if dist[t] == INF:  # If target is unreachable…
        return INF, None  # …report no path.
    return dist[t], reconstruct_path(parent, s, t)  # Return optimal distance and a reconstructed path.


# -----------------------------  # Visual separator for readability.
# Algorithm 3: A* (admissible heuristic)  # Section header: heuristic search.
# -----------------------------  # Visual separator for readability.

def make_min_outgoing_heuristic(adj: List[List[Tuple[int, int]]], t: int) -> List[float]:  # Build a cheap admissible heuristic.
    """
    Admissible heuristic for positive-weight directed graphs:
    h(v) = min outgoing edge cost from v, and h(t)=0.
    Reason: any path from v to t must take at least one outgoing edge (unless v=t).
    """  # Explain why this is a lower bound (admissible).
    n = len(adj)  # Number of nodes.
    h = [0.0] * n  # Initialize heuristic values.
    for v in range(n):  # For every node v…
        if v == t:  # At the goal…
            h[v] = 0.0  # …heuristic must be zero to be admissible and intuitive.
        else:  # For non-goal nodes…
            # By our generator, v<n-1 always has at least the backbone edge to v+1  # Guarantee for DAG generator.
            h[v] = min(w for _, w in adj[v]) if adj[v] else 0.0  # Lower bound: must pay at least one outgoing edge.
    return h  # Return heuristic list aligned with node ids.


def astar_shortest_path_admissible(  # A* that remains correct even if heuristic is admissible but inconsistent.
    adj: List[List[Tuple[int, int]]],  # Adjacency list of the graph.
    s: int,  # Source node.
    t: int,  # Target node.
    h: List[float],  # Heuristic estimates h[v] <= true d(v,t).
) -> Tuple[float, Optional[List[int]]]:  # Return (distance, path) where path may be None.
    """
    A* that is correct for admissible (not necessarily consistent) heuristics:
    it terminates when the smallest f-value in the open set is >= best known goal cost.
    """  # Clarify the stopping condition used for admissible-only heuristics.
    n = len(adj)  # Number of nodes.
    g = [INF] * n  # Best-known cost-to-come values.
    parent = [-1] * n  # Parent pointers for reconstructing best-known paths.
    g[s] = 0.0  # Cost-to-come for the start is zero.

    pq: List[Tuple[float, float, int]] = [(h[s], 0.0, s)]  # Open set as heap of (f=g+h, g, node).
    best_goal = INF  # Best goal cost found so far (upper bound).

    while pq:  # Keep expanding until open set is empty or we can prove optimality.
        f_u, g_u, u = heapq.heappop(pq)  # Pop node with smallest f-value (tie-broken by heap order).
        if g_u != g[u]:  # If this entry is stale (not equal to current best g)…
            continue  # …skip it.

        if u == t:  # If we popped the goal…
            best_goal = g_u  # …update best known goal cost (may improve multiple times with inadmissible? but here admissible).
            # Note: we do not immediately break; we wait until no open node can beat best_goal.  # Rationale.

        # If a goal path is known and no node can lead to a cheaper one, stop.  # Admissible-only safe termination.
        if best_goal < INF:  # If we have some goal solution…
            if not pq or pq[0][0] >= best_goal:  # …and the best remaining f is not better than it…
                break  # …then best_goal is provably optimal.

        for v, w in adj[u]:  # Relax outgoing edges from u.
            alt = g_u + w  # Candidate g-value for v via u.
            if alt < g[v]:  # If this is a better path-to-come…
                g[v] = alt  # …update best g.
                parent[v] = u  # …store predecessor for reconstruction.
                heapq.heappush(pq, (alt + h[v], alt, v))  # Push new open-set entry with updated f and g.

    if best_goal == INF:  # If goal was never reached…
        return INF, None  # …report no path.
    return best_goal, reconstruct_path(parent, s, t)  # Return optimal distance and reconstructed path.


# -----------------------------  # Visual separator for readability.
# Algorithm 4: ALT (Landmarks + A*)  # Section header: landmark-based heuristic.
# -----------------------------  # Visual separator for readability.

def dijkstra_all(adj: List[List[Tuple[int, int]]], source: int) -> List[float]:  # Run Dijkstra from one source to all nodes.
    n = len(adj)  # Number of nodes.
    dist = [INF] * n  # Initialize all distances to infinity.
    dist[source] = 0.0  # Source distance is zero.
    pq: List[Tuple[float, int]] = [(0.0, source)]  # Heap of (distance, node).
    settled = [False] * n  # Track which nodes have been finalized.

    while pq:  # Continue until all reachable nodes are processed.
        du, u = heapq.heappop(pq)  # Extract node with smallest tentative distance.
        if settled[u]:  # Skip if already finalized (stale heap entry).
            continue  # Ignore duplicates.
        settled[u] = True  # Finalize u.
        for v, w in adj[u]:  # Relax edges out of u.
            alt = du + w  # Candidate distance to v.
            if alt < dist[v]:  # Improvement check.
                dist[v] = alt  # Update best known distance.
                heapq.heappush(pq, (alt, v))  # Push updated entry.
    return dist  # Return all-pairs-from-source distances.


def choose_landmarks_even(n: int, k: int, s: int, t: int) -> List[int]:  # Choose landmarks deterministically and evenly.
    """
    Deterministic, simple landmark selection: spread landmarks along the topological order.
    This is not optimal in general, but it is a transparent, reproducible choice for experiments.
    """  # Explain why this selection is used (clarity > optimality).
    candidates = [v for v in range(n) if v not in (s, t)]  # Eligible landmarks: all nodes except endpoints.
    if not candidates or k <= 0:  # If there are no candidates or k is nonpositive…
        return []  # …no landmarks.
    k = min(k, len(candidates))  # Cap k so we don’t request more landmarks than candidates.
    # pick approximately evenly spaced indices  # Describe the strategy.
    idxs = []  # Hold chosen candidate nodes (possibly with duplicates due to rounding).
    for i in range(1, k + 1):  # Choose k positions.
        pos = round(i * (len(candidates) - 1) / (k + 1))  # Map i to an index spread across candidates.
        idxs.append(candidates[pos])  # Add that candidate as a landmark pick.
    # remove duplicates while preserving order  # Rounding can create duplicates; dedupe stably.
    seen = set()  # Track which landmarks have already been kept.
    out = []  # Final landmark list (order-preserving).
    for v in idxs:  # Iterate preliminary selections…
        if v not in seen:  # If it’s new…
            seen.add(v)  # Mark it seen.
            out.append(v)  # Keep it.
    # If duplicates reduced count, fill from left to right  # Ensure we end with exactly k landmarks.
    for v in candidates:  # Scan candidates in order…
        if len(out) >= k:  # Stop once we have k.
            break  # Done filling.
        if v not in seen:  # If not already chosen…
            seen.add(v)  # Mark seen.
            out.append(v)  # Add to output.
    return out  # Return landmarks.


def choose_landmarks_random(n: int, k: int, s: int, t: int, rng: random.Random) -> List[int]:  # Choose landmarks uniformly at random.
    candidates = [v for v in range(n) if v not in (s, t)]  # Eligible landmark nodes (exclude endpoints).
    if not candidates or k <= 0:  # Handle empty candidate set or k<=0.
        return []  # Return no landmarks.
    k = min(k, len(candidates))  # Cap k to available candidates.
    return rng.sample(candidates, k)  # Randomly sample k distinct landmarks.


def alt_preprocess(  # Precompute landmark distances needed for the ALT heuristic.
    adj: List[List[Tuple[int, int]]],  # Forward adjacency list.
    landmarks: List[int],  # Landmark node ids.
) -> Tuple[List[List[float]], List[List[float]]]:  # Return (dist_from, dist_to) matrices.
    """
    Precompute:
    - dist_from[i][v] = d(L_i, v)
    - dist_to[i][v]   = d(v, L_i)   (computed by running Dijkstra on the reversed graph from L_i)
    """  # ALT needs distances to and from landmarks to form lower bounds via triangle inequality.
    radj = reverse_adjacency(adj)  # Build reversed graph to compute “to landmark” distances via a forward run.
    dist_from = []  # dist_from[i] will be a list of distances from landmark i to all v.
    dist_to = []  # dist_to[i] will be a list of distances from all v to landmark i.
    for L in landmarks:  # For each landmark…
        dist_from.append(dijkstra_all(adj, L))  # Run Dijkstra from L on the original graph.
        dist_to.append(dijkstra_all(radj, L))  # Run Dijkstra from L on reversed graph = distances to L in original.
    return dist_from, dist_to  # Return precomputed tables.


def make_alt_heuristic(  # Build an admissible ALT heuristic function for a fixed target t.
    t: int,  # Target node.
    landmarks: List[int],  # Landmark nodes used in the heuristic.
    dist_from: List[List[float]],  # dist_from[i][v] = d(L_i, v).
    dist_to: List[List[float]],  # dist_to[i][v] = d(v, L_i).
) -> Callable[[int], float]:  # Return a callable h(v) giving a lower bound on d(v,t).
    """
    ALT admissible heuristic (directed version) using triangle-inequality lower bounds:

    For any landmark L:
      d(v,t) >= d(L,t) - d(L,v)     (if both finite)
      d(v,t) >= d(v,L) - d(t,L)     (if both finite)

    We take the maximum over landmarks and clamp below by 0.
    """  # Explain the two directed lower bounds used.
    k = len(landmarks)  # Number of landmarks.

    def h(v: int) -> float:  # Heuristic value for node v.
        best = 0.0  # Track maximum lower bound over landmarks.
        for i in range(k):  # For each landmark index i…
            dLv = dist_from[i][v]  # Distance from landmark L_i to v.
            dLt = dist_from[i][t]  # Distance from landmark L_i to target t.
            if dLv < INF and dLt < INF:  # Only use bound if both distances are finite.
                best = max(best, dLt - dLv)  # Bound: d(v,t) >= d(L,t) - d(L,v).

            dvL = dist_to[i][v]  # Distance from v to landmark L_i (via reversed-graph preprocessing).
            dtL = dist_to[i][t]  # Distance from t to landmark L_i.
            if dvL < INF and dtL < INF:  # Only use bound if both distances are finite.
                best = max(best, dvL - dtL)  # Bound: d(v,t) >= d(v,L) - d(t,L).

        if best < 0.0:  # Numerical/edge-case guard: lower bounds shouldn’t be negative as distances are nonnegative.
            return 0.0  # Clamp to 0 to keep heuristic admissible and sane.
        return best  # Return best (max) lower bound.

    return h  # Return the heuristic function.


def alt_query(  # Run ALT query (A* using landmark heuristic) for a single s->t query.
    adj: List[List[Tuple[int, int]]],  # Adjacency list of the graph.
    s: int,  # Source node.
    t: int,  # Target node.
    landmarks: List[int],  # Landmark nodes.
    dist_from: List[List[float]],  # Precomputed distances from landmarks.
    dist_to: List[List[float]],  # Precomputed distances to landmarks.
) -> Tuple[float, Optional[List[int]]]:  # Return (distance, path) where path may be None.
    h_func = make_alt_heuristic(t, landmarks, dist_from, dist_to)  # Build landmark-based heuristic for this target.
    # Build a list heuristic for speed in this run  # Pre-evaluate h on all nodes to avoid repeated computation.
    n = len(adj)  # Number of nodes.
    h_list = [h_func(v) for v in range(n)]  # Compute heuristic values for every node.
    return astar_shortest_path_admissible(adj, s, t, h_list)  # Run admissible-safe A* with this heuristic.


# -----------------------------  # Visual separator for readability.
# Benchmarking  # Section header: timing and repeated trials.
# -----------------------------  # Visual separator for readability.

def time_it(fn, *args, **kwargs):  # Measure wall-clock time of a function call.
    start = time.perf_counter()  # Start high-resolution timer.
    out = fn(*args, **kwargs)  # Execute the function and capture its return value.
    end = time.perf_counter()  # Stop high-resolution timer.
    return end - start, out  # Return elapsed time plus the function’s output.


def benchmark_one_graph(  # Run all algorithms on one graph and record timings/results.
    adj: List[List[Tuple[int, int]]],  # Graph adjacency list.
    s: int,  # Source node.
    t: int,  # Target node.
    alt_k: int,  # Number of landmarks for ALT (0 disables ALT).
    alt_landmarks_method: str,  # Landmark selection method: "even" or "random".
    rng: random.Random,  # RNG for landmark selection when random.
    run_exhaustive: bool,  # Whether to run exponential exhaustive baseline.
) -> dict:  # Return a dict of results/timings/paths.
    n = len(adj)  # Number of nodes.
    results = {}  # Dictionary to accumulate outputs.

    # Ground truth and timing for exhaustive enumeration (optional)  # Explain the role of exhaustive search.
    if run_exhaustive:  # If enabled…
        te, (de, pe) = time_it(exhaustive_shortest_path_dag, adj, s, t)  # Time exhaustive search and capture (dist, path).
        results["exhaustive_time"] = te  # Store exhaustive runtime.
        results["exhaustive_dist"] = de  # Store exhaustive distance.
        results["exhaustive_path"] = pe  # Store exhaustive path.
        truth_dist = de  # Treat exhaustive result as ground truth.
    else:  # If exhaustive is skipped…
        # Use Dijkstra as ground truth (always correct for positive weights)  # Dijkstra is reliable under nonnegative weights.
        truth_dist = None  # Placeholder until Dijkstra runs.
        results["exhaustive_time"] = math.nan  # Mark missing time with NaN.
        results["exhaustive_dist"] = math.nan  # Mark missing dist with NaN.
        results["exhaustive_path"] = None  # No path stored.

    # Dijkstra  # Start timing Dijkstra.
    td, (dd, pd) = time_it(dijkstra_shortest_path, adj, s, t)  # Time Dijkstra and capture (dist, path).
    results["dijkstra_time"] = td  # Store Dijkstra runtime.
    results["dijkstra_dist"] = dd  # Store Dijkstra distance.
    results["dijkstra_path"] = pd  # Store Dijkstra path.

    if truth_dist is None:  # If we skipped exhaustive…
        truth_dist = dd  # …use Dijkstra’s distance as the reference truth.

    # A*  # Start timing A*.
    h_astar = make_min_outgoing_heuristic(adj, t)  # Build an admissible heuristic list for A*.
    ta, (da, pa) = time_it(astar_shortest_path_admissible, adj, s, t, h_astar)  # Time A* and capture results.
    results["astar_time"] = ta  # Store A* runtime.
    results["astar_dist"] = da  # Store A* distance.
    results["astar_path"] = pa  # Store A* path.

    # ALT (preprocessing + query, and query-only)  # ALT has a setup phase plus the search phase.
    if alt_k > 0:  # Only run ALT if landmarks requested.
        if alt_landmarks_method == "even":  # Deterministic landmark selection…
            landmarks = choose_landmarks_even(n, alt_k, s, t)  # …choose evenly spaced landmarks.
        else:  # Otherwise…
            landmarks = choose_landmarks_random(n, alt_k, s, t, rng)  # …choose random landmarks.

        tp, (dist_from, dist_to) = time_it(alt_preprocess, adj, landmarks)  # Time preprocessing (building landmark distance tables).
        tq, (dl, pl) = time_it(alt_query, adj, s, t, landmarks, dist_from, dist_to)  # Time query with ALT heuristic.

        results["alt_landmarks"] = landmarks  # Store which landmarks were used.
        results["alt_preprocess_time"] = tp  # Store preprocessing time.
        results["alt_query_time"] = tq  # Store query-only time.
        results["alt_total_time"] = tp + tq  # Store total time (prep + query).
        results["alt_dist"] = dl  # Store ALT distance.
        results["alt_path"] = pl  # Store ALT path.
    else:  # ALT disabled case.
        results["alt_landmarks"] = []  # No landmarks used.
        results["alt_preprocess_time"] = math.nan  # No preprocessing time.
        results["alt_query_time"] = math.nan  # No query time.
        results["alt_total_time"] = math.nan  # No total time.
        results["alt_dist"] = math.nan  # No distance recorded.
        results["alt_path"] = None  # No path recorded.

    # Consistency checks (distances should match)  # Validate correctness against truth_dist.
    tol = 1e-9  # Numeric tolerance for floating comparison.
    if abs(results["dijkstra_dist"] - truth_dist) > tol:  # Check Dijkstra matches ground truth.
        raise RuntimeError("Dijkstra result disagrees with ground truth.")  # Crash loudly on mismatch.
    if abs(results["astar_dist"] - truth_dist) > tol:  # Check A* matches ground truth.
        raise RuntimeError("A* result disagrees with ground truth.")  # Crash loudly on mismatch.
    if alt_k > 0 and abs(results["alt_dist"] - truth_dist) > tol:  # Check ALT matches ground truth when enabled.
        raise RuntimeError("ALT result disagrees with ground truth.")  # Crash loudly on mismatch.

    results["truth_dist"] = truth_dist  # Store the chosen ground-truth distance.
    return results  # Return all results for this graph/trial.


def summarize(values: List[float]) -> Tuple[float, float]:  # Compute mean and standard deviation, ignoring NaNs.
    vals = [v for v in values if not (isinstance(v, float) and math.isnan(v))]  # Filter out NaN entries.
    if not vals:  # If nothing left after filtering…
        return math.nan, math.nan  # …report NaN summary.
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) >= 2 else 0.0)  # Return (mean, stdev or 0).


def main() -> None:  # Main entry point: parse args, run trials, print summary.
    parser = argparse.ArgumentParser(description="Benchmark shortest path algorithms on random connected DAGs.")  # CLI parser.
    parser.add_argument("--n", type=int, default=12, help="Number of nodes (>=2).")  # Graph size.
    parser.add_argument("--trials", type=int, default=5, help="Number of random graphs to test.")  # Number of trials.
    parser.add_argument("--edge-prob", type=float, default=0.12, help="Probability of an extra edge i->j (i<j).")  # Density control.
    parser.add_argument("--w-min", type=int, default=1, help="Minimum positive edge weight.")  # Weight lower bound.
    parser.add_argument("--w-max", type=int, default=20, help="Maximum edge weight.")  # Weight upper bound.
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed.")  # Base seed for reproducible experiments.
    parser.add_argument("--alt-k", type=int, default=2, help="Number of landmarks for ALT (0 disables ALT).")  # ALT landmarks count.
    parser.add_argument("--alt-landmarks", choices=["even", "random"], default="even",  # ALT landmark selection options.
                        help="Landmark selection strategy for ALT.")  # Help text for landmark strategy.
    parser.add_argument("--skip-exhaustive", action="store_true",  # Flag option: if present, skip exhaustive.
                        help="Skip exhaustive enumeration (recommended for larger n or denser graphs).")  # Help text for skipping.
    args = parser.parse_args()  # Parse command-line args into a namespace.

    n = args.n  # Extract number of nodes.
    trials = args.trials  # Extract number of trials.
    edge_prob = args.edge_prob  # Extract extra-edge probability.
    w_min, w_max = args.w_min, args.w_max  # Extract weight bounds.

    if not (0.0 <= edge_prob <= 1.0):  # Validate edge probability is in [0,1].
        raise ValueError("--edge-prob must be in [0,1].")  # Fail early with clear error.
    if w_min <= 0 or w_max < w_min:  # Validate weight bounds are positive and ordered.
        raise ValueError("Weights must satisfy 1 <= w_min <= w_max.")  # Fail early with clear error.
    if n < 2:  # Validate graph has at least 2 nodes.
        raise ValueError("--n must be at least 2.")  # Fail early with clear error.
    if trials < 1:  # Validate trials is at least 1.
        raise ValueError("--trials must be at least 1.")  # Fail early with clear error.

    s, t = 0, n - 1  # Define source as 0 and target as n-1.

    times_exh = []  # Collect exhaustive runtimes per trial (maybe NaN if skipped).
    times_dij = []  # Collect Dijkstra runtimes per trial.
    times_ast = []  # Collect A* runtimes per trial.
    times_alt_pre = []  # Collect ALT preprocessing times per trial.
    times_alt_q = []  # Collect ALT query-only times per trial.
    times_alt_tot = []  # Collect ALT total times per trial.

    # For reproducibility: each trial uses an independent RNG stream  # Ensure trials are different but reproducible.
    base_rng = random.Random(args.seed)  # RNG that generates per-trial seeds.

    for k in range(trials):  # Repeat benchmark for the desired number of random graphs.
        trial_seed = base_rng.randint(0, 10**9)  # Draw a fresh seed for this trial.
        rng = random.Random(trial_seed)  # Create a trial-specific RNG for graph/landmarks.

        adj = generate_random_connected_dag(  # Build a random connected DAG for this trial.
            n=n,  # Pass node count.
            edge_prob=edge_prob,  # Pass edge probability.
            w_min=w_min,  # Pass minimum weight.
            w_max=w_max,  # Pass maximum weight.
            rng=rng,  # Pass trial RNG.
        )  # End graph generation call.

        res = benchmark_one_graph(  # Run all selected algorithms and gather results.
            adj=adj,  # Provide the graph.
            s=s,  # Provide source.
            t=t,  # Provide target.
            alt_k=args.alt_k,  # Provide ALT landmark count.
            alt_landmarks_method=args.alt_landmarks,  # Provide landmark selection strategy.
            rng=rng,  # Provide RNG (used if landmarks are random).
            run_exhaustive=(not args.skip_exhaustive),  # Decide whether to run exhaustive.
        )  # End benchmark call.

        times_exh.append(res["exhaustive_time"])  # Record exhaustive timing (or NaN if skipped).
        times_dij.append(res["dijkstra_time"])  # Record Dijkstra timing.
        times_ast.append(res["astar_time"])  # Record A* timing.
        times_alt_pre.append(res["alt_preprocess_time"])  # Record ALT preprocessing timing.
        times_alt_q.append(res["alt_query_time"])  # Record ALT query timing.
        times_alt_tot.append(res["alt_total_time"])  # Record ALT total timing.

        print(  # Print per-trial timing summary for quick feedback.
            f"trial={k+1}/{trials}  n={n}  truth_dist={res['truth_dist']:.6g}  "  # Print trial index, n, and truth distance.
            f"exh={res['exhaustive_time']:.6g}s  dij={res['dijkstra_time']:.6g}s  "  # Print exhaustive and Dijkstra times.
            f"astar={res['astar_time']:.6g}s  "  # Print A* time.
            f"alt_pre={res['alt_preprocess_time']:.6g}s  alt_q={res['alt_query_time']:.6g}s  alt_tot={res['alt_total_time']:.6g}s"  # Print ALT times.
        )  # End print call.

    mean_exh, sd_exh = summarize(times_exh)  # Compute mean/stdev for exhaustive runtimes (ignoring NaNs).
    mean_dij, sd_dij = summarize(times_dij)  # Compute mean/stdev for Dijkstra runtimes.
    mean_ast, sd_ast = summarize(times_ast)  # Compute mean/stdev for A* runtimes.
    mean_pre, sd_pre = summarize(times_alt_pre)  # Compute mean/stdev for ALT preprocessing runtimes.
    mean_q, sd_q = summarize(times_alt_q)  # Compute mean/stdev for ALT query runtimes.
    mean_tot, sd_tot = summarize(times_alt_tot)  # Compute mean/stdev for ALT total runtimes.

    print("\nSummary (mean ± sd over trials):")  # Print header for aggregate statistics.
    if not args.skip_exhaustive:  # If exhaustive was run…
        print(f"  Exhaustive: {mean_exh:.6g} ± {sd_exh:.6g} s")  # Print exhaustive mean ± sd.
    else:  # If exhaustive was skipped…
        print("  Exhaustive: (skipped)")  # Note that exhaustive stats are unavailable.
    print(f"  Dijkstra:   {mean_dij:.6g} ± {sd_dij:.6g} s")  # Print Dijkstra mean ± sd.
    print(f"  A*:         {mean_ast:.6g} ± {sd_ast:.6g} s")  # Print A* mean ± sd.
    if args.alt_k > 0:  # If ALT enabled…
        print(f"  ALT pre:    {mean_pre:.6g} ± {sd_pre:.6g} s")  # Print ALT preprocessing mean ± sd.
        print(f"  ALT query:  {mean_q:.6g} ± {sd_q:.6g} s")  # Print ALT query mean ± sd.
        print(f"  ALT total:  {mean_tot:.6g} ± {sd_tot:.6g} s")  # Print ALT total mean ± sd.
    else:  # If ALT disabled…
        print("  ALT: (disabled)")  # Note ALT is off.


if __name__ == "__main__":  # Standard Python entry-point guard (only run main when executed directly).
    main()  # Invoke main to execute the benchmark.
