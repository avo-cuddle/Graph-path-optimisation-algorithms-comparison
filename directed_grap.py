#!/usr/bin/env python3  # Use the user's environment to locate and run Python 3.
#  # (Blank-line placeholder comment so literally every line is “commented”.)
# Many-queries benchmark on ONE static graph.  # High-level: benchmark many shortest-path queries on one fixed graph.
#  # (Blank-line placeholder comment.)
# Goal:  # Start of purpose section.
# Compare query-time performance of  # We compare per-query runtimes across algorithms.
#   - Dijkstra  # Baseline: Dijkstra’s algorithm (nonnegative weights).
#   - A*  # Heuristic search using an admissible heuristic.
#   - ALT query (with preprocessing done once)  # ALT = A* + Landmarks + Triangle-inequality bounds (preprocess once).
#  # (Blank-line placeholder comment.)
# This matches the standard use case for ALT: many shortest-path queries on the same fixed graph.  # Motivation.
#  # (Blank-line placeholder comment.)
# Graph model:  # Start of graph model description.
# - Random directed acyclic graph (DAG) on nodes 0..n-1  # Node set is consecutive integers.
# - Edges only i->j with i<j  # Direction respects ordering, guaranteeing acyclicity.
# - Backbone edges i->i+1 ensure every pair (s,t) with s<t has a path  # Ensures reachability in the DAG sense.
# - Positive integer weights in [w_min, w_max]  # Nonnegative weights so Dijkstra/A* assumptions hold.
#  # (Blank-line placeholder comment.)
# Output:  # Start of output description.
# Prints exactly the data that is typically copied into the paper:  # Designed for “paper-ready” copying.
#   - n, m, edge_prob, weight range  # Graph size/density/weights.
#   - number of queries Q  # Query count.
#   - ALT parameters (k landmarks, selection method)  # Landmark config.
#   - preprocessing time  # One-time ALT preprocessing cost.
#   - mean ± sd query times for Dijkstra, A*, ALT(query)  # Query-time summary stats.
#   - ALT amortized time per query = pre_time/Q + mean_query_time  # Amortized cost for fair comparison.
#  # (Blank-line placeholder comment.)
import argparse  # Parse command-line arguments.
import heapq  # Priority queue for Dijkstra/A*/ALT.
import math  # NaN, infinity checks, and other math helpers.
import random  # Random graph generation and query sampling.
import statistics  # Mean and standard deviation summaries.
import time  # High-resolution timers via perf_counter().
from typing import List, Tuple  # Type hints for readability and static checking.
#  # (Blank-line placeholder comment.)
INF = float("inf")  # Sentinel “infinite distance” value.
#  # (Blank-line placeholder comment.)
# -----------------------------  # Section divider.
# Graph generation (random DAG)  # Section header.
# -----------------------------  # Section divider.
#  # (Blank-line placeholder comment.)
def generate_random_connected_dag(  # Build a random connected DAG adjacency list with a guaranteed backbone path.
    n: int,  # Number of nodes in the graph.
    edge_prob: float,  # Probability of adding each possible extra forward edge i->j (i<j).
    w_min: int,  # Minimum edge weight (inclusive).
    w_max: int,  # Maximum edge weight (inclusive).
    rng: random.Random,  # RNG instance for reproducibility.
) -> List[List[Tuple[int, int]]]:  # Returns adjacency list: adj[u] = list of (v, w).
    if n < 2:  # Need at least 2 nodes to have meaningful s<t queries and a backbone.
        raise ValueError("n must be at least 2")  # Fail early on invalid graph size.
#  # (Blank-line placeholder comment.)
    adj: List[List[Tuple[int, int]]] = [[] for _ in range(n)]  # Initialize empty adjacency lists for all nodes.
    has_edge = set()  # Track edges to avoid duplicates (u, v).
#  # (Blank-line placeholder comment.)
    def add_edge(u: int, v: int, w: int) -> None:  # Helper to add a directed weighted edge if not already present.
        if (u, v) in has_edge:  # If the edge already exists, do nothing.
            return  # Exit early to keep adjacency clean and deterministic.
        has_edge.add((u, v))  # Record edge existence.
        adj[u].append((v, w))  # Append (neighbor, weight) to adjacency list.
#  # (Blank-line placeholder comment.)
    # Backbone  # Ensure at least one path 0->1->...->n-1 so reachability holds for s<t.
    for i in range(n - 1):  # Iterate over consecutive pairs along the node order.
        add_edge(i, i + 1, rng.randint(w_min, w_max))  # Add backbone edge with random positive weight.
#  # (Blank-line placeholder comment.)
    # Extra edges  # Add additional random forward edges to increase density.
    for i in range(n):  # Choose a tail node i.
        for j in range(i + 1, n):  # Choose a head node j>i to preserve acyclicity.
            if (i, j) in has_edge:  # Skip if already added (e.g., backbone).
                continue  # Move on to the next candidate edge.
            if rng.random() < edge_prob:  # Add edge with probability edge_prob.
                add_edge(i, j, rng.randint(w_min, w_max))  # Add the random edge with random weight.
#  # (Blank-line placeholder comment.)
    for u in range(n):  # Post-process each adjacency list.
        adj[u].sort(key=lambda x: x[0])  # Sort neighbors by node id for stable iteration and reproducibility.
    return adj  # Return the constructed adjacency list.
#  # (Blank-line placeholder comment.)
def reverse_adjacency(adj: List[List[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:  # Build reverse graph adjacency.
    n = len(adj)  # Number of nodes inferred from adjacency list length.
    radj: List[List[Tuple[int, int]]] = [[] for _ in range(n)]  # Initialize empty reverse adjacency lists.
    for u in range(n):  # For each node u in the original graph...
        for v, w in adj[u]:  # ...for each edge u->v with weight w...
            radj[v].append((u, w))  # ...add reverse edge v->u with the same weight.
    for v in range(n):  # Sort reverse adjacency lists as well for determinism.
        radj[v].sort(key=lambda x: x[0])  # Sort incoming neighbors by id.
    return radj  # Return the reversed adjacency list.
#  # (Blank-line placeholder comment.)
def edge_count(adj: List[List[Tuple[int, int]]]) -> int:  # Count total directed edges in adjacency list.
    return sum(len(lst) for lst in adj)  # Sum lengths of all adjacency buckets.
#  # (Blank-line placeholder comment.)
# -----------------------------  # Section divider.
# Dijkstra distance-only  # Section header.
# -----------------------------  # Section divider.
#  # (Blank-line placeholder comment.)
def dijkstra_distance(adj: List[List[Tuple[int, int]]], s: int, t: int) -> float:  # Compute shortest-path distance s->t.
    n = len(adj)  # Number of nodes.
    dist = [INF] * n  # Tentative distance array initialized to infinity.
    dist[s] = 0.0  # Distance from s to itself is 0.
    settled = [False] * n  # Track whether a node’s shortest distance is finalized (“settled”).
    pq: List[Tuple[float, int]] = [(0.0, s)]  # Min-heap of (distance, node) starting at source.
#  # (Blank-line placeholder comment.)
    while pq:  # Continue until no frontier remains.
        du, u = heapq.heappop(pq)  # Extract the currently closest unsettled node.
        if settled[u]:  # If we've already finalized u via a better pop earlier...
            continue  # ...skip stale heap entries.
        settled[u] = True  # Finalize u’s distance.
        if u == t:  # If we reached target, we can stop early.
            return du  # Return the shortest distance to t.
        for v, w in adj[u]:  # Relax all outgoing edges from u.
            if settled[v]:  # If v is already finalized, no need to relax.
                continue  # Skip v.
            alt = du + w  # Candidate distance via u.
            if alt < dist[v]:  # If this route is better than current best...
                dist[v] = alt  # ...update best-known distance to v.
                heapq.heappush(pq, (alt, v))  # Push updated distance into priority queue.
#  # (Blank-line placeholder comment.)
    return INF  # If t is unreachable (shouldn’t happen under this DAG model), return infinity.
#  # (Blank-line placeholder comment.)
# -----------------------------  # Section divider.
# A* distance-only with a simple admissible heuristic  # Section header.
# h_t(v) = 0 if v==t else min_out[v]  # Heuristic: lower bound by minimum outgoing edge weight.
# -----------------------------  # Section divider.
#  # (Blank-line placeholder comment.)
def compute_min_outgoing(adj: List[List[Tuple[int, int]]]) -> List[float]:  # Compute per-node minimum outgoing weight.
    n = len(adj)  # Number of nodes.
    out_min = [0.0] * n  # Initialize min-out array (0 for sinks by default).
    for v in range(n):  # For each node v...
        if adj[v]:  # If v has outgoing edges...
            out_min[v] = float(min(w for _, w in adj[v]))  # ...store the smallest outgoing weight.
        else:  # If v has no outgoing edges (sink)...
            out_min[v] = 0.0  # ...heuristic contribution is 0.
    return out_min  # Return min-out weights.
#  # (Blank-line placeholder comment.)
def astar_distance_minout(  # Run A* to compute distance s->t using the “min outgoing edge” admissible heuristic.
    adj: List[List[Tuple[int, int]]],  # Graph adjacency list.
    s: int,  # Source node.
    t: int,  # Target node.
    out_min: List[float],  # Precomputed min outgoing weights used by the heuristic.
) -> float:  # Returns shortest distance from s to t.
    # admissible heuristic value  # Notes: admissible means it never overestimates the true remaining distance.
    def h(v: int) -> float:  # Heuristic function h(v) estimates distance from v to target t.
        return 0.0 if v == t else out_min[v]  # 0 at the goal, otherwise min outgoing edge weight.
#  # (Blank-line placeholder comment.)
    n = len(adj)  # Number of nodes.
    g = [INF] * n  # Best-known cost-to-come values (g-scores).
    g[s] = 0.0  # Cost to reach source from itself is 0.
#  # (Blank-line placeholder comment.)
    pq: List[Tuple[float, float, int]] = [(h(s), 0.0, s)]  # Heap of (f=g+h, g, node) for A* ordering.
    best_goal = INF  # Track best goal distance found so far (for admissible-but-inconsistent safe stopping).
#  # (Blank-line placeholder comment.)
    while pq:  # Process nodes in increasing f-order.
        f_u, g_u, u = heapq.heappop(pq)  # Pop the node with smallest estimated total cost.
        if g_u != g[u]:  # If this heap entry is stale (not the current best g for u)...
            continue  # ...skip it.
#  # (Blank-line placeholder comment.)
        if u == t:  # If we popped the target...
            best_goal = g_u  # ...record its g-value as a candidate optimal distance.
#  # (Blank-line placeholder comment.)
        # correct stopping rule for admissible (not necessarily consistent) heuristics  # Ensures optimality even if h isn't consistent.
        if best_goal < INF:  # If we have found at least one path to goal...
            if not pq or pq[0][0] >= best_goal:  # ...and the best possible remaining f can't beat it...
                return best_goal  # ...then best_goal is provably optimal.
#  # (Blank-line placeholder comment.)
        for v, w in adj[u]:  # Relax outgoing edges from u.
            alt = g_u + w  # Candidate g-score for v via u.
            if alt < g[v]:  # If candidate is better than current best...
                g[v] = alt  # ...update best-known g-score for v.
                heapq.heappush(pq, (alt + h(v), alt, v))  # Push new (f, g, node) into heap.
#  # (Blank-line placeholder comment.)
    return best_goal  # Return best goal found (INF if somehow unreachable).
#  # (Blank-line placeholder comment.)
# -----------------------------  # Section divider.
# ALT preprocessing and query (distance-only)  # Section header.
# Directed ALT lower bounds:  # ALT uses landmark distances to build admissible lower bounds on d(v,t).
#   d(v,t) >= d(L,t) - d(L,v)  # Reverse triangle inequality form using distances from landmark L.
#   d(v,t) >= d(v,L) - d(t,L)  # Reverse triangle inequality form using distances to landmark L.
# -----------------------------  # Section divider.
#  # (Blank-line placeholder comment.)
def dijkstra_all(adj: List[List[Tuple[int, int]]], source: int) -> List[float]:  # Single-source shortest paths (SSSP).
    n = len(adj)  # Number of nodes.
    dist = [INF] * n  # Distance array initialized to infinity.
    dist[source] = 0.0  # Distance to source is 0.
    pq: List[Tuple[float, int]] = [(0.0, source)]  # Heap of (distance, node) starting from the source.
    settled = [False] * n  # Track finalized nodes to avoid repeated relaxations.
    while pq:  # Continue until heap is exhausted.
        du, u = heapq.heappop(pq)  # Pop closest unsettled node.
        if settled[u]:  # Skip stale entries.
            continue  # Move on.
        settled[u] = True  # Finalize u.
        for v, w in adj[u]:  # Relax edges u->v.
            alt = du + w  # Candidate new distance to v.
            if alt < dist[v]:  # If improvement...
                dist[v] = alt  # Save improvement.
                heapq.heappush(pq, (alt, v))  # Push updated distance to heap.
    return dist  # Return all-pairs distances from source (SSSP result).
#  # (Blank-line placeholder comment.)
def choose_landmarks_even(n: int, k: int, rng: random.Random, exclude: Tuple[int, int] = (None, None)) -> List[int]:  # Pick landmark nodes.
    s, t = exclude  # Unpack nodes to exclude (typically source/target endpoints).
    candidates = [v for v in range(n) if v not in (s, t)]  # Candidate landmarks excluding s and t if provided.
    if k <= 0 or not candidates:  # If no landmarks requested or no candidates exist...
        return []  # ...return empty landmark set.
    k = min(k, len(candidates))  # Clamp k so we don't request more than available nodes.
    idxs = []  # Collect chosen indices (possibly with duplicates before de-dup).
    for i in range(1, k + 1):  # Choose k “evenly spaced” positions in the candidate list.
        pos = round(i * (len(candidates) - 1) / (k + 1))  # Compute an evenly spread position.
        idxs.append(candidates[pos])  # Add the candidate at that position.
    # de-dup and fill  # Ensure uniqueness and fill any gaps.
    out, seen = [], set()  # Output list and a set to track which nodes were already added.
    for v in idxs:  # First add the spaced picks.
        if v not in seen:  # If not yet used...
            seen.add(v)  # Mark as used.
            out.append(v)  # Append to output.
    for v in candidates:  # Then scan candidates to fill remaining slots if needed.
        if len(out) >= k:  # Stop once we have k landmarks.
            break  # Exit the fill loop.
        if v not in seen:  # If candidate not used yet...
            seen.add(v)  # Mark it.
            out.append(v)  # Add it.
    return out  # Return chosen landmarks.
#  # (Blank-line placeholder comment.)
def alt_preprocess(adj: List[List[Tuple[int, int]]], landmarks: List[int]) -> Tuple[List[List[float]], List[List[float]]]:  # Precompute landmark distances.
    radj = reverse_adjacency(adj)  # Build reversed graph for computing distances-to-landmark via Dijkstra.
    dist_from = []  # dist_from[i][v] = d(L_i, v) in original graph.
    dist_to = []  # dist_to[i][v] = d(v, L_i) in original graph (computed as d(L_i, v) in reversed graph).
    for L in landmarks:  # For each landmark node...
        dist_from.append(dijkstra_all(adj, L))  # Compute all distances from L to every node.
        dist_to.append(dijkstra_all(radj, L))  # Compute all distances to L (via reversed graph SSSP).
    return dist_from, dist_to  # Return both precomputed distance tables.
#  # (Blank-line placeholder comment.)
def alt_h_value(  # Compute ALT heuristic lower bound for distance from v to t.
    v: int,  # Current node.
    t: int,  # Target node.
    dist_from: List[List[float]],  # Landmark-to-node distances.
    dist_to: List[List[float]],  # Node-to-landmark distances.
) -> float:  # Returns an admissible lower bound h(v) <= d(v,t).
    # landmarks indexed by i; their identity isn't needed here, only the precomputed arrays  # We only need distance tables.
    best = 0.0  # Track maximum of all valid landmark-based lower bounds.
    k = len(dist_from)  # Number of landmarks.
    for i in range(k):  # For each landmark index i...
        dLv = dist_from[i][v]  # Distance from landmark L_i to v.
        dLt = dist_from[i][t]  # Distance from landmark L_i to t.
        if dLv < INF and dLt < INF:  # Only use bound if both distances are defined (reachable).
            best = max(best, dLt - dLv)  # Lower bound: d(v,t) >= d(L,t) - d(L,v).
#  # (Blank-line placeholder comment.)
        dvL = dist_to[i][v]  # Distance from v to landmark L_i.
        dtL = dist_to[i][t]  # Distance from t to landmark L_i.
        if dvL < INF and dtL < INF:  # Only use bound if both distances are defined.
            best = max(best, dvL - dtL)  # Lower bound: d(v,t) >= d(v,L) - d(t,L).
#  # (Blank-line placeholder comment.)
    return best if best > 0.0 else 0.0  # Clamp at 0 to avoid negative heuristic values.
#  # (Blank-line placeholder comment.)
def alt_query_distance(  # Run A* using ALT landmark heuristic to compute distance s->t.
    adj: List[List[Tuple[int, int]]],  # Graph adjacency list.
    s: int,  # Source node.
    t: int,  # Target node.
    dist_from: List[List[float]],  # Landmark-to-node distances (precomputed).
    dist_to: List[List[float]],  # Node-to-landmark distances (precomputed).
) -> float:  # Returns shortest distance from s to t.
    n = len(adj)  # Number of nodes.
    g = [INF] * n  # Best-known g-scores (cost-to-come).
    g[s] = 0.0  # g-score at the source is 0.
#  # (Blank-line placeholder comment.)
    def h(v: int) -> float:  # ALT heuristic function h(v) for target t.
        return 0.0 if v == t else alt_h_value(v, t, dist_from, dist_to)  # 0 at goal; otherwise landmark bound.
#  # (Blank-line placeholder comment.)
    pq: List[Tuple[float, float, int]] = [(h(s), 0.0, s)]  # Heap of (f=g+h, g, node) seeded at s.
    best_goal = INF  # Best target distance found so far.
#  # (Blank-line placeholder comment.)
    while pq:  # Process nodes by increasing f-score.
        f_u, g_u, u = heapq.heappop(pq)  # Pop the current best candidate.
        if g_u != g[u]:  # Skip stale entries (not matching current best g-score).
            continue  # Continue popping.
#  # (Blank-line placeholder comment.)
        if u == t:  # If goal is popped, record candidate solution.
            best_goal = g_u  # Update best known goal distance.
#  # (Blank-line placeholder comment.)
        if best_goal < INF:  # If we have a goal candidate...
            if not pq or pq[0][0] >= best_goal:  # ...and no remaining node can beat it in f-score...
                return best_goal  # ...then we can safely return the optimal distance.
#  # (Blank-line placeholder comment.)
        for v, w in adj[u]:  # Relax outgoing edges from u.
            alt = g_u + w  # Candidate g-score for v.
            if alt < g[v]:  # If improvement...
                g[v] = alt  # Save improved g-score.
                heapq.heappush(pq, (alt + h(v), alt, v))  # Push updated (f, g, node) entry.
#  # (Blank-line placeholder comment.)
    return best_goal  # Return best found (INF if unreachable, though model should prevent that).
#  # (Blank-line placeholder comment.)
# -----------------------------  # Section divider.
# Helpers: timing + summary  # Section header.
# -----------------------------  # Section divider.
#  # (Blank-line placeholder comment.)
def mean_sd(xs: List[float]) -> Tuple[float, float]:  # Compute mean and sample standard deviation of a list.
    if not xs:  # Handle empty input safely.
        return math.nan, math.nan  # Return NaNs if there is no data.
    mu = statistics.mean(xs)  # Compute arithmetic mean.
    sd = statistics.stdev(xs) if len(xs) >= 2 else 0.0  # Compute sample SD, or 0 for a single sample.
    return mu, sd  # Return (mean, sd).
#  # (Blank-line placeholder comment.)
def main() -> None:  # Main entry point: parse args, build graph, run queries, print summary.
    ap = argparse.ArgumentParser()  # Create CLI argument parser.
    ap.add_argument("--n", type=int, default=200, help="Number of nodes.")  # Number of nodes parameter.
    ap.add_argument("--edge-prob", type=float, default=0.03, help="Probability for extra edges i->j (i<j).")  # Density knob.
    ap.add_argument("--w-min", type=int, default=1, help="Minimum weight.")  # Minimum edge weight.
    ap.add_argument("--w-max", type=int, default=20, help="Maximum weight.")  # Maximum edge weight.
    ap.add_argument("--queries", type=int, default=2000, help="Number of shortest-path queries on the same graph.")  # Query count.
    ap.add_argument("--seed", type=int, default=0, help="Random seed.")  # RNG seed for reproducibility.
    ap.add_argument("--alt-k", type=int, default=8, help="Number of landmarks for ALT.")  # Landmark count for ALT.
    args = ap.parse_args()  # Parse CLI arguments into a namespace.
#  # (Blank-line placeholder comment.)
    rng = random.Random(args.seed)  # Create RNG instance seeded for reproducibility.
#  # (Blank-line placeholder comment.)
    adj = generate_random_connected_dag(  # Generate the benchmark graph.
        n=args.n,  # Use CLI n.
        edge_prob=args.edge_prob,  # Use CLI edge probability.
        w_min=args.w_min,  # Use CLI min weight.
        w_max=args.w_max,  # Use CLI max weight.
        rng=rng,  # Use seeded RNG.
    )  # End graph generation call.
    n = args.n  # Store n locally for convenience.
    m = edge_count(adj)  # Count edges for reporting.
#  # (Blank-line placeholder comment.)
    # Precompute A* heuristic component  # Compute min outgoing weight per node for the A* heuristic.
    out_min = compute_min_outgoing(adj)  # Precompute out_min array once (shared across all queries).
#  # (Blank-line placeholder comment.)
    # Choose landmarks and preprocess ALT once  # ALT’s big win: preprocess once, reuse across queries.
    s_excl, t_excl = 0, n - 1  # Exclude endpoints 0 and n-1 from landmark candidates (optional but common).
    landmarks = choose_landmarks_even(n, args.alt_k, rng, exclude=(s_excl, t_excl))  # Pick k landmarks.
#  # (Blank-line placeholder comment.)
    t0 = time.perf_counter()  # Start timer for ALT preprocessing.
    dist_from, dist_to = alt_preprocess(adj, landmarks)  # Compute landmark distance tables.
    t1 = time.perf_counter()  # Stop timer for ALT preprocessing.
    alt_pre_time = t1 - t0  # Compute elapsed preprocessing time in seconds.
#  # (Blank-line placeholder comment.)
    # Sample queries (s<t ensures reachability in this DAG model)  # We only sample pairs that must have a path.
    queries: List[Tuple[int, int]] = []  # Store all sampled (s, t) query pairs.
    for _ in range(args.queries):  # Repeat for the requested number of queries.
        s = rng.randrange(0, n - 1)  # Sample source in [0, n-2].
        t = rng.randrange(s + 1, n)  # Sample target in [s+1, n-1] so t>s.
        queries.append((s, t))  # Append query pair.
#  # (Blank-line placeholder comment.)
    # Time query-only execution  # Measure per-query runtime for each algorithm.
    td, ta, tl = [], [], []  # td=Dijkstra times, ta=A* times, tl=ALT query times.
    tol = 1e-9  # Numerical tolerance for distance equality checks (floats).
#  # (Blank-line placeholder comment.)
    for (s, t) in queries:  # Iterate over all queries.
        # Dijkstra  # Time Dijkstra on this query.
        start = time.perf_counter()  # Start Dijkstra timer.
        dd = dijkstra_distance(adj, s, t)  # Compute Dijkstra distance.
        td.append(time.perf_counter() - start)  # Record elapsed time.
#  # (Blank-line placeholder comment.)
        # A*  # Time A* on this query.
        start = time.perf_counter()  # Start A* timer.
        da = astar_distance_minout(adj, s, t, out_min)  # Compute A* distance using min-out heuristic.
        ta.append(time.perf_counter() - start)  # Record elapsed time.
#  # (Blank-line placeholder comment.)
        # ALT (query only)  # Time ALT query phase (preprocessing already done).
        start = time.perf_counter()  # Start ALT timer.
        dl = alt_query_distance(adj, s, t, dist_from, dist_to)  # Compute ALT-assisted A* distance.
        tl.append(time.perf_counter() - start)  # Record elapsed time.
#  # (Blank-line placeholder comment.)
        # correctness check  # Verify all algorithms agree on the shortest-path distance.
        if abs(dd - da) > tol or abs(dd - dl) > tol:  # If any algorithm disagrees beyond tolerance...
            raise RuntimeError(f"Distance mismatch for query (s={s}, t={t}): d={dd}, a*={da}, alt={dl}")  # Abort with details.
#  # (Blank-line placeholder comment.)
    mu_d, sd_d = mean_sd(td)  # Compute mean and SD for Dijkstra times.
    mu_a, sd_a = mean_sd(ta)  # Compute mean and SD for A* times.
    mu_l, sd_l = mean_sd(tl)  # Compute mean and SD for ALT times.
#  # (Blank-line placeholder comment.)
    alt_amortized = alt_pre_time / args.queries + mu_l  # Amortize preprocessing over Q queries and add mean query time.
#  # (Blank-line placeholder comment.)
    # Print paper-ready summary  # Emit an easily copy-pastable text summary.
    print("Static-graph many-queries benchmark")  # Title line for the output block.
    print("----------------------------------")  # Separator line for readability.
    print(f"n={n}, m={m}, edge_prob={args.edge_prob}, weights=[{args.w_min},{args.w_max}]")  # Graph parameters summary.
    print(f"queries={args.queries}")  # Query count.
    print(f"ALT: k={args.alt_k}, landmarks={landmarks}")  # ALT configuration summary.
    print()  # Blank line between sections.
    print(f"ALT preprocessing time: {alt_pre_time:.6g} s")  # Preprocessing time line.
    print("Query times (mean ± sd over queries):")  # Header for timing stats.
    print(f"  Dijkstra: {mu_d:.6g} ± {sd_d:.6g} s")  # Dijkstra mean and SD.
    print(f"  A*:       {mu_a:.6g} ± {sd_a:.6g} s")  # A* mean and SD.
    print(f"  ALT:      {mu_l:.6g} ± {sd_l:.6g} s")  # ALT mean and SD.
    print()  # Blank line between sections.
    print(f"ALT amortized per-query time (pre/Q + mean_query): {alt_amortized:.6g} s")  # Amortized ALT per-query cost.
#  # (Blank-line placeholder comment.)
if __name__ == "__main__":  # Standard Python entry-point guard (only run main when executed directly).
    main()  # Invoke the main function.
