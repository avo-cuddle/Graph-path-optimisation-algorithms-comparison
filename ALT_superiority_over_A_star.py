import math  # math utilities (we use sqrt via hypot)
import random  # random sampling for query pairs
import time  # timing (perf_counter)
from collections import deque  # fast FIFO queue for BFS
import heapq  # priority queue for A*'s open set (min-heap)

# -----------------------------
# Grid with MULTIPLE "rivers" (vertical barriers) and a single bridge per river.  # scenario description
# Geometric A* keeps getting fooled by repeatedly approaching rivers at the wrong y.  # why Euclidean A* struggles here
# ALT (landmarks) encodes those bottlenecks and prunes much more.  # why ALT tends to win here
# -----------------------------

def node_id(x, y, W):  # convert 2D grid coordinate (x,y) into a single integer node id
    return y * W + x  # row-major indexing: each row has W nodes

def xy_from_id(i, W):  # convert a node id back into (x,y)
    return (i % W, i // W)  # x is remainder, y is integer division

def build_multiriver_blocked_edges(W, H, river_xs, bridge_y, bridge_width=1):  # build a set of blocked undirected edges for multiple rivers
    # Block all crossings between x=r-1 and x=r except at the bridge row(s).  # explanation of what "blocked" means
    blocked = set()  # store blocked edges as (min(u,v), max(u,v)) pairs so lookups are consistent
    y0 = max(0, bridge_y - bridge_width // 2)  # start row of the bridge "gap" (clamped to grid)
    y1 = min(H - 1, bridge_y + bridge_width // 2)  # end row of the bridge "gap" (clamped to grid)
    bridge_rows = set(range(y0, y1 + 1))  # rows where crossing IS allowed (the bridge)

    for r in river_xs:  # iterate each river x-coordinate (a vertical barrier)
        assert 1 <= r < W, "river_x must be in [1, W-1]"  # ensure river is between columns so r-1 and r exist
        for y in range(H):  # scan every row for this river
            if y in bridge_rows:  # if we're at the bridge row(s)...
                continue  # ...do NOT block crossing here (this is the only passage)
            u = node_id(r - 1, y, W)  # node immediately left of the river at row y
            v = node_id(r, y, W)  # node immediately right of the river at row y
            a, b = (u, v) if u < v else (v, u)  # normalize edge ordering so (a,b) is unique for undirected lookup
            blocked.add((a, b))  # mark this crossing edge as blocked
    return blocked  # return the full set of blocked crossing edges

def neighbors(u, W, H, blocked):  # generator yielding valid 4-neighbors of u that are not blocked by rivers
    x, y = xy_from_id(u, W)  # recover (x,y) from node id u
    # 4-neighborhood  # we only allow moves left/right/up/down
    if x > 0:  # can move left if not on left boundary
        v = u - 1  # left neighbor id in row-major indexing
        a, b = (v, u) if v < u else (u, v)  # normalize the undirected edge (u,v) for blocked lookup
        if (a, b) not in blocked:  # only allow if this edge is not blocked
            yield v  # produce neighbor id
    if x + 1 < W:  # can move right if not on right boundary
        v = u + 1  # right neighbor id
        a, b = (v, u) if v < u else (u, v)  # normalize edge ordering
        if (a, b) not in blocked:  # check whether crossing is blocked
            yield v  # produce neighbor id
    if y > 0:  # can move up if not on top boundary
        v = u - W  # up neighbor id (one full row above)
        a, b = (v, u) if v < u else (u, v)  # normalize edge ordering
        if (a, b) not in blocked:  # check blocked edge
            yield v  # produce neighbor id
    if y + 1 < H:  # can move down if not on bottom boundary
        v = u + W  # down neighbor id (one full row below)
        a, b = (v, u) if v < u else (u, v)  # normalize edge ordering
        if (a, b) not in blocked:  # check blocked edge
            yield v  # produce neighbor id

def bfs_all_distances(start, W, H, blocked):  # run BFS from start to compute exact shortest-path distances on unit-weight grid
    # Unit weights => BFS is exact shortest paths.  # BFS is correct because each move costs 1
    n = W * H  # total number of nodes in the grid
    INF = 10**9  # sentinel for "unreached" distance (big number)
    dist = [INF] * n  # distance array initialized to INF for all nodes
    dist[start] = 0  # start node has distance 0 from itself
    q = deque([start])  # BFS queue seeded with the start node
    while q:  # continue until there are no nodes left to explore
        u = q.popleft()  # pop next node in FIFO order
        du = dist[u]  # current known distance to u
        for v in neighbors(u, W, H, blocked):  # iterate all reachable neighbors
            if dist[v] == INF:  # if v has not been discovered yet
                dist[v] = du + 1  # set its distance (one step more than u)
                q.append(v)  # push v into the queue for later expansion
    return dist  # return full distance table from start to every node

def astar(s, t, W, H, blocked, heuristic_fn):  # A* search from s to t using provided heuristic function
    n = W * H  # total number of nodes
    INF = 10**9  # sentinel for "no path found" (shouldn't happen in this constructed grid, but safe)
    g = [INF] * n  # g[v] = best known cost from s to v
    g[s] = 0  # starting node has cost 0
    pq = [(heuristic_fn(s), 0, s)]  # open set as (f=g+h, g, node) prioritized by smallest f
    expanded = 0  # count how many nodes we actually expand (pop + close)
    closed = [False] * n  # closed[v] indicates we've finalized v (standard A* "closed set")

    while pq:  # while there are candidate nodes to explore
        f, gu, u = heapq.heappop(pq)  # extract node u with smallest f-value
        if closed[u]:  # if we already expanded u via a better earlier pop...
            continue  # ...skip this stale heap entry
        closed[u] = True  # mark u as expanded/finalized
        expanded += 1  # increment expansion counter
        if u == t:  # if target reached
            return gu, expanded  # return shortest distance and expansion count

        for v in neighbors(u, W, H, blocked):  # explore neighbors of the expanded node u
            if closed[v]:  # if neighbor already finalized
                continue  # skip it (no need to re-open in this implementation)
            ng = gu + 1  # tentative g-value to v via u (unit edge cost)
            if ng < g[v]:  # if this is an improvement over best-known path to v
                g[v] = ng  # record improved cost-to-come
                heapq.heappush(pq, (ng + heuristic_fn(v), ng, v))  # push new candidate with updated f=g+h

    return INF, expanded  # if the queue empties, no path exists (or grid disconnected by construction)

def euclid_heuristic_factory(W):  # create a Euclidean-distance heuristic function for this grid width W
    def h(u, t):  # heuristic h(u,t) estimates cost from u to target t
        x1, y1 = xy_from_id(u, W)  # coordinates of current node u
        x2, y2 = xy_from_id(t, W)  # coordinates of target node t
        return math.hypot(x1 - x2, y1 - y2)  # Euclidean distance (straight-line) as an admissible heuristic on unit grid
    return h  # return the nested heuristic function

def alt_heuristic_factory(dist_from_landmarks, W):  # create an ALT heuristic using precomputed landmark distances
    # ALT heuristic: max over landmarks of |d(L,t) - d(L,u)|.  # triangle-inequality lower bound on d(u,t)
    def h(u, t):  # heuristic h(u,t) derived from landmark distance tables
        best = 0  # store the maximum lower bound seen so far
        for dist in dist_from_landmarks:  # iterate each landmark's distance table dist[.]
            du = dist[u]  # distance from landmark L to u (d(L,u))
            dt = dist[t]  # distance from landmark L to t (d(L,t))
            val = dt - du  # compute difference d(L,t) - d(L,u)
            if val < 0:  # take absolute value (manual abs to avoid extra function calls)
                val = -val  # now val = |d(L,t) - d(L,u)|
            if val > best:  # if this landmark yields a stronger lower bound
                best = val  # keep the maximum lower bound across landmarks
        return best  # return the ALT heuristic value
    return h  # return the nested heuristic function

def pick_landmarks(W, H, river_xs, bridge_y):  # choose landmark nodes intended to capture bottlenecks (rivers/bridge)
    # Corners + bridge endpoints for each river (both sides)  # heuristic design: corners + around bottlenecks
    corners = [  # list of corner node ids (extremes of the grid)
        node_id(0, 0, W),  # top-left corner
        node_id(W - 1, 0, W),  # top-right corner
        node_id(0, H - 1, W),  # bottom-left corner
        node_id(W - 1, H - 1, W),  # bottom-right corner
    ]
    bridge_nodes = []  # will collect nodes adjacent to each river at the bridge row
    for r in river_xs:  # for each river column position
        bridge_nodes.append(node_id(r - 1, bridge_y, W))  # node just left of the river at the bridge row
        bridge_nodes.append(node_id(r, bridge_y, W))  # node just right of the river at the bridge row

    # Deduplicate preserving order  # avoid repeated landmarks while keeping deterministic order
    seen = set()  # track which node ids have already been included
    landmarks = []  # final ordered list of landmark node ids
    for v in corners + bridge_nodes:  # traverse candidate landmarks in desired priority order
        if v not in seen:  # only add if not already present
            seen.add(v)  # mark as seen
            landmarks.append(v)  # append to landmark list
    return landmarks  # return final landmarks

def generate_hard_pairs(W, H, river_xs, bridge_y, Q, seed=0, avoid_band=40):  # craft queries that are "hard" for geometric A*
    # Force pairs to cross ALL rivers, and keep y far from the bridge row.  # design goal: mislead Euclidean heuristic
    rnd = random.Random(seed)  # deterministic RNG so results are reproducible
    left_limit = min(river_xs) - 2  # max x (exclusive-ish) for sources placed on the left side region
    right_limit = max(river_xs) + 2  # min x for targets placed on the right side region

    def random_y_far():  # pick a y-coordinate far from the bridge row (to encourage "wrong" approach)
        while True:  # repeat until we hit a y that satisfies the constraint
            y = rnd.randrange(0, H)  # sample a row uniformly
            if abs(y - bridge_y) >= avoid_band:  # enforce distance from bridge row
                return y  # accept and return y

    pairs = []  # list of (s,t) queries
    for _ in range(Q):  # generate exactly Q query pairs
        y = random_y_far()  # choose a "bad" row far from the bridge
        s = node_id(rnd.randrange(0, max(1, left_limit)), y, W)  # choose source on far-left side at that row
        t = node_id(rnd.randrange(min(W - 1, right_limit), W), y, W)  # choose target on far-right side at that row
        pairs.append((s, t))  # store the query pair
    return pairs  # return crafted queries

def main():  # orchestrate the experiment: build grid, run A*, preprocess ALT, run ALT-A*
    # These are tuned so baseline A* is ~2 seconds on many machines.  # practical runtime note
    # If yours is much faster/slower, just change Q.  # user-tunable workload knob
    W, H = 540, 220  # grid dimensions (width, height)
    river_xs = [W // 4, W // 2, (3 * W) // 4]  # place three vertical rivers at 1/4, 1/2, 3/4 of width
    bridge_y = H // 2  # bridge row is the middle row
    Q = 18  # number of hard queries to run for timing/expansion totals

    blocked = build_multiriver_blocked_edges(W, H, river_xs, bridge_y, bridge_width=1)  # construct river barriers with a single-row bridge

    print(f"Grid: {W}x{H}  nodes={W*H:,}")  # report grid size and number of nodes
    print(f"Rivers at x={river_xs}, single bridge at y={bridge_y}")  # report barrier layout
    print(f"Blocked crossings total: {len(blocked)}")  # report how many crossing edges were removed
    pairs = generate_hard_pairs(W, H, river_xs, bridge_y, Q=Q, seed=42, avoid_band=40)  # craft query pairs likely to mislead Euclidean A*
    print(f"Queries: {Q} (crafted to be hard for geometric A*)")  # report query count and intent

    # ---- Baseline: geometric A* (Euclidean heuristic)  # experiment section header
    euclid = euclid_heuristic_factory(W)  # build Euclidean heuristic function
    t0 = time.perf_counter()  # start timing baseline queries
    exp_astar = 0  # total expansions across all baseline queries
    for s, t in pairs:  # run A* once per query pair
        _, e = astar(s, t, W, H, blocked, heuristic_fn=lambda u, t=t: euclid(u, t))  # call A* with heuristic bound to this query's target
        exp_astar += e  # accumulate expansion count
    t1 = time.perf_counter()  # stop timing baseline queries
    astar_time = t1 - t0  # compute total baseline elapsed time

    # ---- ALT preprocessing (distances from landmarks)  # preprocessing stage header
    landmarks = pick_landmarks(W, H, river_xs, bridge_y)  # choose landmark nodes
    p0 = time.perf_counter()  # start timing preprocessing
    dist_from_landmarks = [bfs_all_distances(L, W, H, blocked) for L in landmarks]  # compute full BFS distance table from each landmark
    p1 = time.perf_counter()  # stop timing preprocessing
    pre_time = p1 - p0  # compute preprocessing elapsed time

    # ---- ALT-A* queries  # query stage header for ALT heuristic
    alt_h = alt_heuristic_factory(dist_from_landmarks, W)  # build ALT heuristic function using landmark distance tables
    t2 = time.perf_counter()  # start timing ALT query phase
    exp_alt = 0  # total expansions across all ALT queries
    for s, t in pairs:  # run ALT-A* once per query pair (same pairs as baseline)
        _, e = astar(s, t, W, H, blocked, heuristic_fn=lambda u, t=t: alt_h(u, t))  # call A* but swap heuristic to ALT
        exp_alt += e  # accumulate expansion count
    t3 = time.perf_counter()  # stop timing ALT query phase
    alt_time = t3 - t2  # compute ALT total query time

    print("\n--- Results (total over queries) ---")  # results section header
    print(f"A* (Euclidean) time:  {astar_time:6.3f} s")  # report baseline time
    print(f"A* expansions:       {exp_astar:,}")  # report baseline total expansions
    print(f"ALT preprocessing:    {pre_time:6.3f} s  (landmarks={len(landmarks)})")  # report ALT preprocessing time and landmark count
    print(f"ALT A* time:          {alt_time:6.3f} s")  # report ALT query time
    print(f"ALT A* expansions:   {exp_alt:,}")  # report ALT total expansions

    print(f"\nQuery-time speedup (A* / ALT-A*): {astar_time/alt_time:.2f}x")  # compute and report time speedup factor
    print(f"Expansion reduction:                {exp_astar/exp_alt:.2f}x")  # compute and report expansion reduction factor

    print("\nNotes:")  # explanatory notes header
    print("- This demo is constructed so geometric A* keeps getting fooled by multiple bottlenecks.")  # why the instance is adversarial for Euclidean A*
    print("- ALT uses landmark distances, which encode those bottlenecks, so it prunes much more.")  # why ALT helps on this instance
    print("- If your timings are too small, increase Q. If too big, decrease Q.")  # tuning advice for runtime on different machines

if __name__ == "__main__":  # standard Python entry-point guard (only run main when executed as a script)
    main()  # run the experiment
