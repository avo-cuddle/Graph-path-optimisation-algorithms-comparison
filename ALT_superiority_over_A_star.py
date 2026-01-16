import math
import random
import time
from collections import deque
import heapq

# -----------------------------
# Grid with MULTIPLE "rivers" (vertical barriers) and a single bridge per river.
# Geometric A* keeps getting fooled by repeatedly approaching rivers at the wrong y.
# ALT (landmarks) encodes those bottlenecks and prunes much more.
# -----------------------------

def node_id(x, y, W):
    return y * W + x

def xy_from_id(i, W):
    return (i % W, i // W)

def build_multiriver_blocked_edges(W, H, river_xs, bridge_y, bridge_width=1):
    """Block all crossings between x=r-1 and x=r except at the bridge row(s)."""
    blocked = set()
    y0 = max(0, bridge_y - bridge_width // 2)
    y1 = min(H - 1, bridge_y + bridge_width // 2)
    bridge_rows = set(range(y0, y1 + 1))

    for r in river_xs:
        assert 1 <= r < W, "river_x must be in [1, W-1]"
        for y in range(H):
            if y in bridge_rows:
                continue
            u = node_id(r - 1, y, W)
            v = node_id(r, y, W)
            a, b = (u, v) if u < v else (v, u)
            blocked.add((a, b))
    return blocked

def neighbors(u, W, H, blocked):
    x, y = xy_from_id(u, W)
    # 4-neighborhood
    if x > 0:
        v = u - 1
        a, b = (v, u) if v < u else (u, v)
        if (a, b) not in blocked:
            yield v
    if x + 1 < W:
        v = u + 1
        a, b = (v, u) if v < u else (u, v)
        if (a, b) not in blocked:
            yield v
    if y > 0:
        v = u - W
        a, b = (v, u) if v < u else (u, v)
        if (a, b) not in blocked:
            yield v
    if y + 1 < H:
        v = u + W
        a, b = (v, u) if v < u else (u, v)
        if (a, b) not in blocked:
            yield v

def bfs_all_distances(start, W, H, blocked):
    """Unit weights => BFS is exact shortest paths."""
    n = W * H
    INF = 10**9
    dist = [INF] * n
    dist[start] = 0
    q = deque([start])
    while q:
        u = q.popleft()
        du = dist[u]
        for v in neighbors(u, W, H, blocked):
            if dist[v] == INF:
                dist[v] = du + 1
                q.append(v)
    return dist

def astar(s, t, W, H, blocked, heuristic_fn):
    n = W * H
    INF = 10**9
    g = [INF] * n
    g[s] = 0
    pq = [(heuristic_fn(s), 0, s)]  # (f, g, node)
    expanded = 0
    closed = [False] * n

    while pq:
        f, gu, u = heapq.heappop(pq)
        if closed[u]:
            continue
        closed[u] = True
        expanded += 1
        if u == t:
            return gu, expanded

        for v in neighbors(u, W, H, blocked):
            if closed[v]:
                continue
            ng = gu + 1
            if ng < g[v]:
                g[v] = ng
                heapq.heappush(pq, (ng + heuristic_fn(v), ng, v))

    return INF, expanded

def euclid_heuristic_factory(W):
    def h(u, t):
        x1, y1 = xy_from_id(u, W)
        x2, y2 = xy_from_id(t, W)
        return math.hypot(x1 - x2, y1 - y2)
    return h

def alt_heuristic_factory(dist_from_landmarks, W):
    """ALT heuristic: max over landmarks of |d(L,t) - d(L,u)|."""
    def h(u, t):
        best = 0
        for dist in dist_from_landmarks:
            du = dist[u]
            dt = dist[t]
            val = dt - du
            if val < 0:
                val = -val
            if val > best:
                best = val
        return best
    return h

def pick_landmarks(W, H, river_xs, bridge_y):
    # Corners + bridge endpoints for each river (both sides)
    corners = [
        node_id(0, 0, W),
        node_id(W - 1, 0, W),
        node_id(0, H - 1, W),
        node_id(W - 1, H - 1, W),
    ]
    bridge_nodes = []
    for r in river_xs:
        bridge_nodes.append(node_id(r - 1, bridge_y, W))
        bridge_nodes.append(node_id(r, bridge_y, W))

    # Deduplicate preserving order
    seen = set()
    landmarks = []
    for v in corners + bridge_nodes:
        if v not in seen:
            seen.add(v)
            landmarks.append(v)
    return landmarks

def generate_hard_pairs(W, H, river_xs, bridge_y, Q, seed=0, avoid_band=40):
    """Force pairs to cross ALL rivers, and keep y far from the bridge row."""
    rnd = random.Random(seed)
    left_limit = min(river_xs) - 2
    right_limit = max(river_xs) + 2

    def random_y_far():
        while True:
            y = rnd.randrange(0, H)
            if abs(y - bridge_y) >= avoid_band:
                return y

    pairs = []
    for _ in range(Q):
        y = random_y_far()
        s = node_id(rnd.randrange(0, max(1, left_limit)), y, W)
        t = node_id(rnd.randrange(min(W - 1, right_limit), W), y, W)
        pairs.append((s, t))
    return pairs

def main():
    # These are tuned so baseline A* is ~2 seconds on many machines.
    # If yours is much faster/slower, just change Q.
    W, H = 540, 220
    river_xs = [W // 4, W // 2, (3 * W) // 4]
    bridge_y = H // 2
    Q = 18

    blocked = build_multiriver_blocked_edges(W, H, river_xs, bridge_y, bridge_width=1)

    print(f"Grid: {W}x{H}  nodes={W*H:,}")
    print(f"Rivers at x={river_xs}, single bridge at y={bridge_y}")
    print(f"Blocked crossings total: {len(blocked)}")
    pairs = generate_hard_pairs(W, H, river_xs, bridge_y, Q=Q, seed=42, avoid_band=40)
    print(f"Queries: {Q} (crafted to be hard for geometric A*)")

    # ---- Baseline: geometric A* (Euclidean heuristic)
    euclid = euclid_heuristic_factory(W)
    t0 = time.perf_counter()
    exp_astar = 0
    for s, t in pairs:
        _, e = astar(s, t, W, H, blocked, heuristic_fn=lambda u, t=t: euclid(u, t))
        exp_astar += e
    t1 = time.perf_counter()
    astar_time = t1 - t0

    # ---- ALT preprocessing (distances from landmarks)
    landmarks = pick_landmarks(W, H, river_xs, bridge_y)
    p0 = time.perf_counter()
    dist_from_landmarks = [bfs_all_distances(L, W, H, blocked) for L in landmarks]
    p1 = time.perf_counter()
    pre_time = p1 - p0

    # ---- ALT-A* queries
    alt_h = alt_heuristic_factory(dist_from_landmarks, W)
    t2 = time.perf_counter()
    exp_alt = 0
    for s, t in pairs:
        _, e = astar(s, t, W, H, blocked, heuristic_fn=lambda u, t=t: alt_h(u, t))
        exp_alt += e
    t3 = time.perf_counter()
    alt_time = t3 - t2

    print("\n--- Results (total over queries) ---")
    print(f"A* (Euclidean) time:  {astar_time:6.3f} s")
    print(f"A* expansions:       {exp_astar:,}")
    print(f"ALT preprocessing:    {pre_time:6.3f} s  (landmarks={len(landmarks)})")
    print(f"ALT A* time:          {alt_time:6.3f} s")
    print(f"ALT A* expansions:   {exp_alt:,}")

    print(f"\nQuery-time speedup (A* / ALT-A*): {astar_time/alt_time:.2f}x")
    print(f"Expansion reduction:                {exp_astar/exp_alt:.2f}x")

    print("\nNotes:")
    print("- This demo is constructed so geometric A* keeps getting fooled by multiple bottlenecks.")
    print("- ALT uses landmark distances, which encode those bottlenecks, so it prunes much more.")
    print("- If your timings are too small, increase Q. If too big, decrease Q.")

if __name__ == "__main__":
    main()
