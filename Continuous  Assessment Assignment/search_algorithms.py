"""Assignment 2 - Problem solving through search: uninformed, informed, local search and CSP.
Problems: 8-puzzle (uninformed/informed), 8-queens (local search), map colouring + 8-queens (CSP)."""
import heapq, math, random, time
from collections import deque
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

G, R, Y, C, B, X = "\033[92m", "\033[91m", "\033[93m", "\033[96m", "\033[1m", "\033[0m"
GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)
START = (8, 1, 3, 4, 0, 2, 7, 6, 5)  # solvable, optimal depth 14


def neighbours(s):
    i = s.index(0); r, c = divmod(i, 3)
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            j = nr * 3 + nc; l = list(s); l[i], l[j] = l[j], l[i]; yield tuple(l)


def manhattan(s):
    return sum(abs(i // 3 - (v - 1) // 3) + abs(i % 3 - (v - 1) % 3) for i, v in enumerate(s) if v)


def misplaced(s):
    return sum(1 for i, v in enumerate(s) if v and v != GOAL[i])


def path_of(parent, s):
    p = []
    while s is not None: p.append(s); s = parent[s]
    return p[::-1]


def bfs(start):
    q, parent, n = deque([start]), {start: None}, 0
    while q:
        s = q.popleft(); n += 1
        if s == GOAL: return path_of(parent, s), n, len(parent)
        for nb in neighbours(s):
            if nb not in parent: parent[nb] = s; q.append(nb)


def dfs_limited(start, limit):
    n = [0]; stack = [(start, [start])]
    seen_depth = {}
    while stack:
        s, p = stack.pop(); n[0] += 1
        if s == GOAL: return p, n[0]
        if len(p) - 1 < limit:
            for nb in neighbours(s):
                if seen_depth.get(nb, 99) > len(p):
                    seen_depth[nb] = len(p); stack.append((nb, p + [nb]))
    return None, n[0]


def dfs(start, limit=30):
    p, n = dfs_limited(start, limit); return p, n, limit


def ids(start):
    total = 0
    for d in range(1, 40):
        p, n = dfs_limited(start, d); total += n
        if p: return p, total, d


def ucs(start):  # unit step cost -> equivalent to BFS in cost, shows priority-queue behaviour
    pq, best, parent, n = [(0, start)], {start: 0}, {start: None}, 0
    while pq:
        g, s = heapq.heappop(pq); n += 1
        if s == GOAL: return path_of(parent, s), n, len(best)
        for nb in neighbours(s):
            if g + 1 < best.get(nb, 1e9): best[nb] = g + 1; parent[nb] = s; heapq.heappush(pq, (g + 1, nb))


def best_first(start, h, astar):
    pq, best, parent, n, cnt = [(h(start), 0, start)], {start: 0}, {start: None}, 0, 0
    closed = set()
    while pq:
        _, g, s = heapq.heappop(pq)
        if s in closed: continue
        closed.add(s); n += 1
        if s == GOAL: return path_of(parent, s), n, len(best)
        for nb in neighbours(s):
            ng = g + 1
            if ng < best.get(nb, 1e9):
                best[nb] = ng; parent[nb] = s; cnt += 1
                heapq.heappush(pq, ((ng if astar else 0) + h(nb), ng, nb))


# ---------------- Local search: 8-queens (state = row of queen in each column) ----------------
def conflicts(s):
    return sum(1 for i in range(8) for j in range(i + 1, 8) if s[i] == s[j] or abs(s[i] - s[j]) == j - i)


def hill_climb(rng, restarts=0):
    steps = 0
    for _ in range(restarts + 1):
        s = [rng.randrange(8) for _ in range(8)]
        while True:
            cur = conflicts(s); best, bs = cur, None
            for c in range(8):
                for r in range(8):
                    if r != s[c]:
                        t = s[:]; t[c] = r; h = conflicts(t)
                        if h < best: best, bs = h, t
            steps += 1
            if bs is None: break
            s = bs
        if conflicts(s) == 0: return s, steps
    return s, steps


def simulated_annealing(rng, T=2.0, alpha=0.995, max_steps=20000):
    s = [rng.randrange(8) for _ in range(8)]; steps = 0
    while steps < max_steps and conflicts(s) > 0 and T > 1e-3:
        t = s[:]; t[rng.randrange(8)] = rng.randrange(8)
        d = conflicts(t) - conflicts(s)
        if d <= 0 or rng.random() < math.exp(-d / T): s = t
        T *= alpha; steps += 1
    return s, steps


# ---------------- CSP: map colouring of Australia + 8-queens ----------------
AUS = {"WA": ["NT", "SA"], "NT": ["WA", "SA", "Q"], "SA": ["WA", "NT", "Q", "NSW", "V"],
       "Q": ["NT", "SA", "NSW"], "NSW": ["Q", "SA", "V"], "V": ["SA", "NSW"], "T": []}


def csp_solve(vars_, domains, ok, mrv=False, fc=False):
    stats = {"assign": 0}
    def rec(asg, dom):
        if len(asg) == len(vars_): return dict(asg)
        un = [v for v in vars_ if v not in asg]
        v = min(un, key=lambda x: len(dom[x])) if mrv else un[0]
        for val in list(dom[v]):
            if all(ok(v, val, u, asg[u]) for u in asg):
                stats["assign"] += 1; asg[v] = val
                nd = dom
                if fc:
                    nd = {k: [x for x in d if k in asg or ok(k, x, v, val)] if k not in asg else d
                          for k, d in dom.items()}
                    if any(not nd[k] for k in un if k != v): del asg[v]; continue
                r = rec(asg, nd)
                if r: return r
                del asg[v]
        return None
    return rec({}, domains), stats["assign"]


def queens_ok(v, a, u, b): return a != b and abs(a - b) != abs(v - u)
def map_ok(v, a, u, b): return not (u in AUS[v] and a == b)


def timed(f, *a):
    t = time.perf_counter(); r = f(*a); return r, (time.perf_counter() - t) * 1000


if __name__ == "__main__":
    print(f"{B}{C}=== PART A: 8-puzzle, start={START} ==={X}")
    print(f"{'Algorithm':26s}{'Type':11s}{'Cost':>6s}{'Expanded':>10s}{'Stored':>9s}{'ms':>9s}")
    algos = [("BFS", "Uninformed", bfs), ("DFS (limit 30)", "Uninformed", dfs), ("UCS", "Uninformed", ucs),
             ("IDS", "Uninformed", ids),
             ("Greedy (misplaced)", "Informed", lambda s: best_first(s, misplaced, False)),
             ("Greedy (manhattan)", "Informed", lambda s: best_first(s, manhattan, False)),
             ("A* (misplaced)", "Informed", lambda s: best_first(s, misplaced, True)),
             ("A* (manhattan)", "Informed", lambda s: best_first(s, manhattan, True))]
    names, exp, costs, times = [], [], [], []
    for name, typ, f in algos:
        (res, ms) = timed(f, START)
        path, n, mem = res
        cost = len(path) - 1
        col = G if typ == "Informed" else Y
        print(f"{col}{name:26s}{X}{typ:11s}{cost:6d}{n:10d}{mem if isinstance(mem,int) and mem>40 else '-':>9}{ms:9.1f}")
        names.append(name); exp.append(n); costs.append(cost); times.append(ms)

    print(f"\n{B}{C}=== PART B: Local search on 8-queens (50 runs each) ==={X}")
    print(f"{'Algorithm':32s}{'Success%':>9s}{'Avg steps':>11s}{'Avg ms':>9s}")
    ls = {}
    for name, f in [("Hill climbing (no restart)", lambda r: hill_climb(r, 0)),
                    ("Hill climbing (random restart)", lambda r: hill_climb(r, 25)),
                    ("Simulated annealing", simulated_annealing)]:
        ok, st, tm = 0, 0, 0
        for seed in range(50):
            (s, k), ms = timed(f, random.Random(seed)); ok += conflicts(s) == 0; st += k; tm += ms
        ls[name] = ok * 2
        print(f"{name:32s}{ok*2:8d}%{st/50:11.1f}{tm/50:9.2f}")

    print(f"\n{B}{C}=== PART C: Constraint satisfaction ==={X}")
    print(f"{'Problem / method':38s}{'Assignments':>12s}{'ms':>9s}{'Solved':>8s}")
    cs = []
    for pname, vars_, doms, ok in [
        ("Australia map (3 colours)", list(AUS), {v: ["R", "G", "B"] for v in AUS}, map_ok),
        ("8-Queens", list(range(8)), {v: list(range(8)) for v in range(8)}, queens_ok),
        ("12-Queens", list(range(12)), {v: list(range(12)) for v in range(12)}, queens_ok)]:
        for mname, mrv, fc in [("Backtracking", False, False), ("Backtracking+MRV", True, False),
                               ("Backtracking+MRV+FC", True, True)]:
            (sol, n), ms = timed(lambda: csp_solve(vars_, doms, ok, mrv, fc))
            print(f"{pname+' / '+mname:38s}{n:12d}{ms:9.1f}{(G+'yes'+X) if sol else (R+'no'+X):>17s}")
            cs.append((pname.split()[0] + "\n" + mname.replace("Backtracking", "BT"), n))
    if sol: print(f"{Y}Sample 12-Queens solution:{X} {[sol[i] for i in range(12)]}")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ax[0].barh(names, exp, color="#444444"); ax[0].set_xscale("log"); ax[0].set_title("8-puzzle: nodes expanded (log)")
    ax[1].barh(names, times, color="#888888"); ax[1].set_xscale("log"); ax[1].set_title("8-puzzle: time in ms (log)")
    ax[2].bar(list(ls), list(ls.values()), color="#555555"); ax[2].set_title("8-queens local search: success %")
    ax[2].set_xticks(range(len(ls))); ax[2].set_xticklabels([k.replace(" (", "\n(") for k in ls], fontsize=7)
    for a in ax: a.tick_params(labelsize=7)
    plt.tight_layout(); plt.savefig("Assignment_2/search_comparison.png", dpi=140)
    print(f"\n{G}Saved plot: Assignment_2/search_comparison.png{X}")
