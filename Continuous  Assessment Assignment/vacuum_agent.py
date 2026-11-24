"""Assignment 1.1 - Simple Reflex Vacuum Cleaner Agent with PEAS description."""
import random
import statistics

G, R, Y, C, B, X = "\033[92m", "\033[91m", "\033[93m", "\033[96m", "\033[1m", "\033[0m"

PEAS = {
    "Performance measure": "+10 per dirty cell cleaned, -1 per move, -0.5 per suck on a clean cell",
    "Environment": "Grid of rooms (clean/dirty); dirt appears randomly with small probability",
    "Actuators": "Wheels (Left, Right, Up, Down), Suction unit (Suck), NoOp",
    "Sensors": "Dirt sensor (current cell), Location sensor (current cell), Wall/bump sensor",
}


class Environment:
    def __init__(self, rows, cols, dirt_p=0.5, spawn_p=0.02, seed=None):
        self.rng = random.Random(seed)
        self.rows, self.cols, self.spawn_p = rows, cols, spawn_p
        self.dirt = [[self.rng.random() < dirt_p for _ in range(cols)] for _ in range(rows)]
        self.pos = (0, 0)
        self.score = 0.0
        self.cleaned = 0

    def percept(self):
        r, c = self.pos
        return (self.pos, "Dirty" if self.dirt[r][c] else "Clean")

    def step(self, action):
        r, c = self.pos
        if action == "Suck":
            if self.dirt[r][c]:
                self.dirt[r][c] = False
                self.score += 10
                self.cleaned += 1
            else:
                self.score -= 0.5
        elif action in ("Left", "Right", "Up", "Down"):
            dr, dc = {"Left": (0, -1), "Right": (0, 1), "Up": (-1, 0), "Down": (1, 0)}[action]
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                self.pos = (nr, nc)
            self.score -= 1
        for i in range(self.rows):  # dirt re-appears
            for j in range(self.cols):
                if not self.dirt[i][j] and self.rng.random() < self.spawn_p:
                    self.dirt[i][j] = True

    def dirty_count(self):
        return sum(map(sum, self.dirt))


class SimpleReflexAgent:
    """Condition-action rules on the current percept only (no memory).
    Sweeps the grid in a boustrophedon (snake) path using only the location."""
    name = "Simple Reflex"

    def act(self, percept, rows, cols):
        (r, c), status = percept
        if status == "Dirty":                       # rule 1
            return "Suck"
        if rows == 1:                               # two-room world: shuttle A <-> B
            return "Right" if c < cols - 1 else "Left"
        if r % 2 == 0:                              # rule 2: even row -> go right
            if c < cols - 1: return "Right"
        else:                                       # rule 3: odd row -> go left
            if c > 0: return "Left"
        if r < rows - 1: return "Down"              # rule 4: end of row
        return "Up" if rows > 1 else "NoOp"         # rule 5: bottom, restart sweep


class RandomAgent:
    name = "Random"
    def __init__(self, seed=0): self.rng = random.Random(seed)
    def act(self, percept, rows, cols):
        return self.rng.choice(["Suck", "Left", "Right", "Up", "Down"])


def run(agent, rows, cols, steps, seed, verbose=False):
    env = Environment(rows, cols, seed=seed)
    for t in range(steps):
        p = env.percept()
        a = agent.act(p, rows, cols)
        if verbose:
            col = G if a == "Suck" else C
            print(f"  t={t:02d} percept={p[0]},{Y}{p[1]:5s}{X} -> action={col}{a}{X}")
        env.step(a)
    return env


if __name__ == "__main__":
    print(f"{B}{C}=== PEAS Description of Task Environment ==={X}")
    for k, v in PEAS.items():
        print(f"{Y}{k:20s}{X}: {v}")

    print(f"\n{B}{C}=== Trace: Two-room world (A,B), first 12 steps ==={X}")
    run(SimpleReflexAgent(), 1, 2, 12, seed=3, verbose=True)

    print(f"\n{B}{C}=== Performance comparison: 4x4 grid, 200 steps, 30 trials ==={X}")
    print(f"{'Agent':16s}{'Avg score':>10s}{'Std':>8s}{'Avg cleaned':>13s}{'Dirty left':>12s}")
    results = {}
    for make in (SimpleReflexAgent, lambda: RandomAgent(1)):
        name, scores, cl, left = None, [], [], []
        for s in range(30):
            ag = make(); name = ag.name
            e = run(ag, 4, 4, 200, seed=s)
            scores.append(e.score); cl.append(e.cleaned); left.append(e.dirty_count())
        results[name] = statistics.mean(scores)
        print(f"{name:16s}{statistics.mean(scores):10.1f}{statistics.pstdev(scores):8.1f}"
              f"{statistics.mean(cl):13.1f}{statistics.mean(left):12.1f}")
    best = max(results, key=results.get)
    print(f"\n{G}Best agent: {best}{X}  (reflex agent beats random: {R if results['Simple Reflex']<results['Random'] else G}"
          f"{results['Simple Reflex']>results['Random']}{X})")
