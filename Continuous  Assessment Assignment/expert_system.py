"""Assignment 1.2 - Knowledge-based expert system: preliminary respiratory illness advisor.
Knowledge base = facts + IF-THEN rules. Inference = forward chaining and backward chaining,
with an explanation facility.  (Educational toy; NOT medical advice.)"""
G, R, Y, C, B, X = "\033[92m", "\033[91m", "\033[93m", "\033[96m", "\033[1m", "\033[0m"

# ---- Knowledge base: (rule id, set of premises, conclusion) ----
RULES = [
    ("R1", {"fever", "cough", "body_ache"}, "flu_like"),
    ("R2", {"flu_like", "sudden_onset"}, "influenza"),
    ("R3", {"sneezing", "runny_nose", "no_fever"}, "common_cold"),
    ("R4", {"itchy_eyes", "sneezing", "seasonal"}, "allergy"),
    ("R5", {"fever", "cough", "loss_of_smell"}, "covid_suspect"),
    ("R6", {"fever", "cough", "breathlessness"}, "pneumonia_suspect"),
    ("R7", {"covid_suspect"}, "advise_test_and_isolate"),
    ("R8", {"pneumonia_suspect"}, "advise_doctor_urgently"),
    ("R9", {"influenza"}, "advise_rest_fluids_and_doctor"),
    ("R10", {"common_cold"}, "advise_rest_fluids"),
    ("R11", {"allergy"}, "advise_antihistamine_and_avoid_trigger"),
]
GOALS = ["influenza", "common_cold", "allergy", "covid_suspect", "pneumonia_suspect"]


def forward_chain(facts, rules=RULES, log=True):
    facts, trace, changed = set(facts), [], True
    while changed:
        changed = False
        for rid, prem, concl in rules:
            if concl not in facts and prem <= facts:
                facts.add(concl); trace.append((rid, prem, concl)); changed = True
                if log: print(f"  {Y}{rid}{X}: IF {sorted(prem)} THEN {G}{concl}{X}")
    return facts, trace


def backward_chain(goal, facts, depth=0, rules=RULES, asked=None, log=True):
    """Prove goal from facts; returns True/False and prints the proof tree."""
    pad = "  " * (depth + 1)
    if goal in facts:
        if log: print(f"{pad}{G}[fact]{X} {goal}")
        return True
    for rid, prem, concl in rules:
        if concl == goal:
            if log: print(f"{pad}{C}[goal]{X} {goal} via {rid} needs {sorted(prem)}")
            if all(backward_chain(p, facts, depth + 1, rules, asked, log) for p in sorted(prem)):
                return True
    if log: print(f"{pad}{R}[fail]{X} {goal}")
    return False


CASES = {
    "Case 1 (flu pattern)": {"fever", "cough", "body_ache", "sudden_onset"},
    "Case 2 (cold pattern)": {"sneezing", "runny_nose", "no_fever"},
    "Case 3 (allergy pattern)": {"itchy_eyes", "sneezing", "seasonal", "no_fever"},
    "Case 4 (covid pattern)": {"fever", "cough", "loss_of_smell"},
    "Case 5 (pneumonia pattern)": {"fever", "cough", "breathlessness"},
    "Case 6 (insufficient info)": {"cough"},
}

if __name__ == "__main__":
    print(f"{B}{C}Knowledge base: {len(RULES)} rules, goals = {GOALS}{X}")
    summary = []
    for name, symptoms in CASES.items():
        print(f"\n{B}{C}--- {name}: symptoms = {sorted(symptoms)} ---{X}")
        print(f"{B}Forward chaining:{X}")
        derived, trace = forward_chain(symptoms)
        dx = [g for g in GOALS if g in derived]
        adv = sorted(f for f in derived if f.startswith("advise"))
        print(f"  => Diagnosis: {G if dx else R}{dx or 'no conclusion'}{X}  Advice: {adv or '-'}")
        summary.append((name, dx, len(trace)))
    print(f"\n{B}{C}--- Backward chaining (goal-driven) for Case 4 ---{X}")
    for g in ("covid_suspect", "influenza"):
        print(f"{B}Prove '{g}'?{X}")
        ok = backward_chain(g, CASES["Case 4 (covid pattern)"])
        print(f"  => {G if ok else R}{ok}{X}")
    print(f"\n{B}{C}--- Summary ---{X}")
    print(f"{'Case':30s}{'Diagnosis':28s}{'Rules fired':>12s}")
    for n, d, k in summary:
        print(f"{n:30s}{str(d or 'none'):28s}{k:12d}")
