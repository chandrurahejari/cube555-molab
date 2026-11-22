def compact_axis_runs(tokens):
    """Reduce consecutive commuting parallel slice turns modulo four."""
    runs = []
    for token in tokens:
        face = token.lstrip("-")
        axis = face[0]
        if not runs or runs[-1][0] != axis:
            runs.append((axis, {}))
        turns = runs[-1][1]
        count = (turns.get(face, 0) + (-1 if token.startswith("-") else 1)) % 4
        if count:
            turns[face] = count
        else:
            turns.pop(face)
        if not turns:
            runs.pop()
    return [move for _, run in runs for face, count in sorted(run.items())
            for move in (["-" + face] if count == 3 else [face] * count)]


def best_verified_path(pid, official_directory, rules, beam):
    """Pick a replay-verified result for export after the independent search."""
    import csv
    def field(filename, name):
        with (official_directory / filename).open(encoding="utf-8-sig", newline="") as file:
            return next(row[name] for row in csv.DictReader(file) if int(row["initial_state_id"]) == pid)
    initial = [int(v) for v in field("test.csv", "initial_state").split(",")]
    sample = field("sample_submission.csv", "path")
    candidates = [("official_sample", sample)]
    if beam and beam.get("verified"):
        candidates.insert(0, ("beam", beam["path"]))
    best = None
    for source, path in candidates:
        tokens = path.split(".") if path else []
        reduced = compact_axis_runs(tokens)
        if len(reduced) < len(tokens):
            source += "_axis_reduced"
            tokens = reduced
        state = initial
        for move in tokens:
            state = [state[j] for j in rules["generators"][move]]
        if state != rules["central_state"]:
            raise ValueError(f"{source} path failed official replay for cube {pid}")
        if best is None or len(tokens) < best["length"]:
            best = dict(puzzle_id=pid, source=source, path=".".join(tokens), length=len(tokens), verified=True)
    return best
