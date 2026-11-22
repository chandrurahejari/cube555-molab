def prepare_jax_gpu():
    """Keep PJRT entry-point discovery inside one version-pinned worker environment."""
    version = "0.7.2"
    venv = cache / f"jax-cuda13-{version}-venv-py{sys.version_info.major}{sys.version_info.minor}"
    python = venv / "bin/python"
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.update(JAX_ENABLE_X64="True", JAX_PLATFORMS="cuda", XLA_PYTHON_CLIENT_PREALLOCATE="false",
               PYTHONNOUSERSITE="1", VIRTUAL_ENV=str(venv))
    uv = shutil.which("uv")
    if not python.exists():
        create = [uv, "venv", "--python", sys.executable, venv] if uv else [sys.executable, "-m", "venv", venv]
        command(create)
    installer = [uv, "pip", "install", "--python", python] if uv else [python, "-m", "pip", "install"]
    check = (
        "import json, sys, importlib.metadata as m; from pathlib import Path; "
        "names=('jax','jaxlib','jax-cuda13-plugin','jax-cuda13-pjrt'); "
        "versions={n:m.version(n) for n in names}; "
        f"assert set(versions.values())=={{'{version}'}}, versions; "
        "assert all(Path(m.distribution(n).locate_file('')).resolve().is_relative_to(Path(sys.prefix).resolve()) "
        "for n in names), 'JAX package outside the worker environment'; "
        "import jax, jax.numpy as j; d=jax.devices('gpu')[0]; "
        "x=j.ones((16,16),dtype=j.bfloat16); assert float((x@x).block_until_ready()[0,0])==16; "
        "print(json.dumps(dict(jax=jax.__version__, device=d.device_kind, versions=versions,"
        " platform_version=d.client.platform_version, memory_limit_bytes=(d.memory_stats() or {}).get('bytes_limit'))))")
    def probe():
        return command([python, "-c", check], capture=True, check=False)
    result = probe()
    if result.returncode:
        save("installing_jax_cuda_plugin")
        command([*installer, f"jax[cuda13-local]=={version}"])
        result = probe()
    if result.returncode:
        save("installing_jax_cuda_libraries")
        command([*installer, f"jax[cuda13]=={version}"])
        result = probe()
    if result.returncode:
        raise RuntimeError("JAX CUDA initialization failed: " + result.stdout[-3500:])
    report["jax_hardware"] = json.loads(result.stdout.strip().splitlines()[-1])
    if command([python, "-c", "import torch; assert torch.__version__.split('+')[0] == '2.8.0'"],
               capture=True, check=False).returncode:
        save("installing_checkpoint_loader")
        command([*installer, "--index-url", "https://download.pytorch.org/whl/cpu", "torch==2.8.0+cpu"])
    return python


def tpu_case(kind, pid, beam, depth):
    folder = output / f"tpu-{kind}-{pid}-{beam}"
    folder.mkdir()
    log = folder / "worker.log"
    events = folder / "events.jsonl"
    launch = time.monotonic()
    length_map = (baseline or {}).get("lengths", {})
    row = dict(kind=kind, puzzle_id=pid, requested_beam=beam, effective_beam=beam,
               requested_depth=depth, effective_depth=depth, symmetry=symmetry, symmetry_index=symmetry,
               inverse=inverse, baseline_length=length_map.get(pid, length_map.get(str(pid))),
               completed_layers=0, results=[], depth_timings=[], saturated=False, depth5_seconds=None)
    current = report["current_case"] = dict(row, phase="initializing", depth=0, frontier_size=1)
    layers = {}
    report["layer_events"] = []
    cursor = 0
    last_width = 1
    def read_progress():
        nonlocal cursor, last_width
        if not events.is_file():
            return False
        changed = False
        with events.open("rb") as stream:
            stream.seek(cursor)
            for line in stream:
                if not line.endswith(b"\n"):
                    break
                event = json.loads(line)
                cursor += len(line)
                changed = True
                if event["event"] == "probe":
                    current["phase"] = event["stage"]
                    report.setdefault("jax_probes", {})[event["stage"]] = event
                if event["event"] == "layer_audit":
                    report.setdefault("layer_audit", []).append(event)
                if event["event"] == "snapshot":
                    report.setdefault("audit_snapshots", []).append({k:event[k] for k in ("layer","data","states_sha256")})
                if event["event"] not in ("layer_start", "layer_end"):
                    continue
                index = int(event["layer"]) - 1
                stamp = lambda unix: datetime.fromtimestamp(unix, timezone.utc).isoformat(timespec="milliseconds")
                layer = layers.setdefault(index, dict(rank=0, puzzle_id=pid, depth=index, effective_beam=beam,
                                          frontier_size=last_width, started_utc=stamp(event["started_unix"])))
                current.update(phase="running", depth=index, frontier_size=last_width)
                if event["event"] == "layer_end":
                    last_width = int(event["active_count"])
                    layer.update(ended_utc=stamp(event["ended_unix"]), output_frontier_size=last_width,
                                 seconds=round(event["ended_unix"]-event["started_unix"], 3),
                                 saturated=index>=6 and layer["frontier_size"]==beam)
                    if "rank_counts" in event:
                        layer["rank_counts"] = event["rank_counts"]
                    row["completed_layers"] = index + 1
                    current.update(completed_layers=index+1, last_completed_depth=index, frontier_size=last_width)
        if changed:
            row["depth_timings"] = [v for v in layers.values() if "ended_utc" in v]
            current.update(depth_timings=row["depth_timings"], full_layer_statistics=layer_summary(row["depth_timings"]))
            report["layer_events"] = list(layers.values())
        return changed

    args = [python, repo / "tools/tpu_solve.py", "--competition", search_competition, "--assets", inputs / "model",
            "--output", folder, "--pid", pid, "--beam", beam, "--depth", depth, "--device", "gpu"]
    if (pid, symmetry, inverse, beam, depth) == (46, 0, False, 8192, 32):
        args.append("--layer-audit")
    save(f"{kind} PID {pid} / beam {beam}")
    with log.open("wb") as stdout:
        child = subprocess.Popen(list(map(str, args)), env=env, stdout=stdout, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            published = time.monotonic()
            while child.poll() is None:
                changed = read_progress()
                if changed or time.monotonic()-published >= 60:
                    current["search_seconds"] = round(time.monotonic()-launch, 3)
                    save()
                    published = time.monotonic()
                if time.monotonic()-started >= 5*3600-120:
                    raise TimeoutError("Достигнут лимит пятичасовой сессии Molab")
                time.sleep(2)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
    read_progress()
    result = read_json(folder / "result.json")
    with log.open("rb") as stream:
        stream.seek(max(0, log.stat().st_size-4500))
        report["last_log"] = stream.read().decode(errors="replace")
    row.update(wall_seconds=round(time.monotonic()-launch, 3), native_search_seconds=result.get("search_seconds"),
               full_layer_statistics=layer_summary(row["depth_timings"]), completed_layers=row["completed_layers"],
               saturated=any(v["output_frontier_size"]==beam for v in row["depth_timings"]),
               depth5_seconds=layers.get(5, {}).get("seconds"))
    if child.returncode or result.get("status") not in ("solved", "not_found"):
        row.update(status="failed", error=result.get("error") or f"JAX exited with code {child.returncode}",
                   diagnostic=report["last_log"])
    else:
        if (result["pid"], result["beam"], result["depth"], result["layers_completed"]) != (pid, beam, depth, row["completed_layers"]):
            raise ValueError("JAX worker identity or layer count mismatch")
        row.update(status="complete", results=[dict(pid=pid, solve_seconds=result["search_seconds"])])
        if result["status"] == "solved":
            path = original_path(result["path"], symmetry, inverse, frame_tables)
            tokens = path.split(".") if path else []
            if not result["verified"] or result["length"] != len(tokens) or result["prefix_length"] + result["tail_length"] != len(tokens) or not 0 <= result["tail_length"] <= 5:
                raise ValueError("Invalid JAX prefix or endgame path")
            with (inputs / "competition/test.csv").open(encoding="utf-8-sig") as file:
                state = next(r["initial_state"] for r in csv.DictReader(file) if int(r["initial_state_id"]) == pid)
            state = list(map(int, state.split(",")))
            info = read_json(inputs / "model/puzzle_info.json")
            for token in tokens:
                state = [state[i] for i in info["generators"][token]]
            if state != info["central_state"]:
                raise ValueError("JAX path failed independent replay on the original cube")
            row.update(search_outcome="solution_found", solution_orientation=f"S{symmetry}" + (" reverse" if inverse else " forward"),
                       solution=dict(puzzle_id=pid, verified=True, path=path, length=len(tokens), effective_beam=beam,
                                     found_depth=result["prefix_length"], touch_depth=result["tail_length"], symmetry_index=symmetry, inverse=inverse))
        else:
            row.update(search_outcome=result["reason"], stop_reason=f"Поиск завершил {row['completed_layers']} слоёв без пути к шару радиуса 5.")
    if kind != "capacity":
        if row["status"] == "failed":
            report["queue_failed"].append(dict(puzzle_id=pid, status="failed", error=row["error"]))
        else:
            report["queue_search_finished"].append(pid)
            if row.get("solution"):
                report["queue_completed"].append(pid)
    report["attempts"].append(row)
    report.pop("current_case", None)
    save()
    return row
