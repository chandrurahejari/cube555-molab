def run_cube555(puzzle_id, requested_beam, requested_depth, capacity_beams, progress=None,
                queue_ids=(), baseline=None, symmetry=0, inverse=False, notebook_key=None, drive_config=None):
    """Run the original solver as a subprocess and save every measurement."""
    from pathlib import Path
    from datetime import datetime, timezone
    import base64
    import csv
    import io
    import contextlib
    import hashlib
    import json
    import os
    import platform
    import re
    import shutil
    import signal
    import subprocess
    import sys
    import tarfile
    import time
    import urllib.request
    import zipfile
    import uuid

    __SYMMETRY_HELPERS__
    __DRIVE_HELPERS__
    __CHECKPOINT_HELPERS__


    commit = __SOURCE_COMMIT__
    source_hash = __SOURCE_SHA256__
    source_blob = __SOURCE_BLOB__
    source_manifest = __SOURCE_MANIFEST__
    input_data_hash = __INPUT_DATA_SHA256__
    input_blob = __INPUT_DATA_BLOB__
    weights_url = "https://api.kaggle.com/v1/datasets/download/artgor/cube555-tpu-artifacts/"
    archive_sha = "49d61a127540cd197db9bf75aacad5712a7ab54d3fbcb39bf2da9ae9bb477286"
    cache = Path(os.environ.get("CUBE555_WORKDIR", Path.cwd() / ".cube555")).resolve()
    runid = uuid.uuid4().hex
    output = cache / "runs" / runid
    output.mkdir(parents=True)
    env = os.environ.copy()
    started = time.monotonic()
    report = dict(status="starting", run_id=runid, backend="production_runner_libtorch_stream1",
                  started_utc=datetime.now(timezone.utc).isoformat(), weights_dataset="artgor/cube555-tpu-artifacts/versions/3",
                  original_inputs_sha256=archive_sha, inputs_data_sha256=input_data_hash,
                  solver_repository="chandrurahejari/cube555-molab", solver_commit=commit, solver_source_sha256=source_hash,
                  upstream_solver_commit="a152738740044b683e1fe6e759ccb5b4b095d188", original_notebook_sha256=
                  "ae84aa8dea9cc5c245e920986ec551bccf9d370122a39da847ed03bbf2d8f6e7",
                  configuration=dict(puzzle_id=puzzle_id, requested_beam=requested_beam,
                  requested_depth=requested_depth, b_micro=8192, model_micro=128, lanes=1,
                  touch_radius=5, blend=[0.8, 0.2], score_precision="tpu", solution_mode="collect",
                  collect_extra_depths=1, max_collected_solutions=2000, publication=False),
                  attempts=[], max_verified_saturated_beam=None, solution=None, gpus=[],
                  queue_ids=list(queue_ids), queue_completed=[], queue_search_finished=[], queue_failed=[],
                  baseline={k: v for k, v in (baseline or {}).items() if k != "lengths"})
    report["configuration"].update(symmetry_index=symmetry, inverse=inverse)
    report['notebook_key'] = notebook_key
    report["runtime_identity"] = dict(hostname=platform.node(), process_id=os.getpid())
    reportfile = output / "cube555_gpu_result.json"
    job_lock = None
    sink = DriveSink(drive_config, output)

    def resource_usage():
        snapshot = {"sampled_utc": datetime.now(timezone.utc).isoformat()}
        for directory in (cache, Path("/tmp")):
            try:
                usage = shutil.disk_usage(directory)
                snapshot[str(directory)] = {"disk_free_bytes": usage.free, "disk_total_bytes": usage.total}
            except OSError as exc:
                snapshot[str(directory)] = {"sample_error": str(exc)}
        for root in (Path("/sys/fs/cgroup"), Path("/sys/fs/cgroup/memory")):
            current = root / ("memory.current" if root.name == "cgroup" else "memory.usage_in_bytes")
            if not current.is_file():
                continue
            files = ("memory.current", "memory.peak", "memory.max", "memory.events", "memory.stat") if root.name == "cgroup" else (
                "memory.usage_in_bytes", "memory.max_usage_in_bytes", "memory.limit_in_bytes", "memory.failcnt", "memory.stat")
            memory = {}
            for filename in files:
                file = root / filename
                if not file.is_file():
                    continue
                try:
                    value = file.read_text().strip()
                except OSError as exc:
                    memory[filename] = {"sample_error": str(exc)}
                    continue
                if "\n" in value:
                    memory[filename] = {row.split()[0]: int(row.split()[1]) for row in value.splitlines()
                                        if len(row.split()) == 2 and row.split()[1].isdigit()}
                else:
                    memory[filename] = int(value) if value.isdigit() else value
            snapshot["cgroup"] = memory
            break
        return snapshot

    def save(status=None, publish=True):
        if status:
            if report.get('status') != status:
                report.setdefault('stage_events', []).append(dict(stage=status, utc=datetime.now(timezone.utc).isoformat()))
            report["status"] = status
            print("Cube555", status, flush=True)
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report["resources"] = resource_usage()
        report['drive_sync'] = sink.snapshot()
        temporary = reportfile.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(reportfile)
        if publish:
            sink.enqueue(report)
        if progress:
            progress(report, reportfile)

    def command(args, cwd=None, capture=False, check=True):
        if capture:
            return subprocess.run(list(map(str, args)), cwd=cwd, env=env, check=check,
                                  text=True, errors="replace", stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT)
        lines = []
        last_update = time.monotonic()
        with subprocess.Popen(list(map(str, args)), cwd=cwd, env=env,
                              text=True, errors="replace", stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT) as child:
            for line in child.stdout:
                lines.append(line)
                if len(lines) > 100:
                    lines.pop(0)
                if time.monotonic() - last_update > 5:
                    report["last_log"] = "".join(lines)[-2000:]
                    save()
                    last_update = time.monotonic()
            rc = child.wait()
        tail = "".join(lines)
        report["last_log"] = tail[-2000:]
        save()
        if check and rc:
            raise subprocess.CalledProcessError(rc, args, output=tail)
        return subprocess.CompletedProcess(args, rc, tail)

    def digest(path):
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()

    def download(url, path, sha):
        if path.exists() and digest(path) == sha:
            return
        partial = path.with_name(path.name + "." + runid + ".partial")
        try:
            with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as dest:
                shutil.copyfileobj(response, dest)
            if digest(partial) != sha:
                raise ValueError("Download SHA-256 failed: " + path.name)
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)

    def prepare_env():
        # Molab GPU images already include torch and a recent CUDA compiler.
        # Reuse that environment when it provides CUDA support for Blackwell.
        try:
            import torch
            use_current = bool(torch.cuda.is_available() and shutil.which("nvcc"))
        except ImportError:
            use_current = False
        if use_current:
            python = Path(sys.executable)
            if not shutil.which("cmake") or not shutil.which("ninja"):
                uv = shutil.which("uv")
                installer = [uv, "pip", "install", "--python", python] if uv else [python, "-m", "pip", "install"]
                command([*installer, "cmake>=3.28,<4", "ninja>=1.11"])
            env["PATH"] = str(python.parent) + os.pathsep + env["PATH"]
            cuda_root = Path(shutil.which("nvcc")).resolve().parent.parent
            provided_root = cuda_root
            nvcc_version = command([cuda_root / "bin/nvcc", "--version"], capture=True).stdout
            if torch.version.cuda == "13.0" and "release 13.0," not in nvcc_version:
                # Molab supplies CUDA 13.0 headers to torch but a newer nvcc.
                # Install the matching compiler, compiler IR, headers and ptxas together.
                prefix = cache / "toolchain" / "nvcc-13.0.88"
                marker = prefix / ".cuda-13.0-complete"
                cuda_root = prefix / "nvidia/cu13"
                if not marker.is_file():
                    uv = shutil.which("uv")
                    installer = [uv, "pip", "install", "--python", python] if uv else [python, "-m", "pip", "install"]
                    command([*installer, "--target", prefix, "--no-deps", "nvidia-cuda-nvcc==13.0.88",
                             "nvidia-cuda-crt==13.0.88", "nvidia-nvvm==13.0.88", "nvidia-cuda-runtime==13.0.88",
                             "nvidia-cuda-cccl==13.0.85"])
                    # Share the preinstalled math libraries, platform stubs and their headers.
                    for section in ("lib", "lib64", "include"):
                        for original in (provided_root / section).rglob("*"):
                            local = cuda_root / original.relative_to(provided_root)
                            if original.is_file() and not local.exists():
                                local.parent.mkdir(parents=True, exist_ok=True)
                                local.symlink_to(original.resolve())
                    result = command([cuda_root / "bin/nvcc", "--version"], capture=True)
                    if "release 13.0," not in result.stdout:
                        raise RuntimeError("Matching CUDA 13.0 compiler did not install")
                    marker.touch()
                env["PATH"] = str(cuda_root / "bin") + os.pathsep + env["PATH"]
            env["CUDA_HOME"] = env["CUDAToolkit_ROOT"] = env["CUDA_TOOLKIT_ROOT_DIR"] = str(cuda_root)
            env["CUDACXX"] = str(cuda_root / "bin/nvcc")
            env["CC"] = shutil.which("gcc") or "gcc"
            env["CXX"] = env["CUDAHOSTCXX"] = shutil.which("g++") or "g++"
            nvidia = Path(torch.__file__).resolve().parents[1] / "nvidia"
            nccl_headers = list(nvidia.glob("*/include/nccl.h"))
            if not nccl_headers:
                raise RuntimeError("The Molab torch environment has no NCCL development header")
            env["CUBE555_NCCL_ROOT"] = str(nccl_headers[0].parent.parent)
            libs = [cuda_root / "lib", cuda_root / "lib64", Path(torch.__file__).resolve().parent / "lib",
                    *nvidia.glob("*/lib"), *nvidia.glob("*/lib64"), Path("/usr/lib/x86_64-linux-gnu")]
            libs.extend(cuda_root.glob("targets/*/lib"))
            # FindCUDA requires linker names such as libcudart.so. NVIDIA's
            # wheel layout can supply only the versioned runtime. Expose a
            # writable directory of links to those exact installed libraries.
            link_dir = cache / "toolchain" / "cuda-library-links"
            link_dir.mkdir(parents=True, exist_ok=True)
            available = {}
            for library_dir in libs:
                for library in sorted(library_dir.glob("lib*.so*"), key=lambda p: len(p.name)):
                    name = library.name.split(".so", 1)[0] + ".so"
                    if library.is_file() and name not in available:
                        available[name] = library.resolve()
            for name, target in available.items():
                link = link_dir / name
                if link.is_symlink() and link.resolve() != target:
                    link.unlink()
                if not link.exists():
                    link.symlink_to(target)
            if "libcudart.so" not in available:
                raise RuntimeError("NVIDIA CUDA runtime library libcudart is missing")
            cmake = {
                "CUDA_CUDART_LIBRARY": str(available["libcudart.so"]),
                "CUDA_TOOLKIT_ROOT_DIR": str(cuda_root),
                "CUDAToolkit_ROOT": str(cuda_root),
                "CMAKE_LIBRARY_PATH": ";".join(map(str, [link_dir, *libs])),
            }
            env["CUBE555_CUDA_CMAKE_ARGS"] = json.dumps(cmake)
            env["LD_LIBRARY_PATH"] = os.pathsep.join(map(str, [link_dir, *libs])) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
            env["LIBRARY_PATH"] = os.pathsep.join(map(str, [link_dir, *libs])) + os.pathsep + env.get("LIBRARY_PATH", "")
            report["environment"] = dict(python=str(python), torch=torch.__version__,
                                         cuda=torch.version.cuda, cuda_root=str(cuda_root),
                                         nccl_root=env["CUBE555_NCCL_ROOT"], source="molab GPU image",
                                         cuda_cmake=cmake)
            return python
        root = cache / "toolchain"
        prefix = root / "cuda128"
        python = prefix / "bin/python"
        ready = prefix / ".cube555_environment_v2"
        root.mkdir(exist_ok=True)
        if not ready.is_file():
            # Use an unprivileged Linux toolchain even if the notebook Python has
            # no CUDA build of torch or the system image omits the CUDA compiler.
            if platform.machine() not in ("x86_64", "amd64"):
                raise RuntimeError("The bundled CUDA 12.8 bootstrap supports Linux x86_64")
            mamba = root / "micromamba"
            if not mamba.is_file():
                metadata_url = "https://api.anaconda.org/release/conda-forge/micromamba/2.3.2"
                with urllib.request.urlopen(metadata_url, timeout=60) as response:
                    packages = json.load(response)["distributions"]
                package = next(p for p in packages if p["basename"].startswith("linux-64/")
                               and p["basename"].endswith(".tar.bz2") and p.get("sha256"))
                archive = root / "micromamba.tar.bz2"
                download("https://conda.anaconda.org/conda-forge/" + package["basename"], archive, package["sha256"])
                with tarfile.open(archive) as tar:
                    with tar.extractfile("bin/micromamba") as src, mamba.open("wb") as dest:
                        shutil.copyfileobj(src, dest)
                mamba.chmod(0o755)
            env["MAMBA_ROOT_PREFIX"] = str(root / "mamba")
            common = [mamba, "install" if python.is_file() else "create", "-y", "-p", prefix,
                      "--override-channels", "-c", "nvidia/label/cuda-12.8.1", "-c", "conda-forge",
                      "python=3.11", "pip", "cmake>=3.26,<4", "ninja", "gxx_linux-64=13",
                      "cuda-nvcc=12.8", "cuda-cudart-dev=12.8", "cuda-libraries-dev=12.8",
                      "cuda-cupti-dev=12.8", "cuda-nvtx=12.8"]
            command(common)
            command([python, "-m", "pip", "install", "--disable-pip-version-check", "numpy<3", "pandas<3"])
            command([python, "-m", "pip", "install", "--disable-pip-version-check", "torch==2.10.0",
                     "--index-url", "https://download.pytorch.org/whl/cu128"])
            ready.write_text("torch 2.10.0 cu128\n", encoding="utf-8")
        env["PATH"] = str(prefix / "bin") + os.pathsep + env.get("PATH", "")
        env["CUDACXX"] = str(prefix / "bin/nvcc")
        env["CUDA_HOME"] = env["CUDAToolkit_ROOT"] = str(prefix)
        for key, pattern in [("CC", "*-linux-gnu-cc"), ("CXX", "*-linux-gnu-c++")]:
            env[key] = str(next((prefix / "bin").glob(pattern)))
        env["CUDAHOSTCXX"] = env["CXX"]
        # The NVIDIA wheels package their headers and shared libraries beside
        # torch. All subsequent native children inherit this exact search path.
        torch_lib = prefix / "lib/python3.11/site-packages"
        libdirs = [torch_lib / "torch/lib", *(torch_lib / "nvidia").glob("*/lib"),
                   prefix / "lib", prefix / "targets/x86_64-linux/lib"]
        env["LD_LIBRARY_PATH"] = os.pathsep.join(map(str, libdirs)) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = "2"
        return python

    def unpack_bundle(blob, expected_sha, directory):
        raw = base64.b64decode(blob, validate=True)
        if hashlib.sha256(raw).hexdigest() != expected_sha:
            raise ValueError("Embedded archive checksum mismatch: " + directory.name)
        directory.mkdir(parents=True, exist_ok=False)
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:xz") as archive:
            for entry in archive:
                target = (directory / entry.name).resolve()
                if not entry.isfile() or not target.is_relative_to(directory) or target == directory:
                    raise ValueError("Invalid embedded member: " + entry.name)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(entry) as src, target.open("xb") as dest:
                    shutil.copyfileobj(src, dest)

    def prepare_repo():
        repo = output / "source"
        if source_blob:
            unpack_bundle(source_blob, source_hash, repo)
        else:
            if hashlib.sha256(json.dumps(source_manifest, sort_keys=True).encode()).hexdigest() != source_hash:
                raise ValueError("Native source manifest checksum mismatch")
            url = "https://codeload.github.com/chandrurahejari/cube555-molab/zip/" + commit
            prefix = "cube555-molab-" + commit + "/"
            with urllib.request.urlopen(url, timeout=120) as response:
                with zipfile.ZipFile(io.BytesIO(response.read())) as archive:
                    for name, sha in source_manifest.items():
                        data = archive.read(prefix + name)
                        if hashlib.sha256(data).hexdigest() != sha:
                            raise ValueError("Native source checksum mismatch: " + name)
                        target = repo / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(data)
            revision = json.loads((repo / "UPSTREAM.json").read_text())["cutlass_commit"]
            cutlass = repo / "external/cutlass"
            with zipfile.ZipFile(repo / ("vendor/cutlass-" + revision[:7] + ".zip")) as headers:
                for name in headers.namelist():
                    if name.endswith("/"):
                        continue
                    target = (cutlass / name).resolve()
                    if not target.is_relative_to(cutlass):
                        raise ValueError("Invalid vendored CUTLASS member")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(headers.read(name))
            (cutlass / "CUBE555_CUTLASS_REVISION").write_text(revision)
        env["CAYLEYPY_CUTLASS_DIR"] = str(repo / "external/cutlass")
        report["source_provenance"] = json.loads((repo / "UPSTREAM.json").read_text(encoding="utf-8"))
        return repo

    def prepare_inputs():
        root = output / "inputs"
        unpack_bundle(input_blob, input_data_hash, root)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        original = cache / ("inputs-" + archive_sha + ".zip")
        original_url = ("https://github.com/TryDotAtwo/MultiGPUBeamSearch/releases/download/"
                        "cube555-inputs-20261001/cube555-inputs.zip")
        for name, entry in manifest.items():
            path = (root / name).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Invalid manifest path: " + name)
            if name.endswith(".pt"):
                weight = cache / "weights" / entry["sha256"]
                weight.parent.mkdir(parents=True, exist_ok=True)
                try:
                    download(weights_url + path.name + "?datasetVersionNumber=3", weight, entry["sha256"])
                except (OSError, ValueError) as exc:
                    # Some artifact responses are truncated. Only extract the
                    # named tensor from the checksum-pinned original dataset.
                    report.setdefault("weight_fallbacks", {})[name] = str(exc)
                    download(original_url, original, archive_sha)
                    tmp = weight.with_name(weight.name + "." + runid + ".partial")
                    try:
                        with zipfile.ZipFile(original) as archive, archive.open(name) as src, tmp.open("wb") as dst:
                            shutil.copyfileobj(src, dst)
                        if tmp.stat().st_size != entry["bytes"] or digest(tmp) != entry["sha256"]:
                            raise ValueError("Original model tensor checksum mismatch: " + name)
                        tmp.replace(weight)
                    finally:
                        tmp.unlink(missing_ok=True)
                path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(weight, path)
                except OSError:
                    shutil.copyfile(weight, path)
            if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
                raise ValueError("Input checksum mismatch: " + name)
        report["inputs"] = manifest
        return root

    def read_json(path):
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    def merge_layer_samples(samples, rank, text):
        records = samples.setdefault(rank, {})
        for match in re.finditer(r"depth_start=(\d+) frontier_size=(\d+)", text):
            records.setdefault(int(match[1]), {})["frontier_size"] = int(match[2])
        for match in re.finditer(r"depth_done=(\d+) depth_sec=([\d.eE+\-]+)", text):
            records.setdefault(int(match[1]), {})["seconds"] = float(match[2])
        for line in text.splitlines():
            if line.startswith("whole_layer_depth="):
                values = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", line)}
                if "q_cutoff_key" in values:
                    records.setdefault(values.pop("whole_layer_depth"), {})["selection"] = values
            event = re.search(r'depth_(start|done)=(\d+)', line)
            stamp = re.search(r'(started|ended)_unix_ms=(\d+)', line)
            if event and stamp:
                records.setdefault(int(event[2]), {})[stamp[1] + '_utc'] = datetime.fromtimestamp(
                    int(stamp[2]) / 1000, timezone.utc).isoformat(timespec='milliseconds')

    def completed_layers(samples, expected_ranks, effective_beam):
        if len(samples) != expected_ranks:
            return []
        common = set.intersection(*(set(records) for records in samples.values()))
        layers = []
        for depth in sorted(common):
            records = [rank[depth] for rank in samples.values()]
            if not all("seconds" in row and "frontier_size" in row for row in records):
                continue
            frontier = sum(row["frontier_size"] for row in records)
            layers.append(dict(depth=depth, seconds=max(row["seconds"] for row in records),
                               frontier_size=frontier, effective_beam=effective_beam,
                               started_utc=min((r['started_utc'] for r in records if r.get('started_utc')), default=None),
                               ended_utc=max((r['ended_utc'] for r in records if r.get('ended_utc')), default=None),
                               saturated=bool(depth >= 7 and effective_beam and frontier == effective_beam)))
            if len(records) == 1 and "selection" in records[0]:
                layers[-1]["selection"] = records[0]["selection"]
        return layers

    def layer_summary(layers):
        full = [row["seconds"] for row in layers if row["saturated"]]
        return dict(minimum_depth=7, full_layer_count=len(full),
                    mean_seconds=sum(full) / len(full) if full else None,
                    min_seconds=min(full) if full else None,
                    max_seconds=max(full) if full else None)

    def completed_count(samples, ranks):
        # Native indices start at zero. A layer is complete when every rank has
        # emitted depth_done for it. Opening a log during compilation is not a layer.
        if not any(samples.values()):
            return None
        return min((max((index + 1 for index, row in samples.get(rank, {}).items()
                         if 'seconds' in row), default=0) for rank in range(ranks)), default=None)

    def case(kind, pid, beam, depth):
        def stop_solver_tree(proc):
            # torch elastic may give each rank its own session, outside the launcher group.
            # Retain (PID, start time) pairs so an unrelated reused PID is never targeted.
            def identity(statfile):
                fields = statfile.read_text().rsplit(")", 1)[1].split()
                return int(fields[1]), fields[19]  # PPID and /proc starttime
            table = {}
            for entry in Path("/proc").glob("[0-9]*/stat"):
                try:
                    table[int(entry.parent.name)] = identity(entry)
                except (FileNotFoundError, PermissionError, ProcessLookupError):
                    continue
            owned = {proc.pid} if proc.pid in table else set()
            while True:
                children = {child for child, (parent, _) in table.items() if parent in owned}
                if children <= owned:
                    break
                owned.update(children)
            for sig in (signal.SIGTERM, signal.SIGKILL):
                for child in owned:
                    try:
                        if identity(Path(f"/proc/{child}/stat"))[1] == table[child][1]:
                            os.kill(child, sig)
                    except (FileNotFoundError, ProcessLookupError):
                        continue
                # Include tasks newly forked inside the original isolated process group.
                try:
                    os.killpg(proc.pid, sig)
                except ProcessLookupError:
                    pass
                if sig == signal.SIGTERM:
                    time.sleep(3)
        folder = output / f"{kind}-{pid}-{beam}"
        log = output / (folder.name + ".log")
        cuda_diagnostics = (drive_config or {}).get('cuda_diagnostics', env.get('CUBE555_CUDA_DIAGNOSTICS', 'off'))
        if cuda_diagnostics not in ('off', 'sync', 'memcheck'):
            raise ValueError('cuda_diagnostics must be off, sync, or memcheck')
        fingerprint = dict(notebook_key=notebook_key, commit=commit, input_sha256=archive_sha, source_sha256=source_hash,
                           torch=hardware, kind=kind, pid=pid, beam=beam, depth=depth,
                           config=[0.8, 5, 8192, 128, 1, symmetry, inverse, "collect", 1, 2000],
                           symmetry_sha256=report.get("symmetry_data_sha256"),
                           baseline_sha256=report["baseline"].get("sha256"))
        key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
        checkpoint_dir = cache / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        checkpoint = checkpoint_dir / (key + ".json")
        prior = read_json(checkpoint)
        if cuda_diagnostics == 'off' and prior.get("status") == "complete" and prior.get("fingerprint") == fingerprint:
            row = {**prior["row"], "restored_from_run": prior["run_id"]}
            report["attempts"].append(row)
            if kind != "capacity":
                report["queue_search_finished"].append(pid)
                if row.get("solution"):
                    report["queue_completed"].append(pid)
            save(f"Restored {kind} PID {pid} / beam {beam}")
            return row
        args = [python, "-u", "-m", "tools.cube555.run", "--assets", inputs / "model",
                "--competition", search_competition, "--output", folder, "--pids", pid,
                "--beam", beam, "--depth", depth, "--touch-radius", 5, "--transformer-weight", 0.8,
                "--score-precision", "tpu",
                "--b-micro", 8192, "--model-micro", 128, "--inference-concurrency", 1,
                "--reflect-mode", "off", "--solution-mode", "collect", "--collect-extra-depths", 1,
                "--max-collected-solutions", 2000]
        save(f"{kind} PID {pid} / beam {beam}")
        tick = time.monotonic()
        stage_started = time.monotonic()
        previous_stage = "starting"
        stage_seconds = {}
        layer_samples = {}
        last_published = 0
        last_drive_publish = 0
        last_signature = None
        timed_out = False
        trace_rows = {}
        def capture_trace(text):
            for line in text.splitlines():
                match = re.match(r"^(track_solution_(?:host_check|generated|prefinal|prefix|first_missing))(?:=1)?\s+(.*)$", line)
                if not match:
                    continue
                event = {k:(int(v) if v.lstrip("-").isdigit() else v) for k,v in re.findall(r"(\w+)=([^\s]+)", match[2])}
                if event.get("puzzle_id") == pid:
                    trace_rows.setdefault(str(event.get("prefix_len", 0)), {})[match[1].removeprefix("track_solution_")] = event
        with log.open("w", encoding="utf-8") as stream, contextlib.ExitStack() as resources:
            publisher = None
            case_env = env.copy()
            # The Kaggle TPU reference uses 2**22 states. Keep that beam in
            # the single-device search on the Molab 96 GiB GPU as well.
            full_layer = (len(hardware["devices"]) == 1 and beam <= 4_194_304
                          and hardware["devices"][0]["bytes"] >= 80 * 1024**3)
            case_env["CUBE555_SINGLE_GPU_FULL_LAYER"] = str(int(full_layer))
            case_env["CUBE555_TPU_ROUTING"] = str(int(full_layer))
            report["search_dispatch"] = "single_gpu_whole_layer" if full_layer else "sharded"
            report["tpu_logical_ranks"] = 8 if full_layer else 0
            sanitizer_log = output / (folder.name + '-memcheck.log')
            if cuda_diagnostics != 'off':
                # The last failing CUDA API may simply be waiting on unrelated work.
                # Measure normal timings only with diagnostics disabled.
                cmake_args = json.loads(case_env.get('CUBE555_CUDA_CMAKE_ARGS', '{}'))
                cmake_args['BEAM_DEBUG_FINAL_VALIDATE'] = 'ON'
                cmake_args['CMAKE_CUDA_FLAGS'] = (cmake_args.get('CMAKE_CUDA_FLAGS', '') + ' -lineinfo').strip()
                case_env['CUBE555_CUDA_CMAKE_ARGS'] = json.dumps(cmake_args)
                sanitizer = shutil.which('compute-sanitizer', path=case_env.get('PATH'))
                if cuda_diagnostics == 'memcheck' and sanitizer:
                    # Preserve async scheduling for race reproduction and instrument only the native rank.
                    args = [sanitizer, '--tool', 'memcheck', '--target-processes', 'all',
                            '--target-processes-filter', 'production_runner_libtorch_stream1',
                            '--launch-timeout', '0', '--print-limit', '3', '--error-exitcode', '86',
                            '--log-file', sanitizer_log, *args]
                    actual_mode = 'memcheck'
                else:
                    case_env['CUBE555_CUDA_SYNC_STAGES'] = '1'
                    case_env['CUDA_LAUNCH_BLOCKING'] = '1'
                    case_env['TORCH_SHOW_CPP_STACKTRACES'] = '1'
                    actual_mode = 'sync'
                report['cuda_diagnostics'] = dict(requested=cuda_diagnostics, actual=actual_mode,
                    fallback_reason='compute-sanitizer unavailable' if actual_mode != cuda_diagnostics else None)
                save()
            case_env.pop("CUBE555_TRACE_PATH", None)
            trace_source = (drive_config or {}).get("trace_reference")
            if trace_source and kind == "solve" and pid == 68 and not full_layer:
                case_env["CUBE555_TRACE_PATH"] = frame_path(trace_source, symmetry, inverse, frame_tables)
                report["reference_trace"] = {"puzzle_id":pid,"reference_length":len(trace_source.split(".")),
                                             "purpose":"observe_prefixes_only","prefixes":trace_rows}
            for setting in ("CUBE555_SNAPSHOT_DIR", "CUBE555_RESUME_DIR", "CUBE555_SNAPSHOT_IDENTITY", "CUBE555_SNAPSHOT_SECONDS"):
                case_env.pop(setting, None)
            snapshot_options = (drive_config or {}).get("checkpoint") or {}
            if kind == "solve" and snapshot_options.get("enabled"):
                if ranks != 1:
                    raise ValueError("Native snapshots currently require a single GPU rank")
                # Exclude notebook key and host-specific cache paths. The native reader also
                # validates the effective runner depth, beam, ABI, and frontier/history counts.
                compatibility = {k:v for k,v in fingerprint.items() if k not in ("notebook_key", "baseline_sha256")}
                identity_sha = hashlib.sha256(json.dumps(compatibility, sort_keys=True).encode()).hexdigest()
                snap_root = output / "native-snapshots"
                snap_root.mkdir(exist_ok=True)
                transport = DriveCheckpoint(drive_config)
                report["checkpoint"] = dict(enabled=True, state="waiting_for_layer", interval_seconds=1800)
                case_env.update(CUBE555_SNAPSHOT_DIR=str(snap_root), CUBE555_SNAPSHOT_IDENTITY=identity_sha,
                                CUBE555_SNAPSHOT_SECONDS="1800")
                if snapshot_options.get("resume_from"):
                    report["checkpoint"]["state"] = "restoring_from_drive"
                    save()
                    restored = snap_root / "restored"
                    transport.restore(restored, identity_sha)
                    case_env["CUBE555_RESUME_DIR"] = str(restored)
                    report["checkpoint"]["state"] = "download_verified"
                publisher = SnapshotPublisher(transport, snap_root, identity_sha, report["checkpoint"])
                resources.callback(publisher.close)
            with subprocess.Popen(list(map(str, args)), cwd=repo, env=case_env, stdout=stream, start_new_session=True,
                                  stderr=subprocess.STDOUT) as proc:
                while proc.poll() is None:
                    if time.monotonic() - started >= 4*3600 + 45*60:
                        timed_out = True
                        stop_solver_tree(proc)
                        break
                    time.sleep(0.25 if previous_stage == 'running' else 1)
                    if publisher:
                        publisher.poll()
                    try:
                        current = read_json(folder / "run_summary.json")
                    except (ValueError, OSError):
                        current = {}
                    phase = current.get("status", previous_stage)
                    if phase != previous_stage:
                        stage_seconds[previous_stage] = round(time.monotonic() - stage_started, 3)
                        previous_stage, stage_started = phase, time.monotonic()
                        report.setdefault('stage_events', []).append(dict(stage=phase, utc=datetime.now(timezone.utc).isoformat(), puzzle_id=pid))
                    tails, rank_progress = [], []
                    # torchrun writes these streams during search. The readable
                    # rank-N.log copies are only materialized after it exits.
                    rank_streams = {}
                    for path in sorted(folder.rglob("rank-*.log")):
                        match = re.fullmatch(r"rank-(\d+)\.log", path.name)
                        if match:
                            rank_streams[int(match[1])] = path
                    for path in sorted(folder.rglob("stdout.log")):
                        if path.parent.name.isdecimal():
                            rank_streams[int(path.parent.name)] = path
                    rank_by_path = {path: rank for rank, path in rank_streams.items()}
                    for path in [log, *sorted(folder.rglob("cmake-*.log")),
                                 *rank_by_path, *sorted(folder.rglob("stderr.log"))]:
                        try:
                            with path.open("rb") as read:
                                read.seek(max(0, path.stat().st_size - 65536))
                                text = read.read().decode(errors="replace")
                        except OSError:
                            continue
                        tails.append(str(path.relative_to(output)) + "\n" + text[-1500:])
                        if path not in rank_by_path:
                            continue
                        if "CUBE555_TRACE_PATH" in case_env:
                            capture_trace(text)
                        merge_layer_samples(layer_samples, rank_by_path[path], text)
                        starts = list(re.finditer(r"depth_start=(\d+) frontier_size=(\d+)", text))
                        done = list(re.finditer(r"depth_done=(\d+) depth_sec=([\d.eE+\-]+)", text))
                        entry = {"rank": rank_by_path[path]}
                        if starts:
                            entry.update(depth=int(starts[-1][1]), frontier_size=int(starts[-1][2]))
                        if done:
                            entry.update(last_completed_depth=int(done[-1][1]),
                                         last_depth_seconds=float(done[-1][2]))
                        rank_progress.append(entry)
                    try:
                        current_preflight = read_json(folder / "preflight.json")
                        effective_now = current_preflight.get("plan", {}).get("effective_beam")
                    except (ValueError, OSError):
                        current_preflight = {}
                        effective_now = None
                    layer_timings = completed_layers(layer_samples, ranks, effective_now)
                    report["current_case"] = dict(puzzle_id=pid, requested_beam=beam, phase=phase,
                                                  wall_seconds=round(time.monotonic() - tick, 3),
                                                  effective_depth=current.get("effective_max_depth"),
                                                  effective_beam=effective_now,
                                                  memory_plan={key:current_preflight.get(key) for key in ('requested_max_depth','effective_max_depth','budget_max_depth','history_ram_bytes','history_disk_bytes')},
                                                  depth_timings=layer_timings,
                                                  completed_layers=completed_count(layer_samples, ranks),
                                                  full_layer_statistics=layer_summary(layer_timings),
                                                  rank_progress=rank_progress,
                                                  search_seconds=round(time.monotonic() - stage_started, 3) if phase == "running" else 0)
                    if rank_progress and all("depth" in rank for rank in rank_progress):
                        report["current_case"].update(
                            depth=min(rank["depth"] for rank in rank_progress),
                            frontier_size=sum(rank["frontier_size"] for rank in rank_progress))
                    if rank_progress and all("last_completed_depth" in rank for rank in rank_progress):
                        report["current_case"]["last_completed_depth"] = min(
                            rank["last_completed_depth"] for rank in rank_progress)
                    report["last_log"] = "\n".join(tails)[-6000:]
                    report['layer_events'] = [dict(rank=rank, depth=layer, puzzle_id=pid, **row)
                        for rank, rows in sorted(layer_samples.items()) for layer, row in sorted(rows.items())]
                    signature = (phase, json.dumps(layer_samples, sort_keys=True))
                    elapsed_now = time.monotonic()
                    if signature != last_signature or elapsed_now - last_published >= 60:
                        # Send a cumulative layer history up to once a minute during search.
                        # Startup and terminal writes outside this loop remain immediate.
                        to_drive = elapsed_now - last_drive_publish >= 60 or phase != (last_signature or (None,))[0]
                        save(publish=to_drive)
                        if to_drive:
                            last_drive_publish = elapsed_now
                        last_signature, last_published = signature, time.monotonic()
                rc = proc.wait()
        stage_seconds[previous_stage] = round(time.monotonic() - stage_started, 3)
        summary = read_json(folder / "run_summary.json")
        preflight = read_json(folder / "preflight.json")
        perf = read_json(folder / "performance.json")
        effective = preflight.get("plan", {}).get("effective_beam")
        # Final rank logs also contain the last layer, which can finish between polls.
        final_streams = {}
        for path in sorted(folder.rglob("stdout.log")):
            if path.parent.name.isdecimal():
                final_streams[int(path.parent.name)] = path
        for path in sorted(folder.rglob("rank-*.log")):
            match = re.fullmatch(r"rank-(\d+)\.log", path.name)
            if match:
                final_streams[int(match[1])] = path
        for rank, path in final_streams.items():
            final_text = path.read_text(errors="replace")
            merge_layer_samples(layer_samples, rank, final_text)
            if "CUBE555_TRACE_PATH" in case_env:
                capture_trace(final_text)
        layer_timings = completed_layers(layer_samples, ranks, effective)
        report['layer_events'] = [dict(rank=rank, depth=layer, puzzle_id=pid, **row)
            for rank, rows in sorted(layer_samples.items()) for layer, row in sorted(rows.items())]
        row = dict(kind=kind, puzzle_id=pid, requested_beam=beam, effective_beam=effective,
                   requested_depth=depth, effective_depth=summary.get("effective_max_depth"),
                   returncode=rc, wall_seconds=round(time.monotonic()-tick, 3),
                   results=summary.get("results", []), sampled_peak_gpu_mib=perf.get("sampled_peak_gpu_mib", {}),
                   depth5_seconds=None, depth5_frontiers=[], saturated=False,
                   depth_timings=layer_timings, raw_depth_timings=perf.get("depths", []), symmetry=symmetry,
                   full_layer_statistics=layer_summary(layer_timings),
                   memory_plan={key:preflight.get(key) for key in ('requested_max_depth','effective_max_depth','budget_max_depth','history_ram_bytes','history_disk_bytes')},
                   baseline_length=(baseline or {}).get("lengths", {}).get(pid), stage_seconds=stage_seconds,
                   native_search_seconds=round(sum(d["seconds"] for d in layer_timings), 3))
        row.update(symmetry_index=symmetry, inverse=inverse, completed_layers=completed_count(layer_samples, ranks))
        if cuda_diagnostics != 'off':
            row['cuda_diagnostics'] = dict(report['cuda_diagnostics'])
            if sanitizer_log.is_file():
                with sanitizer_log.open(encoding='utf-8', errors='replace') as diagnostic_stream:
                    row['cuda_diagnostics']['memcheck'] = diagnostic_stream.read(10000)
        if timed_out:
            row.update(status="resource_limit", search_outcome="session_deadline",
                       error="Достигнут лимит сессии 4 ч 45 мин с момента старта ноутбука.",
                       diagnostic="Сохранены все полученные до остановки результаты слоёв.")
        elif rc or summary.get("status") != "complete":
            tails = [*sorted(folder.rglob("cmake-*.log")), log, *sorted(folder.rglob("rank-*.log")),
                     *sorted(folder.rglob("stderr.log"))]
            messages = []
            for path in tails:
                with path.open("rb") as f:
                    f.seek(max(0, path.stat().st_size-15000))
                    messages.append(path.name + "\n" + f.read().decode(errors="replace"))
            detail = "\n".join(messages)
            shortage = re.findall(r"[^\n]*(?:out of memory|cudaErrorMemoryAllocation|budget.exceed|exceed.{0,28}budget|history.{0,20}too small|cannot preserve.{0,20}headroom)[^\n]*", detail, re.I)
            gpu_lines = re.findall(r"[^\n]*(?:illegal memory access|device-side assert|misaligned address|invalid device pointer)[^\n]*", detail, re.I)
            memcheck = row.get('cuda_diagnostics', {}).get('memcheck', '')
            first_fault = re.search(r'(?im)^=+\s+(?:Invalid (?:__|global|shared|local)|Program hit|Hardware exception|Uninitialized)[^\n]*(?:\n=+[ \t]+(?:at |by |Address )[^\n]*){0,5}', memcheck)
            stage_fault = re.search(r'(?im)^[^\n]*(?:cuda_stage_error|final request validation failed)[^\n]*', detail)
            gpu_fault = bool(gpu_lines or first_fault or stage_fault)
            diag = (first_fault.group() if first_fault else stage_fault.group() if stage_fault else
                    (gpu_lines or shortage or [detail[-3500:]])[0]).strip()
            last_started = max((n for rs in layer_samples.values() for n in rs), default=None)
            row.update(status="resource_limit" if shortage and not gpu_fault else "failed",
                       search_outcome="cuda_fault" if gpu_fault else "resource_limit" if shortage else "runner_error",
                       last_started_depth=last_started, diagnostic=diag,
                       error=(f"Ошибка CUDA на слое {last_started}: {diag}" if gpu_fault else
                              shortage[-1].strip() if shortage else summary.get("error") or detail[-900:]))
        else:
            row["status"] = "complete"
            row["search_outcome"] = "solution_found" if any(r.get("solution_count", 0) for r in row["results"]) else "depth_reached"
            for ranklog in sorted(folder.rglob("rank-*.log")):
                records = ranklog.read_text(errors="replace")
                frontier = re.search(r"depth_start=5 frontier_size=(\d+)", records)
                if frontier:
                    row["depth5_frontiers"].append(int(frontier[1]))
            times = [d["seconds"] for d in perf.get("depths", []) if d["depth"] == 5]
            row["depth5_seconds"] = times[0] if len(times) == 1 else None
            row["saturated"] = any(layer["saturated"] for layer in layer_timings)
            best = folder / f"puzzle-{pid}/solutions/solutions.csv"
            if best.exists():
                with best.open(encoding="utf-8") as f:
                    solved = list(csv.DictReader(f))
                if solved:
                    chosen = min(solved, key=lambda item: len(item["original_oriented_path"].split(".")) if item["original_oriented_path"] else 0)
                    moves = chosen["original_oriented_path"]
                    row["solution_orientation"] = chosen.get("variant", "original")
                    row["frame_path"] = moves
                    moves = original_path(moves, symmetry, inverse, frame_tables)
                    row["solution_orientation"] = f"symmetry_{symmetry}" + ("_inverse" if inverse else "")
                    with (inputs / "competition/test.csv").open(encoding="utf-8") as f:
                        state = next(x["initial_state"] for x in csv.DictReader(f) if int(x["initial_state_id"]) == pid)
                    state = list(map(int, state.split(",")))
                    info = read_json(inputs / "model/puzzle_info.json")
                    tokens = moves.split(".") if moves else []
                    replay_steps = []
                    for move_index, token in enumerate(tokens, 1):
                        state = [state[j] for j in info["generators"][token]]
                        replay_steps.append(dict(move_number=move_index, move=token, state=state.copy()))
                    if state != info["central_state"]:
                        raise RuntimeError("Independent 150-sticker replay failed")
                    row["solution"] = dict(puzzle_id=pid, path=moves, length=len(tokens),
                                           verified=True, requested_beam=beam, effective_beam=effective)
                    row['solution']['replay_steps'] = replay_steps
                    row["solution"].update(symmetry_index=symmetry, inverse=inverse)
                    if row["baseline_length"] is not None:
                        row["improvement"] = row["baseline_length"] - len(tokens)
        native_end = {}
        if row["status"] == "complete":
            for rank, logfile in final_streams.items():
                with logfile.open("rb") as stream:
                    stream.seek(max(0, logfile.stat().st_size - 64*1024))
                    tail = stream.read().decode(errors="replace")
                values = {k:int(v) for k,v in re.findall(r"(?m)^\s*(completed_depths|last_final_frontier_size)=(\d+)\s*$", tail)}
                if values:
                    native_end[rank] = values
            if len(native_end) == ranks and all("completed_depths" in v for v in native_end.values()):
                row["completed_layers"] = min(v["completed_depths"] for v in native_end.values())
        if row["status"] == "complete" and not row.get("solution"):
            finished = min((v.get("completed_depths", 0) for v in native_end.values()), default=0)
            last_width = (sum(r["last_final_frontier_size"] for r in native_end.values())
                          if len(native_end)==ranks and all("last_final_frontier_size" in r for r in native_end.values()) else None)
            limit = row["effective_depth"] or depth
            row.update(completed_depths=finished or None, last_frontier_size=last_width)
            if last_width == 0:
                outcome, reason = "frontier_empty", f"Луч исчерпан после {finished} слоёв. Состояний для следующего слоя: 0."
            elif finished >= limit and len(native_end)==ranks:
                outcome, reason = "depth_reached", f"Достигнут предел {limit} слоёв. Среди проверенных кандидатов не найден путь к шару радиуса 5."
            else:
                outcome, reason = "no_solution_unclassified", "Поиск завершился без пути. В журнале недостаточно данных для определения причины остановки."
            row.update(search_outcome=outcome, stop_reason=reason)
            report["stop_reason"] = reason
        report["attempts"].append(row)
        if row["status"] == "complete" and cuda_diagnostics == 'off':
            temp = checkpoint.with_suffix(".tmp")
            temp.write_text(json.dumps(dict(status="complete", run_id=runid, fingerprint=fingerprint,
                                            row=row), ensure_ascii=False), encoding="utf-8")
            temp.replace(checkpoint)
        report.pop("current_case", None)
        if kind != "capacity":
            if row["status"] == "complete":
                report["queue_search_finished"].append(pid)
                if row.get("solution"):
                    report["queue_completed"].append(pid)
            else:
                report["queue_failed"].append(dict(puzzle_id=pid, status=row["status"], error=row.get("error")))
        save()
        return row

    try:
        if type(symmetry) is not int or not 0 <= symmetry < 48 or type(inverse) is not bool:
            raise ValueError("TPU symmetry index must be 0..47 and inverse must be boolean")
        if sys.platform != 'linux':
            raise RuntimeError('Native CUDA notebook requires a Linux GPU runtime')
        import fcntl
        identity = dict(notebook_key=notebook_key, configuration=report['configuration'], queue=list(queue_ids),
                        capacity=list(capacity_beams), baseline_sha=(baseline or {}).get('sha256'),
                        commit=commit, source_sha256=source_hash, hostname=platform.node())
        session_path = cache / ('session-' + hashlib.sha256(str(notebook_key).encode()).hexdigest() + '.json')
        owner_path = cache / 'native-gpu.owner.json'
        job_lock = (cache / 'native-gpu.lock').open('a+')
        try:
            fcntl.flock(job_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            owner = read_json(owner_path)
            if owner.get('identity') != identity:
                raise RuntimeError('Another Cube555 calculation owns this GPU workspace: ' + str(owner.get('identity', {}).get('configuration')))
            existing_file = Path(owner['report_path']).resolve()
            if not existing_file.is_relative_to(cache):
                raise RuntimeError('Invalid active report path')
            while True:
                existing = read_json(existing_file)
                if existing and progress:
                    progress(existing, existing_file)
                if existing.get('status') in ('complete','partial','failed','gpu_unavailable'):
                    sink.enqueue(existing)
                    return existing, str(existing_file)
                try:
                    fcntl.flock(job_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    time.sleep(10)
                    continue
                existing = read_json(existing_file)
                if existing.get('status') in ('complete','partial','failed','gpu_unavailable'):
                    if progress:
                        progress(existing, existing_file)
                    sink.enqueue(existing)
                    return existing, str(existing_file)
                raise RuntimeError('Previous solver stopped without a final report; last run: ' + str(owner.get('run_id')))
        previous_session = read_json(session_path) if notebook_key else {}
        if previous_session.get('identity') == identity:
            prior_path = Path(previous_session['report_path']).resolve()
            if prior_path.is_relative_to(cache):
                prior_report = read_json(prior_path)
                if prior_report.get('status') in ('complete','partial','failed','gpu_unavailable'):
                    if progress:
                        progress(prior_report, prior_path)
                    sink.enqueue(prior_report)
                    return prior_report, str(prior_path)
        owner = dict(identity=identity, process_id=os.getpid(), run_id=runid, report_path=str(reportfile), started_utc=report['started_utc'])
        for target in (owner_path, session_path):
            temporary = target.with_suffix('.tmp')
            temporary.write_text(json.dumps(owner), encoding='utf-8')
            temporary.replace(target)
        report['messages'] = [dict(utc=datetime.now(timezone.utc).isoformat(), message='я запущен из маримо при старте', origin='molab_python')]
        save('starting')
        if drive_config and not sink.flush(90):
            raise RuntimeError('First Molab → Google Drive write failed: ' + str(sink.error))
        report['simple_operation'] = dict(expression='1 + 1', result=1 + 1)
        report['messages'].append(dict(utc=datetime.now(timezone.utc).isoformat(), message='я запущен чуть позже из маримо', origin='molab_python'))
        save('drive_probe_complete')
        if drive_config and not sink.flush(90):
            raise RuntimeError('Second Molab → Google Drive write failed: ' + str(sink.error))
        save("probing_gpu")
        smi = shutil.which("nvidia-smi")
        if smi:
            smi_result = command([smi, "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader,nounits"], capture=True, check=False)
            if smi_result.returncode == 0:
                report["gpus"] = [line.strip() for line in smi_result.stdout.splitlines() if line.strip()]
        if not report["gpus"]:
            report["next_action"] = "Molab: нажмите на название текущей машины в верхней панели, выберите NVIDIA GPU и запустите все ячейки."
            save("gpu_unavailable")
            return report, str(reportfile)
        if type(puzzle_id) is not int or not 0 <= puzzle_id <= 1034:
            raise ValueError("Cube ID must be an integer in 0..1034")
        if type(requested_beam) is not int or not 1 <= requested_beam < 2**64:
            raise ValueError("Beam must be a positive uint64 integer")
        if type(requested_depth) is not int or not 1 <= requested_depth < 2**32:
            raise ValueError("Maximum depth must be a positive uint32 integer")
        if type(requested_beam) is not int or not 1 <= requested_beam < 2**64:
            raise ValueError("Requested beam must be a positive uint64")
        if any(type(b) is not int or not 1 <= b < 2**64 for b in capacity_beams):
            raise ValueError("Invalid capacity beam")
        save("preparing_cuda_toolchain")
        python = prepare_env()
        probe = command([python, "-c", "import torch,json;print(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,devices=[dict(name=torch.cuda.get_device_name(i),cc=list(torch.cuda.get_device_capability(i)),bytes=torch.cuda.get_device_properties(i).total_memory) for i in range(torch.cuda.device_count())])))"], capture=True)
        hardware = json.loads(probe.stdout.splitlines()[-1])
        report["torch_hardware"] = hardware
        devices = hardware["devices"][:2]
        ranks = len(devices)
        if not ranks:
            raise RuntimeError("NVIDIA driver visible but PyTorch CUDA initialization found no devices")
        architectures = sorted({tuple(dev["cc"]) for dev in devices})
        arch = ";".join(f"{major}{minor}" for major, minor in architectures)
        env.update(CUBE555_WORLD_SIZE=str(ranks), CUBE555_CUDA_ARCH=arch,
                   CUBE555_MAX_DEPTH=str(requested_depth),
                   CUBE555_DEVICE_BYTES=str(min(dev["bytes"] for dev in devices)),
                   TORCH_CUDA_ARCH_LIST=";".join(f"{a}.{b}" for a, b in architectures),
                   CMAKE_CUDA_ARCHITECTURES=arch,
                   CAYLEYPY_BUILD_DIR=str(output / "build"))
        save("verifying_source_and_inputs")
        repo = prepare_repo()
        inputs = prepare_inputs()
        search_competition = inputs / "competition"
        frame_tables = symmetry_tables()
        validate_symmetry_tables(frame_tables, read_json(inputs / "model/puzzle_info.json"))
        report["symmetry_data_sha256"] = frame_tables["sha256"]
        report["symmetry_source"] = frame_tables["source"]
        search_competition = prepare_frame_competition(search_competition, output / "frame-competition",
                                                      symmetry, inverse, frame_tables)
        # Solve the requested cube at the exact width selected in the form.
        solved = case("solve", puzzle_id, requested_beam, requested_depth)
        if solved.get("solution"):
            report["solution"] = solved["solution"]
        if solved["status"] == "failed":
            raise RuntimeError(solved["error"] + " — " + solved.get("diagnostic", ""))
        # Measure the same saturated iteration as the retained original run.
        for beam in sorted(set(capacity_beams)):
            probe_row = case("capacity", 165, beam, 6)
            if probe_row["status"] == "failed":
                raise RuntimeError(probe_row["error"] + " — " + probe_row.get("diagnostic", ""))
            if probe_row["saturated"]:
                report["max_verified_saturated_beam"] = dict(requested=beam, effective=probe_row["effective_beam"],
                        depth5_seconds=probe_row["depth5_seconds"], tested_candidates=sorted(set(capacity_beams)))
                if beam == 3_500_000:
                    report["reference_comparison"] = dict(original_two_t4_seconds=374.932,
                        reference_commit=commit, notebook="kamilla1158/cube555-native-2xt4-blend", version=485,
                        pid=165, requested_beam=3500000, reference_effective_beam=3506176,
                        observed_effective_beam=probe_row["effective_beam"], depth_done=5,
                        ratio_reference_to_observed=374.932 / probe_row["depth5_seconds"],
                        note="Saturated iteration, same commit, model weights and batch sizes. GPU count changes beam alignment and rank distribution.")
                if not report["solution"] and beam != requested_beam:
                    retry = case("solve-at-tested-beam", puzzle_id, beam, requested_depth)
                    report["solution"] = retry.get("solution")
        for queued_pid in queue_ids:
            if queued_pid == puzzle_id:
                continue
            queued = case("length111", queued_pid, requested_beam, requested_depth)
            if queued["status"] == "failed":
                raise RuntimeError(queued["error"] + " — " + queued.get("diagnostic", ""))
            save()
        if set(queue_ids).issubset(report["queue_completed"]) and report["solution"]:
            save("complete")
        else:
            if report["queue_failed"]:
                report["error"] = "\n".join(
                    f'PID {item["puzzle_id"]}: {item["status"]}: {item.get("error") or ""}'
                    for item in report["queue_failed"])
            elif report["queue_search_finished"]:
                report["stop_reason"] = "depth_reached_without_solution"
            save("partial")
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error)[-4500:])
        if isinstance(error, subprocess.CalledProcessError) and error.stdout:
            report["subprocess_diagnostic"] = error.stdout[-3000:]
        save("failed")
    finally:
        if not sink.flush():
            report['drive_sync'] = sink.snapshot()
            report['drive_sync']['warning'] = 'Final report not acknowledged by Drive; durable outbox retained'
            reportfile.write_text(json.dumps(report, ensure_ascii=False), encoding='utf-8')
            if progress:
                progress(report, reportfile)
        if job_lock:
            job_lock.close()
    return report, str(reportfile)
