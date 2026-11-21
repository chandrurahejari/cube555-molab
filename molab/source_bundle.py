"""Create the native payload exclusively from objects in the indicated Git commit."""
import base64
import hashlib
import io
import json
import subprocess
import tarfile
import zipfile


SOURCE_PATHS = ("CMakeLists.txt", "UPSTREAM.json", "src", "cuda", "cmake", "tests", "configs", "tools", "vendor")
ORIGINAL_INPUTS_SHA256 = "49d61a127540cd197db9bf75aacad5712a7ab54d3fbcb39bf2da9ae9bb477286"


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def add_file(archive, name, data):
    entry = tarfile.TarInfo(name)
    entry.size = len(data)
    entry.mode = 0o644
    archive.addfile(entry, io.BytesIO(data))


def pack_sources(repo, commit):
    packed = git(repo, "archive", "--format=zip", commit, "--", *SOURCE_PATHS)
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(packed)) as source, tarfile.open(fileobj=output, mode="w:xz", preset=5) as dest:
        upstream = json.loads(source.read("UPSTREAM.json"))
        vendor_name = f"vendor/cutlass-{upstream['cutlass_commit'][:7]}.zip"
        for entry in source.infolist():
            if not entry.is_dir() and not entry.filename.startswith("vendor/"):
                add_file(dest, entry.filename, source.read(entry))
        with zipfile.ZipFile(io.BytesIO(source.read(vendor_name))) as headers:
            for entry in headers.infolist():
                if not entry.is_dir():
                    add_file(dest, "external/cutlass/" + entry.filename, headers.read(entry))
        add_file(dest, "external/cutlass/CUBE555_CUTLASS_REVISION", upstream["cutlass_commit"].encode())
    return encoded(output.getvalue())


def pack_small_inputs(path):
    # Competition data stays in the notebook built for this user. Weights are
    # downloaded by their original filenames and checked against this manifest.
    with path.open("rb") as f:
        actual = hashlib.file_digest(f, "sha256").hexdigest()
    if actual != ORIGINAL_INPUTS_SHA256:
        raise ValueError(f"Original Cube555 inputs archive has SHA-256 {actual}")
    output = io.BytesIO()
    with zipfile.ZipFile(path) as original, tarfile.open(fileobj=output, mode="w:xz", preset=5) as dest:
        manifest = json.loads(original.read("manifest.json"))
        for name in ["manifest.json", *sorted(manifest)]:
            if not name.endswith(".pt"):
                add_file(dest, name, original.read(name))
    return encoded(output.getvalue())


def encoded(data):
    return hashlib.sha256(data).hexdigest(), base64.b64encode(data).decode("ascii")
