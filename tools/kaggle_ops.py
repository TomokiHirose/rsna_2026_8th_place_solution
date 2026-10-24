from __future__ import annotations

import contextlib
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

import click

REPO_ROOT = Path(__file__).resolve().parents[1]
KAGGLE = str(REPO_ROOT / ".venv" / "bin" / "kaggle") if (REPO_ROOT / ".venv" / "bin" / "kaggle").exists() else shutil.which("kaggle")
COMPETITION = "rsna-knee-abnormality-detection"
PKG_SLUG = "rsna-knee-pkg"
MODEL_SLUG = "rsna-knee-model"
# Staging root; it must be on the same filesystem as the artifacts because files are hard-linked
STAGE_ROOT = REPO_ROOT / "output"
SUBMIT_STORE = REPO_ROOT / "output" / "submit"
KERNEL_DIRS = {target: REPO_ROOT / "kernels" / target for target in ("A", "B", "C")}
PLACEHOLDER = "OWNER"
# ``--dir-mode zip`` strips one top-level directory, so uploads are wrapped in this one
WRAPPER = "payload"


class OpsError(RuntimeError):
    """A failed Kaggle operation; main prints it on one line and exits with code 1."""


def stamp() -> str:
    return time.strftime("%H:%M:%S")


def run(cmd: list, *, check: bool = True, echo: bool = True) -> subprocess.CompletedProcess:
    """Run one external command without a shell and capture its output."""
    argv = [str(part) for part in cmd]
    if echo:
        print(f"$ {' '.join(argv)}", flush=True)
    proc = subprocess.run(argv, cwd=REPO_ROOT, capture_output=True, text=True)
    if check and proc.returncode != 0:
        tail = ((proc.stdout or "") + (proc.stderr or "")).strip().splitlines()[-3:]
        raise OpsError(f"{argv[0]} {argv[1] if len(argv) > 1 else ''} failed with rc={proc.returncode}: {' / '.join(tail)}")
    return proc


def kaggle_text(*args: object, check: bool = True) -> str:
    """Run the kaggle CLI quietly and return stdout+stderr without its "Warning:" lines (e.g. the outdated-version notice printed first)."""
    proc = run([KAGGLE, *args], check=check, echo=False)
    return "\n".join(line for line in ((proc.stdout or "") + (proc.stderr or "")).splitlines() if not line.startswith("Warning:"))


def is_server_error(text: str) -> bool:
    """Whether the output is a Kaggle 5xx, the only error that is retried."""
    low = text.lower()
    if any(word in low for word in ("internal server error", "bad gateway", "service unavailable", "gateway time")):
        return True
    return bool(re.search(r"\b50[0-9]\b", low)) and ("error" in low or "server" in low)


def dataset_files(dataset: str) -> dict[str, str]:
    """Return every file of the dataset as {path: creationDate}, following page tokens."""
    files: dict[str, str] = {}
    token: str | None = None
    for _ in range(50):
        args: list[object] = ["datasets", "files", dataset, "-v", "--page-size", "200"]
        if token:
            args += ["--page-token", token]
        lines = kaggle_text(*args).splitlines()
        token = next((line.split("=", 1)[1].strip() for line in lines if line.startswith("Next Page Token")), None) or None
        rows = [line for line in lines if line.strip() and not line.startswith("Next Page Token")]
        for row in csv.DictReader(rows):
            if row.get("name") != "name":
                files[row["name"]] = row.get("creationDate", "")
        if not token:
            break
    return files


def dataset_ready(dataset: str) -> bool:
    return kaggle_text("datasets", "status", dataset, check=False).strip().lower().startswith("ready")


def wait_until(what: str, probe, *, timeout_min: float, interval_s: float = 30.0) -> None:
    """Poll probe() until it returns true, giving up after timeout_min."""
    deadline = time.monotonic() + timeout_min * 60.0
    while True:
        if probe():
            print(f"{stamp()} {what}: ok", flush=True)
            return
        if time.monotonic() >= deadline:
            raise OpsError(f"{what}: not done after {timeout_min:g} min (Kaggle may still be processing; rerun the same command later)")
        time.sleep(interval_s)


@contextlib.contextmanager
def staging(owner: str, slug: str, keep: bool) -> Iterator[Path]:
    """Yield a dataset staging directory with its metadata; remove it on success unless ``keep``."""
    STAGE_ROOT.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=f"{slug}-", dir=STAGE_ROOT))
    meta = {"title": slug, "id": f"{owner}/{slug}", "licenses": [{"name": "CC0-1.0"}]}
    (directory / "dataset-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    try:
        yield directory
    except BaseException:
        print(f"{stamp()} kept staging directory: {directory}", file=sys.stderr, flush=True)
        raise
    if keep:
        print(f"{stamp()} kept staging directory: {directory}", flush=True)
    else:
        shutil.rmtree(directory)


def link(src: Path, dest: Path) -> None:
    """Hard-link ``src`` to ``dest``."""
    try:
        os.link(src, dest)
    except OSError as exc:
        raise OpsError(f"cannot hard-link {src} to {dest} (different filesystem?): {exc}") from exc


def dataset_exists(dataset: str) -> bool:
    return run([KAGGLE, "datasets", "status", dataset], check=False, echo=False).returncode == 0


def datasets_version(directory: Path, message: str, *, create: bool, attempts: int = 4) -> None:
    """Run `kaggle datasets create` (first upload, private) or `kaggle datasets version`, retrying only on Kaggle 5xx."""
    for attempt in range(1, attempts + 1):
        cmd = [KAGGLE, "datasets", "create", "-p", directory, "--dir-mode", "zip"] if create else [KAGGLE, "datasets", "version", "-p", directory, "-m", message, "--dir-mode", "zip"]
        proc = run(cmd, check=False)
        text = ((proc.stdout or "") + (proc.stderr or "")).strip()
        if text:
            print(text, flush=True)
        if proc.returncode == 0 and not is_server_error(text):
            return
        if attempt == attempts or not is_server_error(text):
            raise OpsError(f"kaggle datasets {cmd[2]} failed (rc={proc.returncode})")
        print(f"{stamp()} Kaggle 5xx; retrying in 30 s ({attempt + 1}/{attempts})", flush=True)
        time.sleep(30)


def upload_and_wait(dataset: str, directory: Path, message: str, wanted: set[str], timeout_min: float) -> None:
    """Version the dataset and wait until it is ready with a new file list containing ``wanted``."""
    create = not dataset_exists(dataset)
    before = {} if create else dataset_files(dataset)
    datasets_version(directory, message, create=create)

    def replaced() -> bool:
        if not dataset_ready(dataset):
            return False
        files = dataset_files(dataset)
        return wanted <= set(files) and files != before

    wait_until(f"{dataset} new version live", replaced, timeout_min=timeout_min)


@click.group(help="Kaggle operations to reproduce a submission: pkg dataset, model dataset, inference kernel, submit.")
@click.option("--owner", required=True, help="Kaggle username that owns the datasets and kernels")
@click.pass_context
def main(ctx: click.Context, owner: str) -> None:
    ctx.obj = owner


@contextlib.contextmanager
def ops_errors() -> Iterator[None]:
    """Report an OpsError as a click error (exit code 1)."""
    try:
        yield
    except OpsError as error:
        raise click.ClickException(str(error)) from error


@main.command("pkg", help=f"upload rsna_knee/ as <owner>/{PKG_SLUG} and wait until live")
@click.option("-m", "--message", default="sync rsna_knee", show_default=True, help="version message")
@click.option("--build-only", is_flag=True, help="build the staging directory only, do not upload")
@click.option("--timeout-min", type=float, default=30.0, show_default=True, help="max minutes to wait until live")
@click.pass_obj
def cmd_pkg(owner: str, message: str, build_only: bool, timeout_min: float) -> None:
    """Upload rsna_knee/ as a new version of the pkg dataset and wait until it is live."""
    with ops_errors(), staging(owner, PKG_SLUG, build_only) as directory:
        root = directory / WRAPPER / "rsna_knee"
        shutil.copytree(REPO_ROOT / "rsna_knee", root, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
        print(f"collected: {sum(1 for _ in root.rglob('*.py'))} .py files from rsna_knee")
        if build_only:
            return
        upload_and_wait(f"{owner}/{PKG_SLUG}", directory, message, {"rsna_knee/inference.py"}, timeout_min)


def build_submit_model(directory: Path) -> list[str]:
    """Hard-link every experiment in output/submit/ into the staging directory and return the placed folds."""
    sys.path.insert(0, str(REPO_ROOT))
    from rsna_knee.experiments import ExperimentConfig

    dests = []
    for experiment in sorted(path for path in SUBMIT_STORE.iterdir() if path.is_dir()):
        config = ExperimentConfig.model_validate_json((experiment / "inference.json").read_text())
        target = directory / WRAPPER / experiment.name
        target.mkdir(parents=True)
        link(experiment / "inference.json", target / "inference.json")
        folds = sorted(path for path in experiment.glob("fold*") if path.is_dir())
        if not folds:
            raise OpsError(f"no fold in submit store: {experiment}")
        for fold in folds:
            (target / fold.name).mkdir()
            for filename in ("model.pt", "config.json"):
                if not (fold / filename).is_file():
                    raise OpsError(f"missing artifact: {fold / filename}")
                link(fold / filename, target / fold.name / filename)
            if config.model.kind == "vlm":
                shutil.copytree(fold / "qwen", target / fold.name / "qwen")
            dests.append(f"{experiment.name}/{fold.name}")
    if not dests:
        raise OpsError(f"no experiment in submit store: {SUBMIT_STORE}")
    return dests


@main.command("model", help=f"upload output/submit/ as <owner>/{MODEL_SLUG} and wait until live")
@click.option("-m", "--message", default="update members", show_default=True, help="version message")
@click.option("--build-only", is_flag=True, help="build the staging directory only, do not upload")
@click.option("--timeout-min", type=float, default=30.0, show_default=True, help="max minutes to wait until live")
@click.pass_obj
def cmd_model(owner: str, message: str, build_only: bool, timeout_min: float) -> None:
    """Upload the whole output/submit/ store as a new version of the model dataset."""
    with ops_errors(), staging(owner, MODEL_SLUG, build_only) as directory:
        dests = build_submit_model(directory)
        total = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
        print(f"collected: {len(dests)} folds, {total / 2**20:.0f} MiB")
        for dest in dests:
            print(f"  {dest}")
        if build_only:
            return
        wanted = {f"{dest}/model.pt" for dest in dests} | {f"{dest.split('/')[0]}/inference.json" for dest in dests}
        upload_and_wait(f"{owner}/{MODEL_SLUG}", directory, message, wanted, timeout_min)


def kernel_id(owner: str, target: str) -> str:
    """The kernel id ``<owner>/<slug>``, with the slug read from the target's metadata."""
    placeholder_id = json.loads((KERNEL_DIRS[target] / "kernel-metadata.json").read_text())["id"]
    return f"{owner}/{placeholder_id.split('/', 1)[1]}"


def stage_kernel(source: Path, directory: Path, owner: str) -> None:
    """Copy the notebook and metadata into ``directory`` with the placeholder owner replaced."""
    meta = json.loads((source / "kernel-metadata.json").read_text())
    meta["id"] = meta["id"].replace(f"{PLACEHOLDER}/", f"{owner}/", 1)
    meta["dataset_sources"] = [ref.replace(f"{PLACEHOLDER}/", f"{owner}/", 1) for ref in meta["dataset_sources"]]
    (directory / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    notebook = json.loads((source / meta["code_file"]).read_text())
    line = f'{PLACEHOLDER} = "{PLACEHOLDER}"'
    hits = 0
    for cell in notebook["cells"]:
        source_lines = cell["source"] if isinstance(cell["source"], list) else cell["source"].splitlines(keepends=True)
        hits += sum(1 for text in source_lines if text.startswith(line))
        cell["source"] = [f'{PLACEHOLDER} = "{owner}"' + text[len(line) :] if text.startswith(line) else text for text in source_lines]
    if hits != 1:
        raise OpsError(f"expected exactly one line `{line}` in {source / meta['code_file']}, found {hits}")
    (directory / meta["code_file"]).write_text(json.dumps(notebook, indent=1) + "\n")


def kernel_status(kid: str) -> str:
    """Status of the kernel's latest run (COMPLETE / ERROR / RUNNING ...), empty if unreadable."""
    match = re.search(r"KernelWorkerStatus\.([A-Z_]+)", kaggle_text("kernels", "status", kid, check=False))
    return match.group(1) if match else ""


def wait_for_kernel(kid: str, target: str, version: int, timeout_min: float) -> None:
    """Wait until the kernel is COMPLETE; on ERROR download its output to output/logs/ and fail."""
    deadline = time.monotonic() + timeout_min * 60.0
    while True:
        status = kernel_status(kid)
        if status == "COMPLETE":
            print(f"{stamp()} kernel v{version} COMPLETE")
            return
        if status in ("ERROR", "CANCEL_ACKNOWLEDGED"):
            out_dir = REPO_ROOT / "output" / "logs" / target / f"kernel_v{version}"
            out_dir.mkdir(parents=True, exist_ok=True)
            run([KAGGLE, "kernels", "output", kid, "-p", out_dir], check=False, echo=False)
            for path in sorted(out_dir.glob("*.log")):
                print(f"--- {path} (tail) ---")
                print("\n".join(path.read_text(errors="replace").splitlines()[-40:]))
            raise OpsError(f"kernel v{version} ended with {status} (logs in {out_dir.relative_to(REPO_ROOT)}/)")
        if time.monotonic() >= deadline:
            raise OpsError(f"kernel v{version} did not finish in {timeout_min:g} min (last status: {status or 'unknown'})")
        time.sleep(30)


@main.command("kernel", help="push the inference kernel in kernels/<target>/")
@click.option("--target", type=click.Choice(list(KERNEL_DIRS)), required=True, help="kernel to push")
@click.option("--wait", is_flag=True, help="wait for COMPLETE / ERROR (on ERROR, logs go to output/logs/<target>/kernel_v<N>/)")
@click.option("--timeout-min", type=float, default=240.0, show_default=True, help="max minutes to wait for the kernel")
@click.pass_obj
def cmd_kernel(owner: str, target: str, wait: bool, timeout_min: float) -> None:
    """Push kernels/<target>/ with the owner filled in (``--wait`` waits for completion)."""
    with ops_errors():
        kid = kernel_id(owner, target)
        with tempfile.TemporaryDirectory() as tmp:
            stage_kernel(KERNEL_DIRS[target], Path(tmp), owner)
            proc = run([KAGGLE, "kernels", "push", "-p", tmp], check=False)
        text = ((proc.stdout or "") + (proc.stderr or "")).strip()
        print(text, flush=True)
        match = re.search(r"Kernel version (\d+)", text)
        if proc.returncode != 0 or not match:
            raise OpsError("kernel push failed (could not read the version number)")
        version = int(match.group(1))
        print(f"{stamp()} kernel version {version} pushed: {kid}")
        if wait:
            wait_for_kernel(kid, target, version, timeout_min)


@main.command("submit", help="submit a kernel version to the competition")
@click.option("--target", type=click.Choice(list(KERNEL_DIRS)), required=True, help="kernel to submit")
@click.option("--version", type=int, required=True, help="kernel version number")
@click.option("--message", required=True, help="submission message")
@click.pass_obj
def cmd_submit(owner: str, target: str, version: int, message: str) -> None:
    """Submit one version of the target kernel to the competition."""
    with ops_errors():
        run([KAGGLE, "competitions", "submit", "-c", COMPETITION, "-f", "submission.csv", "-k", kernel_id(owner, target), "-v", str(version), "-m", message])


if __name__ == "__main__":
    main()
