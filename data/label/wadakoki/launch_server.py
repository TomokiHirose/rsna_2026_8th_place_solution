"""config の設定で llama.cpp のサーバー（``llama-server``）を 127.0.0.1 に立て、``/health`` が ok になるまで待つ。

``generate.py`` の前に 1 度だけ動かす。サーバーのログは ``--output/server.log``、起動したコマンドは ``--output/server.json``。
wadakoki の v0.4 のラベルは llama.cpp ``b11026`` の GPU 版（Colab の A100、全層を GPU に載せる）で作った。

    uv run python data/label/wadakoki/launch_server.py --binary <llama-server> --model <gguf> --config data/label/wadakoki/gemma_v04.json --output output/llm_labels/gemma_v04/server
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import click


def server_command(binary: Path, model: Path, cfg: dict) -> list[str]:
    """``llama-server`` の引数。並列数 x 1 本あたりの文脈長を文脈長にし、thinking と予算は config の ``generation`` に合わせる。"""
    engine, generation = cfg["engine"], cfg["generation"]
    return [
        str(binary), "--model", str(model), "--alias", engine["alias"], "--host", "127.0.0.1", "--port", str(engine["port"]), "--n-gpu-layers", "999",
        "--parallel", str(engine["parallel"]), "--ctx-size", str(engine["parallel"] * engine["context_per_slot"]),
        "--flash-attn", "on", "--no-context-shift", "--no-webui", "--metrics",
        "--reasoning", "on" if generation["enable_thinking"] else "off", "--reasoning-budget", str(generation["reasoning_budget"]), "--reasoning-format", "deepseek", "--cache-ram", "1024",
    ]  # fmt: skip


@click.command(help="Start llama-server on 127.0.0.1 with the config's settings and wait until /health is ok.")
@click.option("--binary", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="llama.cpp llama-server.")
@click.option("--model", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="The GGUF named by the config's model.repository, revision and filename.")
@click.option("--config", type=click.Path(exists=True, path_type=Path), required=True, help="data/label/wadakoki/<gemma_v04|qwen_v04>.json")
@click.option("--output", type=click.Path(path_type=Path), required=True, help="Log directory (must not exist).")
def main(binary: Path, model: Path, config: Path, output: Path) -> None:
    cfg = json.loads(config.read_text())
    output.mkdir(parents=True, exist_ok=False)
    cmd = server_command(binary, model, cfg)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(binary.parent) + ":" + env.get("LD_LIBRARY_PATH", "")
    with (output / "server.log").open("x") as log:
        process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (output / "server.json").write_text(json.dumps({"pid": process.pid, "command": cmd}))
    for _ in range(120):
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited; see {output / 'server.log'}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{cfg['engine']['port']}/health", timeout=2) as response:
                if json.load(response).get("status") == "ok":
                    print("INFERENCE_SERVER_READY", process.pid, flush=True)
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(2)
    process.terminate()
    raise TimeoutError("llama-server did not become ready within 240 s")


if __name__ == "__main__":
    main()
