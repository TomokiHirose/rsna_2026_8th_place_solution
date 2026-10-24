from __future__ import annotations

import copy
import logging
import multiprocessing as mp
import threading
import time
import warnings
from collections import deque
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from rsna_knee.constants import NUM_LABELS
from rsna_knee.data.dicom import PreprocessParams, StudyTensor, build_study_variants
from rsna_knee.inference import canonical_device, predict_study
from rsna_knee.models import WindowModel

__all__ = ["DEFAULT_PREFETCH", "DEFAULT_WORKERS", "Bank", "PhaseCounters", "StudyStream", "iter_studies", "place_models", "score_stream"]

logger = logging.getLogger("rsna_knee.pipeline")

#: デコードを担う worker process の数。Kaggle の 4 vCPU で、main（配膳）と GPU ごとの consumer thread を残した値。
DEFAULT_WORKERS = 3
#: main より先にデコードしておく study の上限。深さは RAM ではなく GPU を待たせないことで決める。
DEFAULT_PREFETCH = 6

#: study 1 件のデコードをこれ以上待たない。``Pool`` は worker が死ぬ（OOM kill）とその task を失い、``imap`` は
#: 永久に待つ。NFS のコールド読みでも 1 study は数秒なので、10 分は故障以外では届かない。
DECODE_TIMEOUT_S = 600.0

#: fork した worker が initializer で受け取る、束ごとの前処理レシピ。
_GROUP_PARAMS: list[PreprocessParams] = []


@dataclass
class PhaseCounters:
    """段ごとの実時間。main thread・consumer thread・worker がそれぞれ自分の項目だけを書く。

    ``decode_s`` はデコード、``decode_wait_s`` は main の待ち、``gpu_s[device]`` はその device の採点の合計。
    """

    decode_s: float = 0.0
    decode_wait_s: float = 0.0
    n_decoded: int = 0
    gpu_s: dict[str, float] = field(default_factory=dict)
    n_scored: dict[str, int] = field(default_factory=dict)

    def summary(self, n_studies: int) -> str:
        """study 1 件あたりに直した 1 行。"""
        n = max(n_studies, 1)
        parts = [f"decode {self.decode_s / n:.3f}s/study in workers, main waited {self.decode_wait_s / n:.3f}s/study"]
        for name, seconds in self.gpu_s.items():
            count = self.n_scored.get(name, 0)
            line = f"{name} {seconds / max(count, 1):.3f}s/study x{count}"
            if name.startswith("cuda"):
                line += f" peak {torch.cuda.max_memory_allocated(torch.device(name)) / 2**30:.2f}GiB"
            parts.append(line)
        return "; ".join(parts)


def _init_worker(group_params: list[PreprocessParams]) -> None:
    """fork した子の初期化。レシピを受け取り、cv2 と torch のスレッド数を 1 に固定する。"""
    global _GROUP_PARAMS
    _GROUP_PARAMS = list(group_params)
    torch.set_num_threads(1)
    import cv2  # noqa: PLC0415 - rsna_knee.data.dicom が既に読んでいる

    cv2.setNumThreads(1)


def _decode(task: tuple[int, str, list[dict]]) -> tuple[int, list[StudyTensor], float]:
    """worker 側の 1 study。``(index, studies, decode_seconds)`` を返す。例外はそのまま Pool が親へ運ぶ。"""
    index, study_dir, rows = task
    started = time.perf_counter()
    studies = build_study_variants(study_dir, rows, _GROUP_PARAMS, study_id=Path(study_dir).name)
    return index, studies, time.perf_counter() - started


class StudyStream:
    """study を worker process でデコードし、``(index, studies)`` を投入順に返す反復子。

    ``workers > 0`` なら ``fork`` の Pool を **コンストラクタで** 作る（子で CUDA は使えないので、CUDA を
    初期化する前に作ること）。先読みは ``prefetch`` 件まで。worker の例外は ``(index, exception)`` として返す。
    """

    def __init__(self, items: Sequence[tuple[str | Path, list[dict]]], group_params: Sequence[PreprocessParams], workers: int = DEFAULT_WORKERS, prefetch: int = DEFAULT_PREFETCH, counters: PhaseCounters | None = None) -> None:
        if workers < 0:
            raise ValueError(f"workers must be >= 0, got {workers}")
        if prefetch < 1:
            raise ValueError(f"prefetch must be >= 1, got {prefetch}")
        self.items = [(str(study_dir), list(rows)) for study_dir, rows in items]
        self.group_params = list(group_params)
        self.counters = counters if counters is not None else PhaseCounters()
        self.prefetch = prefetch
        self._stop = threading.Event()
        self._pool = None
        if workers > 0:
            with warnings.catch_warnings():
                # 子が走らせるのは numpy / cv2 / PIL / pydicom だけで、cv2 のプールは initializer で畳む。
                warnings.filterwarnings("ignore", message="This process .* is multi-threaded", category=DeprecationWarning)
                self._pool = mp.get_context("fork").Pool(workers, initializer=_init_worker, initargs=(self.group_params,))
        self._gen = self._inline() if self._pool is None else self._pooled()

    def __iter__(self) -> StudyStream:
        return self

    def __next__(self) -> tuple[int, list[StudyTensor] | Exception]:
        return next(self._gen)

    def __enter__(self) -> StudyStream:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """worker を止める。使い切った後は何もしない。途中なら投入を止めて Pool を terminate する。"""
        self._stop.set()
        self._gen.close()
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

    def _inline(self) -> Iterator[tuple[int, list[StudyTensor] | Exception]]:
        for index, (study_dir, rows) in enumerate(self.items):
            if self._stop.is_set():
                return
            started = time.perf_counter()
            try:
                studies: list[StudyTensor] | Exception = build_study_variants(study_dir, rows, self.group_params, study_id=Path(study_dir).name)
            except Exception as exc:  # 呼び出し側に判断させる。ここで握り潰すと黙って中立値になる
                studies = exc
            elapsed = time.perf_counter() - started
            self.counters.decode_wait_s += elapsed
            if not isinstance(studies, Exception):
                self.counters.decode_s += elapsed
                self.counters.n_decoded += 1
            yield index, studies

    def _feed(self, sem: threading.Semaphore) -> Iterator[tuple[int, str, list[dict]]]:
        """Pool の task handler thread が回す投入側。main が受け取るまで ``prefetch`` 件で止まる。"""
        for index, (study_dir, rows) in enumerate(self.items):
            while not sem.acquire(timeout=0.05):  # close() で止められるよう、待ちは短く刻む
                if self._stop.is_set():
                    return
            if self._stop.is_set():
                return
            yield index, study_dir, rows

    def _pooled(self) -> Iterator[tuple[int, list[StudyTensor] | Exception]]:
        assert self._pool is not None
        sem = threading.Semaphore(self.prefetch)
        results = self._pool.imap(_decode, self._feed(sem), chunksize=1)  # imap は順序を保つ
        for index in range(len(self.items)):
            started = time.perf_counter()
            try:
                got_index, studies, elapsed = results.next(timeout=DECODE_TIMEOUT_S)
            except StopIteration:  # close() が投入を止めた
                return
            except mp.TimeoutError as exc:
                dead = [p.pid for p in self._pool._pool if p.exitcode is not None]  # noqa: SLF001 - Pool は死んだ worker を公開しない
                raise RuntimeError(f"study #{index} ({self.items[index][0]}) was not decoded within {DECODE_TIMEOUT_S:.0f}s; dead workers: {dead or 'none'} (an OOM-killed worker loses its task and imap waits forever)") from exc
            except Exception as exc:  # worker の例外。位置は投入順なので index で分かる
                self.counters.decode_wait_s += time.perf_counter() - started
                sem.release()
                yield index, exc
                continue
            self.counters.decode_wait_s += time.perf_counter() - started
            self.counters.decode_s += elapsed
            self.counters.n_decoded += 1
            sem.release()
            assert got_index == index, f"study order broke: got {got_index}, expected {index}"
            yield index, studies
        self._pool.close()
        self._pool.join()
        self._pool = None


def iter_studies(items: Sequence[tuple[str | Path, list[dict]]], group_params: Sequence[PreprocessParams], workers: int = DEFAULT_WORKERS, prefetch: int = DEFAULT_PREFETCH, counters: PhaseCounters | None = None) -> StudyStream:
    """``(study_dir, series_rows)`` の列を、束ごとのレシピでデコードして ``(index, studies)`` を順に返す。

    ``studies`` は ``group_params`` と同じ順の :class:`StudyTensor` か、デコードが失敗していればその例外。
    ``workers > 0`` なら **この呼び出しで** fork するので、CUDA を触る前に呼ぶこと。
    """
    return StudyStream(items, group_params, workers=workers, prefetch=prefetch, counters=counters)


@dataclass
class Bank:
    """1 つの device に載ったメンバー一式。``models[i]`` は読み込んだ ``i`` 番目のメンバーのこの device 上のコピー。"""

    device: torch.device
    models: list[WindowModel]


def place_models(models: Sequence[WindowModel], devices: Sequence[str | torch.device]) -> list[Bank]:
    """CPU 上のメンバーを device ごとにコピーして載せる。

    最初の device には読み込んだインスタンスそのものを、2 つ目以降には CPU 上で deepcopy したものを載せる。
    """
    if not models:
        raise ValueError("no models given")
    if not devices:
        raise ValueError("no devices given")
    copies = [list(models)] + [[copy.deepcopy(model) for model in models] for _ in devices[1:]]  # deepcopy は CPU 上で
    banks: list[Bank] = []
    for device, bank_models in zip(devices, copies):
        device = canonical_device(device)
        for model in bank_models:
            model.to(device).eval()
        banks.append(Bank(device, bank_models))
        logger.info("%s: %d member(s) placed", device, len(bank_models))
    return banks


def _score(
    bank: Bank,
    group_members: Sequence[Sequence[int]],
    stacks: Sequence[tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, list[dict]]],
    tta_hflip: bool,
    counters: PhaseCounters,
    weights: Sequence[float] | None = None,
    parts: Sequence[int] | None = None,
) -> np.ndarray:
    """consumer thread 側の 1 study: メンバーの重み付き和 ``float64[12]``。足す順は束の順、束の中はメンバーの順。

    ``parts``（member ごとの part 番号）があれば part ごとに分けて足し、``float64[n_parts, 12]`` を返す。

    ``inference_mode`` は **この thread で** 開く。main thread のコンテキストマネージャは consumer に
    引き継がれず、無いと forward が autograd のために活性を保持して 166 窓の Bag が T4 で OOM する。
    """
    started = time.perf_counter()
    name = str(bank.device)
    n_models = len(bank.models)
    # 重みは member ごとのスカラか、member ごとの所見別ベクトル [12]。後者は predict_study の返り [12] にそのまま掛かる。
    share = [1.0 / n_models] * n_models if weights is None else [torch.as_tensor(w, dtype=torch.float64) if not isinstance(w, (int, float)) else float(w) for w in weights]
    total = np.zeros(NUM_LABELS if parts is None else (max(parts) + 1, NUM_LABELS), dtype=np.float64)
    with torch.inference_mode():
        for members, stack in zip(group_members, stacks):
            pixels, mask = stack[:2]
            headers = stack[2] if len(stack) >= 3 else None
            subseries = stack[3] if len(stack) == 4 else None
            group_total: torch.Tensor | None = None
            for i in members:
                extra = ({"series_headers": headers} if headers is not None else {}) | ({"subseries": subseries} if subseries is not None else {})
                probs = predict_study(bank.models[i], pixels, mask, device=bank.device, tta_hflip=tta_hflip, **extra).double() * share[i]
                if parts is not None:
                    total[parts[i]] += probs.numpy()
                    continue
                group_total = probs if group_total is None else group_total + probs
            if parts is None:
                assert group_total is not None
                total += group_total.numpy()
    counters.gpu_s[name] = counters.gpu_s.get(name, 0.0) + (time.perf_counter() - started)
    counters.n_scored[name] = counters.n_scored.get(name, 0) + 1
    return total


def score_stream(
    items: Iterable[tuple[int, Sequence[tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, list[dict]]]]],
    banks: Sequence[Bank],
    group_members: Sequence[Sequence[int]],
    tta_hflip: bool = False,
    counters: PhaseCounters | None = None,
    inflight: int | None = None,
    weights: Sequence[float] | None = None,
    parts: Sequence[int] | None = None,
) -> Iterator[tuple[int, np.ndarray | Exception]]:
    """デコード済みの study を device に round-robin で配り、``(index, probabilities)`` を投入順に返す。

    device ごとに consumer thread を 1 本持つ。分けるのは study 単位であって窓単位ではないので、study
    1 件の確率は 1 枚の GPU で直列に採点したものと bit 単位で同じになる。forward が失敗した study は
    ``(index, exception)`` で返る。``parts`` を渡すと確率は part ごとの ``[n_parts, 12]`` になる。
    """
    if not banks:
        raise ValueError("no banks given")
    counters = counters if counters is not None else PhaseCounters()
    inflight = inflight if inflight is not None else 2 * len(banks)
    if inflight < 1:
        raise ValueError(f"inflight must be >= 1, got {inflight}")
    executors = [ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"score{b}-{bank.device}") for b, bank in enumerate(banks)]
    pending: deque[tuple[int, Future[np.ndarray]]] = deque()

    def take() -> tuple[int, np.ndarray | Exception]:
        index, future = pending.popleft()
        try:
            return index, future.result()
        except Exception as exc:  # forward の失敗。呼び出し側に判断させる
            return index, exc

    try:
        for turn, (index, stacks) in enumerate(items):
            while len(pending) >= inflight:
                yield take()
            b = turn % len(banks)
            pending.append((index, executors[b].submit(_score, banks[b], group_members, stacks, tta_hflip, counters, weights, parts)))
        while pending:
            yield take()
    finally:
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=True)
