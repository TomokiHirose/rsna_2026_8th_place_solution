from __future__ import annotations

import copy
import math
from collections import deque
from typing import Any

import lightning as L
import torch
import torch.nn as nn

from rsna_knee.config import Config
from rsna_knee.constants import NUM_LABELS
from rsna_knee.data.depth_prior import finding_density_table
from rsna_knee.models import WindowModel
from rsna_knee.models.factory import build_model
from rsna_knee.models.multibranch import BRANCH_NAMES
from rsna_knee.models.window_model import PLANES
from rsna_knee.modules.labels import binarize
from rsna_knee.modules.losses import masked_bce_with_logits, pairwise_rank_loss, soft_fcl_loss, soft_lse_rank_loss, soft_ral_loss, window_bce_with_logits
from rsna_knee.modules.metrics import macro_auc, macro_spearman

__all__ = ["EMA_PREFIX", "MODEL_PREFIX", "LitWindowModel", "WeightEma", "build_scheduler", "export_weights_from_checkpoint"]

#: checkpoint の中で、包んだモデルの重みに Lightning が付ける接頭辞。
MODEL_PREFIX = "model."
#: checkpoint の中で、重みの EMA に付く接頭辞。
EMA_PREFIX = "ema.module."


def export_weights_from_checkpoint(state: dict[str, Any], prefix: str = MODEL_PREFIX) -> dict[str, Any]:
    """Lightning の ``state_dict`` から包んだモデルの重み（``prefix=EMA_PREFIX`` なら EMA）だけ取り出す。"""
    return {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}


class WeightEma(nn.Module):
    """学習中のモデルの重みの指数移動平均。buffer（BatchNorm の統計など）は学習中のものをそのまま写す。"""

    def __init__(self, model: nn.Module, decay: float) -> None:
        super().__init__()
        self.module = copy.deepcopy(model)
        self.decay = decay
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def set(self, model: nn.Module) -> None:
        """EMA を ``model`` の重みそのものにする。"""
        for ema, value in zip(self.module.state_dict().values(), model.state_dict().values()):
            ema.copy_(value)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """``ema = decay * ema + (1 - decay) * model``。buffer は写す。"""
        torch._foreach_lerp_(list(self.module.parameters()), [p.detach() for p in model.parameters()], 1.0 - self.decay)
        for ema, value in zip(self.module.buffers(), model.buffers()):
            ema.copy_(value)


def build_scheduler(optimizer: torch.optim.Optimizer, kind: str, max_lr: list[float], total_steps: int, pct_start: float) -> torch.optim.lr_scheduler.LRScheduler:
    """``interval="step"`` で進める学習率の曲線。``kind`` は "onecycle" か "cosine"（線形 warmup + cosine 減衰）。"""
    if kind == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=pct_start)
    if kind != "cosine":
        raise ValueError(f"unknown scheduler {kind!r}")
    warmup = max(int(total_steps * pct_start), 1)

    def curve(step: int) -> float:
        """0 -> 1 -> 0。warmup は線形、そのあとは cosine で 0 まで。"""
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, curve)


class LitWindowModel(L.LightningModule):
    """:class:`~rsna_knee.models.WindowModel` を学習する。

    AdamW でバックボーンを ``train.encoder_lr``、ヘッドを ``train.learning_rate`` に置き、学習率の曲線を
    run 全体に掛ける。分類項は ``train.cls_loss``（所見ごとの出現率の逆数を正例に掛ける）で、``train.rank_weight``
    が正ならそこへ ``train.rank_loss`` の順位項を足す。その比較相手には直近 ``train.rank_queue`` 個の学習 batch の
    logits（head ごとに ``[H, B, 12]``）・target・study ID を detach して FIFO で持ち回る。``model.kind="multibranch"`` は 3 head の損失を
    ``model.multibranch.loss_weights`` で足し、順位項の比較相手は head ごとに取る。``train.severity_weight`` が正なら、重症度 head の出力と
    ``batch["severity"]`` の soft BCE をその重みで足す。``train.window_loss_weight`` が正なら、窓 logit と深さ prior から作る窓ターゲットの BCE（:meth:`_window_loss`）を
    その重みで足す。``train.ema_decay`` があれば重みの EMA を持ち、warmup が
    終わるまでは学習中の重みを写し、その後は step ごとに追う。検証は EMA で行い、学習中の重みでの検証も ``val_raw/*`` に並べる。勾配クリップと precision は Trainer 側が持つ。

    Args:
        cfg: ルート config。
        pos_weight: ``[12]`` の正例重み。``None`` なら重み無しの BCE に落ちる。
    """

    def __init__(self, cfg: Config, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        # 補助 head を作るなら学習もする。推論は train の設定なしで Config(model=...) を組むので、この組み合わせは Config ではなく学習側で拒む
        if cfg.model.plane_aux and cfg.train.plane_aux_weight == 0.0:
            raise ValueError("model.plane_aux needs train.plane_aux_weight > 0: otherwise the auxiliary heads are never trained")
        self.save_hyperparameters({"cfg": cfg.model_dump(mode="json")})
        self.model: WindowModel = build_model(cfg)
        if cfg.train.grad_checkpointing and not self.model.set_grad_checkpointing(True):
            raise ValueError(f"train.grad_checkpointing is set but {cfg.model.backbone!r} does not support it")
        self.ema = None if cfg.train.ema_decay is None else WeightEma(self.model, cfg.train.ema_decay)
        # EMA を追い始める step（warmup の終わり）と、最後に EMA を進めた global_step。configure_optimizers が前者を決める。
        self._ema_start = 0
        self._ema_step = -1
        # buffer にしておけば module と一緒に GPU へ移り、これが形作った重みの隣で checkpoint に載る。
        self.register_buffer("pos_weight", None if pos_weight is None else pos_weight.float().clone())
        # 窓ターゲットの所見別 prior 密度 ``[2, Slot, 12, 格子]``（:func:`finding_density_table`）。prior の JSON から作り直せるので checkpoint には載せない。
        self.register_buffer("window_prior", torch.from_numpy(finding_density_table(cfg.data.depth_prior.path)) if cfg.train.window_loss_weight > 0.0 else None, persistent=False)
        # ランク損失の比較相手。maxlen 0 なら append しても何も残らないので、無効時は触らない。
        self._rank_queue: deque[tuple[torch.Tensor, torch.Tensor, list[str]]] = deque(maxlen=cfg.train.rank_queue)
        self._val_logits: list[torch.Tensor] = []
        # multibranch の head ごとの検証 logits ``[3, B, 12]``。``val/auc_macro_<head>`` に出す。
        self._val_branch_logits: list[torch.Tensor] = []
        # EMA を持つ run で、学習中の重み（EMA なし）でも検証した logits。``val_raw/*`` に出す。
        self._val_raw_logits: list[torch.Tensor] = []
        self._val_labels: list[torch.Tensor] = []
        self._val_severity_logits: list[torch.Tensor] = []
        self._val_severity: list[torch.Tensor] = []
        # 撮像面ごとの補助 head の検証 logits ``[B, 面, 12]`` と、面に窓があるか ``[B, 面]``。``val/plane_*`` に出す。
        self._val_plane_logits: list[torch.Tensor] = []
        self._val_plane_present: list[torch.Tensor] = []

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """ランク損失の比較相手を batch の順序ごと保存する。"""
        checkpoint["rank_queue"] = [(logits.detach().float().cpu().clone(), target.detach().float().cpu().clone(), list(ids)) for logits, target, ids in self._rank_queue]

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """ランク損失の比較相手を復元する。"""
        self._rank_queue = deque(((logits.detach().float(), target.detach().float(), list(ids)) for logits, target, ids in checkpoint["rank_queue"]), maxlen=self.cfg.train.rank_queue)

    def on_train_start(self) -> None:
        """復元した比較相手を学習デバイスへ移す。"""
        self._rank_queue = deque(((logits.to(self.device), target.to(self.device), ids) for logits, target, ids in self._rank_queue), maxlen=self.cfg.train.rank_queue)

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        """:func:`~rsna_knee.modules.datamodule.collate_batch` の batch -> logits ``[B, 12]``。"""
        return self.model.forward_batch(batch)

    def _heads(self, batch: dict[str, Any], model: nn.Module) -> tuple[torch.Tensor, torch.Tensor | None]:
        """batch -> ``(logits, severity_logits)``。補助損失が無効なら重症度 head は走らせず ``None``。"""
        if self.cfg.train.severity_weight == 0.0:
            return model.forward_batch(batch), None
        return model.forward_heads(batch["windows"], batch.get("window_mask"), batch.get("window_slot"), batch.get("window_pos"), batch.get("window_rows"))

    def _plane_bce(self, plane_logits: torch.Tensor, present: torch.Tensor, batch: dict[str, Any], stage: str) -> torch.Tensor:
        """撮像面ごとの補助 head の分類項を、面に窓がある study だけで面ごとに求めて ``{stage}/plane_bce/<面>`` に log し、窓のある面で平均したもの。"""
        target, weight = batch["label"], batch.get("weight")
        losses = []
        for index, name in enumerate(PLANES):
            rows = present[:, index]
            if not bool(rows.any()):
                continue
            loss = self._cls_loss(plane_logits[rows, index], target[rows], None if weight is None else weight[rows])
            self.log(f"{stage}/plane_bce/{name}", loss, on_step=False, on_epoch=True, batch_size=int(rows.sum()))
            losses.append(loss)
        return torch.stack(losses).mean() if losses else plane_logits.sum() * 0.0

    def _severity_loss(self, severity_logits: torch.Tensor, batch: dict[str, Any], stage: str) -> torch.Tensor:
        """重症度 head の soft BCE に ``train.severity_weight`` を掛けたもの。``{stage}/severity`` には掛ける前の値を log する。"""
        if "severity" not in batch:
            raise ValueError("train.severity_weight > 0 but the batch has no severity: set data.severity_csv")
        severity = masked_bce_with_logits(severity_logits, batch["severity"])
        self.log(f"{stage}/severity", severity, on_step=stage == "train", on_epoch=True, batch_size=severity_logits.shape[0])
        return self.cfg.train.severity_weight * severity

    def _window_loss(self, window_logits: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
        """窓 logit ``[B, K, 12]`` と窓ターゲット ``y_i * ghat(Slot, u)`` の BCE に ``train.window_loss_weight`` を掛けたもの。``train/win`` には掛ける前の値を log する。

        ghat は ``batch["window_u"]`` を prior の格子の最近傍に引き、左右対称化する窓（``batch["window_u_sym"]``）は対称化した密度を読む。
        """
        n_grid = self.window_prior.shape[-1]
        cell = ((batch["window_u"].clamp(-1.0, 1.0) + 1.0) * ((n_grid - 1) / 2)).round().long()
        findings = torch.arange(NUM_LABELS, device=cell.device)
        prior = self.window_prior[batch["window_u_sym"].long()[..., None], batch["window_slot"][..., None], findings, cell[..., None]]
        win = window_bce_with_logits(window_logits, batch["label"], prior, batch["window_mask"], pos_weight=self.pos_weight)
        self.log("train/win", win, on_step=False, on_epoch=True, batch_size=window_logits.shape[0])
        return self.cfg.train.window_loss_weight * win

    def _rank_ref(self, head: int = 0) -> tuple[torch.Tensor | None, torch.Tensor | None, list[str] | None]:
        """queue に溜まった直近の batch の head ``head`` の logits と target を ``[M, 12]`` 2 本、study ID を ``M`` 個にまとめる（空なら ``None`` 3 つ）。"""
        if not self._rank_queue:
            return None, None, None
        return torch.cat([logits[head] for logits, _, _ in self._rank_queue]), torch.cat([target for _, target, _ in self._rank_queue]), [name for _, _, ids in self._rank_queue for name in ids]

    def _cls_loss(self, logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
        """``train.cls_loss`` の分類項。どれも正例重みとセル重み・NaN mask の扱いは共通。"""
        train = self.cfg.train
        if train.cls_loss == "soft_ral":
            return soft_ral_loss(logits, target, pos_weight=self.pos_weight, weight=weight)
        if train.cls_loss == "soft_fcl":
            return soft_fcl_loss(logits, target, train.fcl_gamma, train.fcl_beta, pos_weight=self.pos_weight, weight=weight)
        return masked_bce_with_logits(logits, target, pos_weight=self.pos_weight, weight=weight)

    def _rank(self, logits: torch.Tensor, target: torch.Tensor, ids: list[str], ref: tuple[torch.Tensor | None, torch.Tensor | None, list[str] | None]) -> torch.Tensor:
        """``train.rank_loss`` の順位項。"""
        if self.cfg.train.rank_loss == "soft_lse":
            return soft_lse_rank_loss(logits, target, ids, *ref, temperature=self.cfg.train.rank_temperature)
        return pairwise_rank_loss(logits, target, ref[0], ref[1])

    def _loss(self, logits: torch.Tensor, batch: dict[str, Any], stage: str, ref: tuple[torch.Tensor | None, torch.Tensor | None, list[str] | None] = (None, None, None), weight: torch.Tensor | None = None) -> torch.Tensor:
        """``cls + rank_weight * rank``。``rank_weight`` が 0 なら順位項は計算せず分類項そのもの。

        ``weight`` ``[B, 12]`` は分類項のセルごとの係数で、順位項には掛からない。内訳は ``rank_weight`` が
        正のときだけ ``{stage}/bce``（選んだ分類項の値）と ``{stage}/rank`` に log する。
        """
        target = batch["label"]
        cls = self._cls_loss(logits, target, weight)
        rank_weight = self.cfg.train.rank_weight
        if rank_weight == 0.0:
            return cls
        rank = self._rank(logits, target, batch["study_id"], ref)
        on_step = stage == "train"
        self.log(f"{stage}/bce", cls, on_step=on_step, on_epoch=True, batch_size=logits.shape[0])
        self.log(f"{stage}/rank", rank, on_step=on_step, on_epoch=True, batch_size=logits.shape[0])
        return cls + rank_weight * rank

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        if self.cfg.model.kind == "multibranch":
            logits_all = self.model.forward_batch_branches(batch)
            loss = logits_all.new_zeros(())
            for i, (name, lam) in enumerate(zip(BRANCH_NAMES, self.cfg.model.multibranch.loss_weights)):
                head_loss = self._loss(logits_all[i], batch, "train", self._rank_ref(i), batch.get("weight"))
                self.log(f"train/loss_{name}", head_loss, on_step=False, on_epoch=True, batch_size=logits_all.shape[1])
                loss = loss + lam * head_loss
        elif self.cfg.train.window_loss_weight > 0.0:
            logits, window_logits = self.model.forward_batch_windows(batch)
            logits_all = logits[None]
            loss = self._loss(logits, batch, "train", self._rank_ref(), batch.get("weight")) + self._window_loss(window_logits, batch)
        elif self.cfg.model.plane_aux:
            logits, plane_logits, present = self.model.forward_plane_aux(batch)
            logits_all = logits[None]
            aux = self._plane_bce(plane_logits, present, batch, "train")
            self.log("train/plane_aux", aux, on_step=False, on_epoch=True, batch_size=logits.shape[0])
            loss = self._loss(logits, batch, "train", self._rank_ref(), batch.get("weight")) + self.cfg.train.plane_aux_weight * aux
        else:
            logits, severity_logits = self._heads(batch, self.model)
            logits_all = logits[None]
            loss = self._loss(logits, batch, "train", self._rank_ref(), batch.get("weight"))
            if severity_logits is not None:
                loss = loss + self._severity_loss(severity_logits, batch, "train")
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=logits_all.shape[1])
        if self._rank_queue.maxlen:
            self._rank_queue.append((logits_all.detach().float(), batch["label"].detach().float(), list(batch["study_id"])))
        return loss

    def on_train_batch_end(self, outputs: Any, batch: dict[str, Any], batch_idx: int) -> None:
        """optimizer が進んだ step ごとに EMA を進める。warmup が終わるまでは学習中の重みをそのまま写す。"""
        if self.ema is None or self.trainer.global_step == self._ema_step:
            return
        self._ema_step = self.trainer.global_step
        if self._ema_step <= self._ema_start:
            self.ema.set(self.model)
        else:
            self.ema.update(self.model)

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """クリップ前の全勾配の L2 ノルムを ``train/grad_norm`` に log する。"""
        norms = [p.grad.detach().norm() for p in self.parameters() if p.grad is not None]
        self.log("train/grad_norm", torch.linalg.vector_norm(torch.stack(norms)), on_step=True, on_epoch=False)

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        model = self.model if self.ema is None else self.ema.module
        if self.cfg.model.kind == "multibranch":
            # 損失と checkpoint 選択の AUC は混合後の logits で取り、head ごとの AUC は並べて見るためだけに出す。
            logits_all = model.forward_batch_branches(batch)
            self._val_branch_logits.append(logits_all.detach().float().cpu())
            logits, severity_logits = model.mix(logits_all), None
        elif self.cfg.model.plane_aux:
            # 主の logits は通常の経路と同じ。補助 head の分類項と AUC は記録するだけで、検証の損失と checkpoint の選択には入れない。
            logits, plane_logits, present = model.forward_plane_aux(batch)
            severity_logits = None
            self._plane_bce(plane_logits, present, batch, "val")
            self._val_plane_logits.append(plane_logits.detach().float().cpu())
            self._val_plane_present.append(present.detach().cpu())
        else:
            logits, severity_logits = self._heads(batch, model)
        # 検証では queue を混ぜない。epoch 間で比較できる数字にしたいので、対は batch 内だけで取る。
        loss = self._loss(logits, batch, "val")
        if severity_logits is not None:
            loss = loss + self._severity_loss(severity_logits, batch, "val")
            self._val_severity_logits.append(severity_logits.detach().float().cpu())
            self._val_severity.append(batch["severity"].detach().float().cpu())
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=logits.shape[0])
        self._val_logits.append(logits.detach().float().cpu())
        if self.ema is not None:
            self._val_raw_logits.append(self.model.forward_batch(batch).detach().float().cpu())
        self._val_labels.append(batch["label"].detach().float().cpu())
        return loss

    def on_validation_epoch_start(self) -> None:
        self._val_logits.clear()
        self._val_branch_logits.clear()
        self._val_raw_logits.clear()
        self._val_labels.clear()
        self._val_severity_logits.clear()
        self._val_severity.clear()
        self._val_plane_logits.clear()
        self._val_plane_present.clear()

    def on_validation_epoch_end(self) -> None:
        """検証 epoch 全体に対する macro / ラベル別の ROC AUC と、重症度 head の所見ごとの Spearman の macro 平均。"""
        if not self._val_logits:
            return
        probs = torch.sigmoid(torch.cat(self._val_logits)).numpy()
        labels = binarize(torch.cat(self._val_labels).numpy())
        macro, per_label = macro_auc(labels, probs)
        self.log("val/auc_macro", float(macro), prog_bar=True, sync_dist=True)
        for name, score in per_label.items():
            self.log(f"val/auc/{name}", float(score), sync_dist=True)
        if self._val_branch_logits:
            branch_logits = torch.cat(self._val_branch_logits, dim=1)
            for name, head_logits in zip(BRANCH_NAMES, branch_logits):
                self.log(f"val/auc_macro_{name}", float(macro_auc(labels, torch.sigmoid(head_logits).numpy())[0]), sync_dist=True)
        if self._val_raw_logits:
            # checkpoint の選択は EMA の val/auc_macro のまま。学習中の重みの値は並べて見るためだけに出す。
            raw_macro, raw_per_label = macro_auc(labels, torch.sigmoid(torch.cat(self._val_raw_logits)).numpy())
            self.log("val_raw/auc_macro", float(raw_macro), sync_dist=True)
            for name, score in raw_per_label.items():
                self.log(f"val_raw/auc/{name}", float(score), sync_dist=True)
        if self._val_severity_logits:
            spearman, _ = macro_spearman(torch.cat(self._val_severity).numpy(), torch.sigmoid(torch.cat(self._val_severity_logits)).numpy())
            self.log("val/severity_spearman", float(spearman), sync_dist=True)
        if self._val_plane_logits:
            # 撮像面ごとの補助 head の AUC。その面に窓がある study だけで採点する。
            plane_probs = torch.sigmoid(torch.cat(self._val_plane_logits)).numpy()
            plane_present = torch.cat(self._val_plane_present).numpy()
            for index, name in enumerate(PLANES):
                rows = plane_present[:, index]
                if not rows.any():
                    continue
                plane_macro, plane_per_label = macro_auc(labels[rows], plane_probs[rows, index])
                self.log(f"val/plane_auc_macro/{name}", float(plane_macro), sync_dist=True)
                for label, score in plane_per_label.items():
                    self.log(f"val/plane_auc/{name}/{label}", float(score), sync_dist=True)
        self._val_logits.clear()
        self._val_branch_logits.clear()
        self._val_raw_logits.clear()
        self._val_labels.clear()
        self._val_severity_logits.clear()
        self._val_severity.clear()
        self._val_plane_logits.clear()
        self._val_plane_present.clear()

    def export_state_dict(self, ema: bool = False) -> dict[str, torch.Tensor]:
        """``model.load_state_dict`` にそのまま渡せる CPU の state dict。``ema`` なら EMA の重み（EMA が無効なら例外）。"""
        if ema and self.ema is None:
            raise ValueError("export_state_dict(ema=True) needs train.ema_decay")
        model = self.ema.module if ema else self.model
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def configure_optimizers(self) -> dict[str, Any]:
        """AdamW（``model.backbone`` は ``encoder_lr``、それ以外のヘッドは ``learning_rate``）と学習率の曲線。"""
        if hasattr(self.model, "optimizer_groups"):
            encoder_lr = self.cfg.train.encoder_lr if self.cfg.train.encoder_lrs is None else self.cfg.train.encoder_lrs
            groups = self.model.optimizer_groups(encoder_lr, self.cfg.train.learning_rate)
        else:
            backbone_ids = {id(p) for p in self.model.backbone.parameters()}
            backbone = [p for p in self.model.backbone.parameters() if p.requires_grad]
            head = [p for p in self.model.parameters() if id(p) not in backbone_ids and p.requires_grad]
            groups = ([{"params": backbone, "lr": self.cfg.train.encoder_lr}] if backbone else []) + [{"params": head, "lr": self.cfg.train.learning_rate}]
        max_lr = [group["lr"] for group in groups]

        optimizer = torch.optim.AdamW(groups, lr=self.cfg.train.learning_rate, weight_decay=self.cfg.train.weight_decay)
        total_steps = max(int(self.trainer.estimated_stepping_batches), 1)
        scheduler = build_scheduler(optimizer, self.cfg.train.scheduler, max_lr, total_steps, self.cfg.train.pct_start)
        self._ema_start = max(int(total_steps * self.cfg.train.pct_start), 1)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
