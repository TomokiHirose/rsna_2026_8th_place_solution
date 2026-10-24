import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from rsna_knee.constants import LABELS
from rsna_knee.utils import get_logger

logger = get_logger(__name__)


def macro_auc(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, dict[str, float]]:
    """``[N, 12]`` の target と予測確率から ``(macro, per_label)`` の ROC AUC を求める。

    有効な target が 1 クラスしか無いラベルは定義できないので ``NaN`` にし、macro 平均から落とす。
    採点できるラベルが 1 つも無ければ ``macro`` も ``NaN``。
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    per_label: dict[str, float] = {}
    for i, name in enumerate(LABELS):
        valid = ~np.isnan(y_true[:, i])
        target = y_true[valid, i]
        per_label[name] = float(roc_auc_score(target, y_prob[valid, i])) if np.unique(target).size > 1 else float("nan")

    scores = np.array(list(per_label.values()), dtype=np.float64)
    excluded = int(np.isnan(scores).sum())
    if excluded:
        logger.info("macro_auc: excluded %d/%d labels with a single class present", excluded, len(LABELS))
    macro = float("nan") if excluded == len(LABELS) else float(np.nanmean(scores))
    return macro, per_label


def macro_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, dict[str, float]]:
    """``[N, 12]`` の target と予測から ``(macro, per_label)`` の Spearman 順位相関を求める。

    ``NaN`` の target は落とす。有効な対が 2 つ未満か、どちらかが定数のラベルは定義できないので ``NaN`` にし、
    macro 平均から落とす。採点できるラベルが 1 つも無ければ ``macro`` も ``NaN``。
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    per_label: dict[str, float] = {}
    for i, name in enumerate(LABELS):
        valid = ~np.isnan(y_true[:, i])
        target, pred = y_true[valid, i], y_pred[valid, i]
        defined = target.size >= 2 and np.unique(target).size > 1 and np.unique(pred).size > 1
        per_label[name] = float(spearmanr(target, pred).statistic) if defined else float("nan")

    scores = np.array(list(per_label.values()), dtype=np.float64)
    macro = float("nan") if np.isnan(scores).all() else float(np.nanmean(scores))
    return macro, per_label
