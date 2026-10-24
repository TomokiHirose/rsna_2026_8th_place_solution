from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AugConfig(BaseModel):
    """学習時の augmentation。既定は明度ゲイン・Study 単位の鏡像と、Slot ごとの回転・拡縮・並進・低解像度化・ぼかし・ノイズ。

    明度は乗算のゲインで、study ごとに 1 度引いて bag のすべての window で共有する。``translate_px`` と
    ``scale_range`` は study の Stack 全体に同じ変換を掛けてから window を切る。``mirror_prob`` 以降は
    window を切った後に掛ける（``rsna_knee.modules.window_aug``）: 鏡像 -> Slot ごとの幾何・画質 -> 明度の順。
    """

    brightness_gain: float = Field(default=0.10, ge=0.0)
    # dy と dx をそれぞれ [-translate_px, translate_px] から一様に引く。
    translate_px: int = Field(default=0, ge=0)
    # 拡縮率の範囲 (lo, hi)。None で無効。
    scale_range: tuple[float, float] | None = None
    scale_prob: float = Field(default=0.5, ge=0.0, le=1.0)
    depth_flip: float = Field(default=0.0, ge=0.0, le=1.0)
    # 膝を反対側の膝に見せる鏡像を Study 単位で 1 回抽選する。ラベルは変えない（内側/外側は解剖学的な呼び名）。
    mirror_prob: float = Field(default=0.5, ge=0.0, le=1.0)
    # 回転（度）・拡縮・並進（画像幅に対する割合）を Slot ごとに 1 組引く。はみ出しは 0 で埋める。
    affine_prob: float = Field(default=0.5, ge=0.0, le=1.0)
    rotate_deg: float = Field(default=15.0, ge=0.0)
    affine_scale: tuple[float, float] = (0.9, 1.1)
    affine_translate: float = Field(default=0.05, ge=0.0, le=0.5)
    # 画像幅の x 倍へ縮めて元の大きさへ戻す（低解像度の撮像の模擬）。
    lowres_prob: float = Field(default=0.25, ge=0.0, le=1.0)
    lowres_range: tuple[float, float] = (0.5, 1.0)
    blur_prob: float = Field(default=0.2, ge=0.0, le=1.0)
    blur_sigma: tuple[float, float] = (0.5, 1.0)
    # 加法ガウスノイズ。標準偏差を [0, noise_std] から引く（値域 [0, 1] の画素に対して）。
    noise_prob: float = Field(default=0.1, ge=0.0, le=1.0)
    noise_std: float = Field(default=0.03, ge=0.0)

    def model_post_init(self, context: object, /) -> None:
        """範囲の組が ``0 < lo <= hi`` を満たすか検める。"""
        if self.scale_range is not None and not (0.0 < self.scale_range[0] <= self.scale_range[1]):
            raise ValueError(f"aug.scale_range must satisfy 0 < lo <= hi, got {self.scale_range}")
        for name, (lo, hi) in {"affine_scale": self.affine_scale, "lowres_range": self.lowres_range, "blur_sigma": self.blur_sigma}.items():
            if not 0.0 < lo <= hi:
                raise ValueError(f"aug.{name} must satisfy 0 < lo <= hi, got {(lo, hi)}")
        if self.lowres_range[1] > 1.0:
            raise ValueError(f"aug.lowres_range must not exceed 1, got {self.lowres_range}")


class DepthPriorConfig(BaseModel):
    """学習時に Slot 内の窓を深さ u の prior で重み付けして採る設定（:mod:`rsna_knee.data.depth_prior`）。"""

    model_config = ConfigDict(extra="forbid")
    # Slot ごとの混合ガウス prior の JSON（``tools/fit_depth_prior.py`` の出力）。
    path: Path
    # 窓の採取重みは ``(1 - alpha) / n + alpha * q_prior``。0 なら一様（現行の採り方と同じ分布）、1 なら prior だけ。
    alpha: float = Field(ge=0.0, le=1.0)


class UnionBagConfig(BaseModel):
    """union bag（全 series × 全版の窓の Bag）の組み方。cache の画素で Bag を組み、オンラインで学習する。

    版は series の元の 1 本と、その series の sub-series 全部（番号順）で、実在スライスの無い版は除く。窓は版の中だけで切り、Slot は series 自身の撮像面と液体強調で決める。
    学習は Slot ごとに series の本数を 1〜候補数から一様に引いて無作為に選び、選んだ series ごとに版の数を 1〜版数から一様に引いて選ぶ。選んだ版の全窓から
    ``k_min``〜``k_max`` の一様な数を重複なしで採る（窓が少なければ全部）。検証・推論は全版の全窓で、``eval_cap`` を超える study だけ Slot ごとの窓数の比を保って固定乱数で採る。
    """

    model_config = ConfigDict(extra="forbid")
    k_min: int = Field(default=32, ge=1)
    k_max: int = Field(default=352, ge=1)
    eval_cap: int = Field(default=384, ge=1)

    @model_validator(mode="after")
    def _check_k(self) -> "UnionBagConfig":
        if self.k_min > self.k_max:
            raise ValueError(f"union_bag.k_min ({self.k_min}) must be <= k_max ({self.k_max})")
        return self


class DataConfig(BaseModel):
    """データセットとデータ読み込みの設定。"""

    # 公式の train.csv。gold の 0/1 ラベルを持つのは ~58 study だけ。
    train_csv: Path = Path("input/rsna-knee-abnormality-detection/train.csv")
    # レポートしか無い ~4,400 study 向けの学習ラベル。テキスト 2 源の平均を 0.25、画像側（福嶋画像と較正 OOF の平均、画像ラベルが無ければ較正 OOF）を 0.75 で混ぜる。
    labels_csv: Path = Path("data/label/image_weight/v1/labels_soft.csv")
    # セルごとの損失の重み。表に無い study と空セルは 1。
    weights_csv: Path | None = None
    # 所見ごとの重症度（long 形式、``severity_soft``）。``model.severity_head`` の補助損失の target で、表に無い study は NaN。
    severity_csv: Path | None = None
    folds_csv: Path = Path("data/folds_v1.csv")
    # 「レポートがこの finding に触れていない」を表す番兵（``rsna_knee.modules.labels`` を見よ）。
    unknown_value: float = 0.05
    # 番兵のセルを NaN に置き換え、masked loss がそこを飛ばす。
    unknown_to_nan: bool = False
    batch_size: int = Field(default=8, gt=0)
    num_workers: int = 8
    pin_memory: bool = True
    # "folds" は scanner でグループ分けした CV、"gold" は gold ラベルを持つ ~58 study を検証に回す
    # protocol、"all" は検証を持たない。この 3 つの数字は比較できない。
    split: Literal["folds", "gold", "all"] = "gold"
    # CV の fold を 1 つ学習 split から外して OOF を採点する。``split="gold"`` のときだけ意味がある。
    holdout_fold: int | None = None
    # OOF を採点する相手のラベル。学習に使ったラベルと物差しを分けたいときに指定する。
    eval_labels_csv: Path | None = None
    # ``tools/build_cache.py`` が書いた study cache。
    cache_dir: Path | None = Path("data/cache/cache_native_384_series_all_exact_uniform_sub")
    # 1 study に両膝の series が入っている検査を、膝ごとの子 study に分けて学習する表。``data/label/ayfukushima/bilateral_knees/knee_series.csv``
    # をそのまま渡せる（``child_study_id`` を子の ID、``StudyInstanceUID`` を親とみなし、``knee`` が left / right の行だけ使う）。
    # ``StudyInstanceUID``（子の ID）・``ParentStudyInstanceUID``・``SeriesInstanceUID`` を持つ表でも良い。子は親の cache 行のうち自分の
    # series だけを読み、表に出た親は学習・検証から外れる。ラベルは子の ID で引く（``exp118_image/v7_knees`` など）。
    # ``series_layout == "all"`` の cache でだけ使える。
    subcases_csv: Path | None = None
    aug: AugConfig = AugConfig()
    # 学習 split の窓の採り方を深さ prior で寄せる。None なら Slot を区別せず一様。series_layout "all" の cache と ``window.train_runs="window"`` だけ。
    depth_prior: DepthPriorConfig | None = None
    # 学習時、Slot に選んだ series が sub-series（``build_cache.py --subseries``）を持てば、元の 1 本と m 本の sub-series の m + 1 通りから
    # 均等に 1 つを選んで画素を差し替える。series の選び方は変えない（2 段階）。評価・推論は元の 1 本のまま。
    subseries: bool = False
    # union bag の組み方。cache の画素で Bag を組んでオンラインで学習する
    # （``series_layout="all"`` + sub-series の cache・``depth_prior``・``window.train_runs="window"`` が要り、``subseries`` とは併用しない）。
    union_bag: UnionBagConfig | None = None


class PreprocessConfig(BaseModel):
    """モデルが fitted された画素が DICOM からどう作られたか。

    :class:`rsna_knee.data.dicom.PreprocessParams` とフィールドが 1 対 1 で対応する（``slot_table`` は
    あちらの ``caps``）。重みの隣を旅して、提出時に同じ画素を再現する。
    """

    crop_mm: float = Field(gt=0.0)
    size: int = Field(gt=0)
    span: tuple[float, float]
    percentiles: tuple[float, float]
    # 5 つの解剖学的な Slot それぞれから取るスライス数を Slot の順に並べたもの。Stack のレイアウトの正本。
    slot_table: list[int]
    # 短い Series は水増しせず、上限を超える Series の採り方は long_series_sampling に従う。
    native_spacing: bool = False
    default_pixel_spacing: float = 0.5
    # "area" は整数の中心ボックス + cv2.INTER_AREA、"exact" は float の中心ボックス + PIL bicubic。
    resample: Literal["area", "exact"] = "area"
    # native の上限を超える Series は、中央の連続区間（contiguous、両端を捨てる）か span 全体の等間隔（uniform）から採る。
    long_series_sampling: Literal["contiguous", "uniform"] = "contiguous"
    # crop_mm のボックスが画像からはみ出す分を 0 で埋めて 1 画素あたりの mm を保つか。キーの無い重み・cache は False（はみ出す series を拡大）で焼かれている。
    crop_pad: bool = False
    # Slot ごとの標準ランドマーク。None なら ``percentiles`` の輝度窓、あればヒストグラム標準化。
    histogram_standard: list[list[float]] | None = None
    # クロップの中心を患者の前方へずらす mm（cache の manifest を写す）。
    anterior_shift_mm: float = 0.0


class WindowConfig(BaseModel):
    """Slot 内の Window の枚数・チャネル数・正規化。評価の ``k_eval=None`` は全実在窓を一度ずつ使う。"""

    k_train: int = Field(default=12, gt=0)
    k_val: int = Field(default=24, gt=0)
    k_eval: int | None = Field(default=None, gt=0)
    # 学習中に backbone へ 1 度に通す窓の数。``None`` なら batch の全窓を一括で通す。backbone は窓同士を混ぜないので勾配は変わらず、変わるのはピークメモリと BatchNorm の統計を取る窓数だけ。
    train_chunk: int | None = Field(default=None, gt=0)
    # 1 窓に積む連続スライスの枚数 = backbone の in_chans。奇数。3 以外では timm が conv1 の重みを詰め直す。
    context: int = Field(default=3, ge=1)
    # "imagenet" は RGB の 3 組なので context 3 専用、"mri" は 1 組を全チャネルに、"none" は 0 / 1、"half" は 0.5 / 0.5。
    norm: Literal["none", "imagenet", "mri", "half"] = "imagenet"
    drop: float = Field(default=0.2, ge=0.0, lt=1.0)
    # 学習時の K 窓の採り方。"window" は窓ごとに無作為、"slot" は窓を持つ Slot ごとに 1 本の連続区間（中心スライスが 1 ずつ並ぶ窓）。
    train_runs: Literal["window", "slot"] = "window"
    # 窓の集約。"attention" は所見別 attention-MIL、"lse" は窓 logit の log-sum-exp（tau→0 で max、tau→∞ で mean）。
    pool: Literal["attention", "lse"] = "attention"
    # "lse" の温度 tau。
    pool_tau: float = Field(default=1.0, gt=0.0)

    def model_post_init(self, context: object, /) -> None:
        """``context`` が奇数であることを検める。"""
        if self.context % 2 == 0:
            raise ValueError(f"window.context must be odd, got {self.context}")


class VlmConfig(BaseModel):
    """動画入力、固定指示、言語 LoRA の設定。"""

    model_config = ConfigDict(extra="forbid")
    source: Path
    prompt: str
    pseudo_fps: float = Field(default=2.0, gt=0)
    lora_rank: int = Field(default=16, gt=0)
    lora_alpha: int = Field(default=32, gt=0)
    lora_dropout: float = Field(default=0.05, ge=0, lt=1)
    lora_lr: float = Field(default=1e-4, gt=0)
    merger_lr: float = Field(default=1e-5, gt=0)


class FsxConfig(BaseModel):
    """所見 query が Slot 特徴に cross-attention し、FiLM で所見特徴を変調して logits に上限付きの差分を足す head。"""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    # FiLM logits と基本 logits の差を ``cap * tanh(delta / cap)`` で抑える上限。学習中と評価中で分ける。
    cap_train: float = Field(default=0.5, gt=0.0)
    cap_eval: float = Field(default=0.3, gt=0.0)


class SeqConfig(BaseModel):
    """同じ Slot で隣り合う窓の特徴を近傍 Conv で混ぜ、LayerNorm 後の窓特徴に残差として足す枝。出力層はゼロ初期化。"""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    width: int = Field(default=256, gt=0)
    layers: int = Field(default=2, ge=1)


class DepthConfig(BaseModel):
    """窓の 3 スライスを奥行き方向に混ぜる並列の枝。位置の attention で集約した特徴を窓特徴に連結して head に渡す。"""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    # 奥行き encoder。"conv_depth" は深さ方向だけの Conv3d、"conv3d" は 3x3x3 の Conv3d、"lstm" は画素ごとの双方向 LSTM。
    kind: Literal["conv_depth", "conv3d", "lstm"] = "conv_depth"
    # 共有 stem と奥行き encoder のチャネル数。
    width: int = Field(default=256, gt=0)
    # 奥行き encoder の中央面に backbone の最後から 2 段目の特徴マップを連結し、3x3 Conv で混ぜてから位置の attention に渡す。conv 系の kind だけ。
    fusion: bool = False
    # 共有 stem の出力の縮小率。32 なら 384 px で 12x12、16 なら最後の段で縮めず 24x24。
    stride: Literal[16, 32] = 32

    def model_post_init(self, context: object, /) -> None:
        """``lstm`` は前向き・後ろ向きを結合して ``width`` に戻すので ``width`` が偶数であること、``fusion`` は conv 系の kind であることを検める。"""
        if self.kind == "lstm" and self.width % 2:
            raise ValueError(f"model.depth.kind=lstm needs an even width, got {self.width}")
        if self.fusion and self.kind == "lstm":
            raise ValueError("model.depth.fusion needs a conv kind (conv_depth or conv3d), got lstm")


class PairConfig(BaseModel):
    """depth 枝の stem 3 段目の画像別特徴から、窓内の 2 画像の局所対応つき soft AND を作り、半月板 2 所見の logits に足す枝。"""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    # 対応候補を探す近傍の一辺（画素）。奇数。
    neighborhood: int = Field(default=5, ge=1)
    # 局所 cross-attention の q / k の次元。
    qk_dim: int = Field(default=32, gt=0)
    # 所見マップ A の head の中間チャネル数。
    hidden: int = Field(default=64, gt=0)
    # LSE 平均の温度の初期値（所見ごと・段ごとに学習する）。
    tau_init: float = Field(default=10.0, gt=0.0)
    # logits に足す係数 w の初期値。
    w_init: float = 0.1

    def model_post_init(self, context: object, /) -> None:
        """近傍が中心を持つよう ``neighborhood`` が奇数であることを検める。"""
        if self.neighborhood % 2 == 0:
            raise ValueError(f"model.pair.neighborhood must be odd, got {self.neighborhood}")


class SpatialConfig(BaseModel):
    """最終特徴マップから GAP とは別に、所見 x Slot の query で位置を重み付けして平均した所見ごとの窓特徴を作る枝。

    窓方向の attention-MIL は GAP の枝とは別に持ち、所見ごとに 2 本の集約特徴を連結して読む。query と読み出しの枝側は 0 から始まるので、学習開始時は GAP だけのモデルと同じ。
    """

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    # 空間 attention の key / query の次元。
    dim: int = Field(default=128, gt=0)
    # 枝の窓方向の attention の隠れ次元（所見で共有する tanh 層の幅）。
    att_hidden: int = Field(default=256, gt=0)


class ContextConfig(BaseModel):
    """実在窓への self-attention 1 層で作った窓間の文脈で条件付けた残差を所見ごとの窓特徴に足し、同じ ``clsW`` で窓 logit を読む枝。出力層はゼロ初期化。"""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    # GAP 側の窓特徴を射影する文脈の次元。
    dim: int = Field(default=256, gt=0)
    # 残差 MLP の隠れ次元。
    hidden: int = Field(default=256, gt=0)
    # self-attention の head 数。``dim`` を割り切ること。
    heads: int = Field(default=4, gt=0)

    def model_post_init(self, context: object, /) -> None:
        """``dim`` が ``heads`` で割り切れることを検める。"""
        if self.dim % self.heads:
            raise ValueError(f"model.context.dim {self.dim} must be divisible by heads {self.heads}")


class SeriesNeckConfig(BaseModel):
    """全 series の Bag の窓を series ごとに深さの順で双方向 GRU に通し、series の中を所見ごとの attention で、series の間を所見ごとの attention で集約する neck。

    GRU の入力は LayerNorm 後の GAP 特徴に、深さ u・前後の窓との u の差・Slot の one-hot の MLP を足したもの。空間枝の所見ごとの特徴は GRU を通さず series の中で集約して連結する。
    学習中は series ごと・所見ごとにスコア上位 ``topk_frac`` の窓を確率 ``topk_drop`` で集約から外す。
    """

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    # GRU の片方向の幅（出力はその 2 倍）。
    gru_hidden: int = Field(default=256, gt=0)
    # 深さと Slot の埋め込み MLP の隠れ幅。
    pos_hidden: int = Field(default=64, gt=0)
    # GRU の入力と出力に掛ける dropout。
    drop: float = Field(default=0.2, ge=0.0, lt=1.0)
    # series の中と series の間の attention の tanh 層の幅。
    att_hidden: int = Field(default=256, gt=0)
    # 学習中に外す候補（series の中のスコア上位の窓）の割合と、候補ごとに外す確率。
    topk_frac: float = Field(default=0.1, ge=0.0, le=1.0)
    topk_drop: float = Field(default=0.3, ge=0.0, lt=1.0)


class MultiBranchConfig(BaseModel):
    """CNN と Transformer の 2 encoder それぞれの GAP / SAP 窓特徴を、枝 0・枝 1・両者の連結（fusion）の 3 組の MIL + head で読み、確率を混ぜるモデル。"""

    model_config = ConfigDict(extra="forbid")
    # 枝 0（CNN）と枝 1（Transformer）の timm backbone。
    backbones: tuple[str, str] = ("convnextv2_tiny.fcmae_ft_in22k_in1k_384", "maxvit_tiny_tf_384.in1k")
    # 推論で sigmoid 確率を混ぜる重み（枝 0、枝 1、fusion）。
    blend: tuple[float, float, float] = (0.3, 0.3, 0.4)
    # 学習損失の head 別の係数（枝 0、枝 1、fusion）。
    loss_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    # 空間 attention の key / query の次元。
    spatial_dim: int = Field(default=128, gt=0)
    # SAP 側の窓方向の attention の隠れ次元。
    att_hidden: int = Field(default=256, gt=0)
    # fusion の入力（2 枝の連結）を窓単位で非線形に D 次元へ射影する（参考論文の concat → MLP）。None なら連結のまま MIL に渡す。
    fusion_dim: int | None = Field(default=1024, gt=0)

    def model_post_init(self, context: object, /) -> None:
        """``blend`` が非負で和 1、``loss_weights`` が非負であることを検める。"""
        if min(self.blend) < 0.0 or abs(sum(self.blend) - 1.0) > 1e-6:
            raise ValueError(f"model.multibranch.blend must be non-negative and sum to 1, got {self.blend}")
        if min(self.loss_weights) < 0.0:
            raise ValueError(f"model.multibranch.loss_weights must be non-negative, got {self.loss_weights}")


class SlotContextConfig(BaseModel):
    """所見別の Slot 内 -> Slot 間の 2 段 attention で作った slot / study 文脈を窓特徴へ残差で戻し、別の head ``ctxW`` で窓 logit を読む枝。出力層はゼロ初期化。

    ``model.context`` と併用するときは、その残差を足した後の所見ごとの窓特徴を入力にする。
    """

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    # 文脈の次元。
    dim: int = Field(default=256, gt=0)
    # Slot 内・Slot 間の attention スコアの tanh 層の幅。
    att_hidden: int = Field(default=256, gt=0)
    # 残差 MLP の隠れ次元。
    hidden: int = Field(default=256, gt=0)


class ModelConfig(BaseModel):
    """窓 bag モデルのアーキテクチャ。

    ``extra="forbid"`` は意図的である。解釈できないキーの入った config は黙って既定値に戻るのではなく
    失敗しなければならない。
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["window", "vlm", "multibranch"] = "window"
    vlm: VlmConfig | None = None
    # ``kind="multibranch"`` のときだけ必須。このとき ``backbone`` は読まれない。
    multibranch: MultiBranchConfig | None = None
    backbone: str = "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"
    pretrained: bool = True
    img_size: int = Field(default=384, gt=0)
    # backbone と SAP（``model.spatial``）を固定し、外側の LayerNorm・head・枝だけを学習する。
    freeze_encoder: bool = False
    # attention-MIL の pooled 特徴から所見ごとの重症度を当てる補助の線形 head。``forward`` の logits には関わらない。
    severity_head: bool = False
    # 撮像面（Sagittal / Coronal / Axial）ごとに、その面の窓だけを主の attention で集約して 12 所見を当てる補助の線形 head。学習の補助損失（``train.plane_aux_weight``）と検証の記録だけが読み、``forward`` の logits には関わらない。
    plane_aux: bool = False
    # この重みが fitted された DICOM -> tensor のレシピ。``build_datasets`` が cache の manifest から埋める。
    preprocess: PreprocessConfig | None = None
    window: WindowConfig = WindowConfig()
    fsx: FsxConfig = FsxConfig()
    seq: SeqConfig = SeqConfig()
    depth: DepthConfig = DepthConfig()
    pair: PairConfig = PairConfig()
    spatial: SpatialConfig = SpatialConfig()
    context: ContextConfig = ContextConfig()
    slot_context: SlotContextConfig = SlotContextConfig()
    series_neck: SeriesNeckConfig = SeriesNeckConfig()

    def model_post_init(self, context: object, /) -> None:
        """モデル種別に対応した設定だけを受け付ける。"""
        if (self.kind == "vlm") != (self.vlm is not None):
            raise ValueError("model.vlm is required exactly when model.kind=vlm")
        if (self.kind == "multibranch") != (self.multibranch is not None):
            raise ValueError("model.multibranch is required exactly when model.kind=multibranch")
        if self.kind == "multibranch" and (self.fsx.enabled or self.seq.enabled or self.depth.enabled or self.pair.enabled or self.spatial.enabled or self.severity_head):
            raise ValueError("model.kind=multibranch has its own GAP / SAP heads: disable fsx, seq, depth, pair, spatial and severity_head")
        if self.kind == "vlm" and (self.severity_head or self.freeze_encoder):
            raise ValueError("VLM trains its vision encoder and has no severity head")
        if self.kind == "vlm" and self.fsx.enabled:
            raise ValueError("model.fsx is for the window model only")
        if self.kind == "vlm" and self.seq.enabled:
            raise ValueError("model.seq is for the window model only")
        if self.kind == "vlm" and self.depth.enabled:
            raise ValueError("model.depth is for the window model only")
        if self.pair.enabled and not (self.depth.enabled and self.depth.kind in ("conv_depth", "conv3d")):
            raise ValueError("model.pair needs model.depth.enabled=true with kind conv_depth or conv3d (it reads the depth stem)")
        if self.pair.enabled and self.fsx.enabled:
            raise ValueError("model.pair cannot run with model.fsx")
        if self.plane_aux and self.kind != "window":
            raise ValueError("model.plane_aux is for the window model only")
        if self.spatial.enabled and (self.kind != "window" or self.fsx.enabled or self.seq.enabled or self.depth.enabled or self.pair.enabled):
            raise ValueError("model.spatial runs on the plain window model: kind=window with fsx, seq, depth and pair disabled")
        if self.window.pool != "attention" and (self.kind != "window" or self.fsx.enabled or self.pair.enabled or self.severity_head or self.plane_aux):
            raise ValueError(f"model.window.pool={self.window.pool} pools window logits: kind=window with fsx, pair, severity_head and plane_aux disabled")
        if self.context.enabled and (not self.spatial.enabled or self.window.pool == "attention" or self.fsx.enabled or self.seq.enabled or self.depth.enabled or self.pair.enabled):
            raise ValueError("model.context needs model.spatial and model.window.pool other than attention (it adds to the per-finding window features), with fsx, seq, depth and pair disabled")
        if self.slot_context.enabled and (not self.spatial.enabled or self.window.pool == "attention" or self.fsx.enabled or self.seq.enabled or self.depth.enabled or self.pair.enabled):
            raise ValueError("model.slot_context needs model.spatial and model.window.pool other than attention (it adds to the per-finding window features), with fsx, seq, depth and pair disabled")
        if self.series_neck.enabled and (not self.spatial.enabled or self.window.pool != "attention" or self.severity_head or self.context.enabled):
            raise ValueError("model.series_neck needs model.spatial and model.window.pool=attention, with severity_head and context disabled")


class TrainConfig(BaseModel):
    """学習ループの設定。"""

    epochs: int = 20
    learning_rate: float = 1e-3  # head / attention のパラメータ
    encoder_lr: float = 3e-5  # timm の backbone
    # ``model.kind="multibranch"`` で枝 0・枝 1 の backbone に別々の学習率を置くとき。None なら両方 ``encoder_lr``。
    encoder_lrs: tuple[float, float] | None = None
    weight_decay: float = 0.02
    grad_clip_val: float = 3.0
    accumulate_grad_batches: int = 1
    precision: str = "bf16-mixed"
    # どちらの scheduler でも、最初のこの割合で学習率を上げる。
    pct_start: float = Field(default=0.15, gt=0.0, lt=1.0)
    # "onecycle" は OneCycleLR（momentum も一緒に振る）、"cosine" は線形 warmup のあと cosine で 0 まで。
    scheduler: Literal["onecycle", "cosine"] = "cosine"
    # backbone の活性を保存せず再計算する。24 GB のカードで 384 の CoAtNet-2 を回すには必須。
    grad_checkpointing: bool = True
    # :func:`rsna_knee.modules.losses.pairwise_rank_loss` を BCE に足す重み。0 で無効。
    rank_weight: float = Field(default=1.0, ge=0.0)
    # ランク損失の比較相手に足す直近の学習 batch の数。0 なら batch 内の対だけ。
    rank_queue: int = Field(default=16, ge=0)
    # 分類項。"bce" は masked BCE、"soft_ral" は RAL の陽性枝・陰性枝を y で線形に混ぜたもの、"soft_fcl" は soft focal + 二乗誤差。
    cls_loss: Literal["bce", "soft_ral", "soft_fcl"] = "bce"
    # "soft_fcl" の focusing 指数 γ と二乗誤差の係数 β。
    fcl_gamma: float = Field(default=2.0, ge=0.0)
    fcl_beta: float = Field(default=1.0, ge=0.0)
    # 順位項。"ranknet" は pairwise_rank_loss、"soft_lse" は anchor ごとの LSE 集約（soft_lse_rank_loss）。
    rank_loss: Literal["ranknet", "soft_lse"] = "ranknet"
    # "soft_lse" の温度 T。
    rank_temperature: float = Field(default=1.0, gt=0.0)
    # gold でない study のセルの BCE に掛ける係数。``data.weights_csv`` の重みとの積が 1 セルの重みになる。
    pseudo_weight: float = Field(default=1.0, ge=0.0)
    # 重症度の補助損失（``model.severity_head`` の出力と ``data.severity_csv`` の soft BCE）を主損失に足す重み。0 で無効。
    severity_weight: float = Field(default=0.0, ge=0.0)
    # 学習開始前にモデルの重みだけを読み込む Lightning checkpoint か artifact の ``model.pt``。optimizer と学習率の曲線は新しく始める。
    # checkpoint に無い重み（新しく有効にした枝）は初期値のまま、モデルに無い重みがあれば落ちる。
    init_weights: Path | None = None
    # ``init_weights`` から読まない重みの名前の接頭辞（例 ``["clsW", "clsb"]`` で head を初期値から学習する）。
    init_weights_skip: list[str] = []
    # 毎 epoch の終わりにモデルの重みを ``checkpoints/epoch{NN}.pt`` に書き出す（SWA 用）。
    save_every_epoch: bool = False
    # 重みの EMA の減衰率（optimizer の step ごと）。warmup が終わるまでは学習中の重みをそのまま写す。検証・checkpoint の選択・export は EMA で行う。None で無効。
    ema_decay: float | None = Field(default=None, gt=0.0, lt=1.0)
    # 深さ prior から作る窓ターゲット（study の soft ラベル x 所見別 prior 密度の最大値正規化）と窓 logit の BCE を主損失に足す重み。0 で無効。``model.window.pool`` が "attention" 以外で ``data.depth_prior`` があるときだけ。
    window_loss_weight: float = Field(default=0.0, ge=0.0)
    # 撮像面ごとの補助 head（``model.plane_aux``）の分類項を、窓のある面で平均して主損失に足す重み。0 で無効。
    plane_aux_weight: float = Field(default=0.0, ge=0.0)

    def model_post_init(self, context: object, /) -> None:
        """queue はランク損失にしか渡らないので、重みが 0 のまま立っていたら拒む。"""
        if self.rank_queue > 0 and self.rank_weight == 0.0:
            raise ValueError(f"train.rank_queue {self.rank_queue} only feeds the rank loss: set train.rank_weight > 0 as well")


class Config(BaseSettings):
    """ルートの設定。

    優先度は キーワード引数（``--config`` の JSON）> ``RSNA_KNEE_`` 付きの環境変数
    （入れ子はアンダースコア 2 個。例 ``RSNA_KNEE_TRAIN__EPOCHS=20``）> 既定値。
    """

    model_config = SettingsConfigDict(env_prefix="RSNA_KNEE_", env_nested_delimiter="__", extra="ignore")

    experiment_name: str = "baseline"
    wandb_project: str = "rsna-knee-abnormality-detection"
    wandb_mode: Literal["online", "offline", "disabled"] = "disabled"
    seed: int = Field(default=42, ge=0)
    output_dir: Path = Path("output")
    debug: bool = False

    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()

    def model_post_init(self, context: object, /) -> None:
        """重症度の補助損失は head と表の両方、窓の補助損失は窓 logit と深さ prior の両方が要る。片方だけ立った config は拒む。"""
        if self.train.encoder_lrs is not None and self.model.kind != "multibranch":
            raise ValueError("train.encoder_lrs is for model.kind=multibranch only")
        if self.train.severity_weight > 0.0 and (self.data.severity_csv is None or not self.model.severity_head):
            raise ValueError(f"train.severity_weight {self.train.severity_weight} needs data.severity_csv and model.severity_head=true")
        if self.train.window_loss_weight > 0.0 and (self.model.window.pool == "attention" or self.data.depth_prior is None):
            raise ValueError(f"train.window_loss_weight {self.train.window_loss_weight} needs model.window.pool other than attention (window logits) and data.depth_prior (window targets)")
        if self.train.plane_aux_weight > 0.0 and not self.model.plane_aux:
            raise ValueError(f"train.plane_aux_weight {self.train.plane_aux_weight} needs model.plane_aux=true")
        if self.model.plane_aux and self.train.severity_weight > 0.0:
            raise ValueError("model.plane_aux cannot be combined with train.severity_weight")
