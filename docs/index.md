# 8th Place Solution: Image-First Label Distillation and Global–Local View Modeling

このコンペティションを開催してくださったRSNAとKaggleチーム、チームメイトの@nantohock, @wadakokiに感謝します。
主な戦略は、複数の断面・撮像条件のglobal特徴とlocal特徴を組み合わせるVisionモデルと、ソフトラベルの改善だった。追加のMRIデータは使わなかった。

## 1. DICOMキャッシュとAugmentation

隣接スライスの関係をできるだけ保つため、各撮像条件群の保存枚数を増やした。矢状断・冠状断をそれぞれ液体強調優先と非液体強調優先の2群に分け、横断像と合わせて5群とした。

| 項目 | 条件 |
|---|---|
| 採取枚数 | 5群から32／32／32／32／40枚、最大168枚 |
| 切り出し範囲 | 140 × 140 mm。110 × 110 mmも使用 |
| 画像サイズ | 384 × 384画素 |
| 入力 | 同じ系列の中心と前後各1枚の計3枚を3チャネルに積む |
| スライス中心間隔 | 上限枚数に収まる系列では元の間隔を保持（典型的に3〜5 mm） |

上限枚数を超える系列は等間隔に間引いた。終盤には、採取対象が30枚を超える系列で、開始位置をずらしたsub-seriesも作った。例えば96枚から32枚を採る場合、「1・4・7…94枚目」「2・5・8…95枚目」「3・6・9…96枚目」の3通りとなる。学習時には通常の等間隔採取版も含めてランダムに1本を選んだ。

キャッシュは約110GBとなった。64枚のキャッシュを使ったモデルのPublic LBは0.930、最大168枚では0.937だった。

### 1.1 Augmentation

| Augmentation | 設定 | 適用確率・単位 |
|---|---|---|
| 明度変化 | ゲイン0.9〜1.1倍 | 毎回、検査全体で共通 |
| 鏡像反転 | 冠状断・横断像は左右反転、矢状断は3チャネルの順序を反転 | 50%、検査単位 |
| 回転・拡縮・平行移動 | 回転±15°、拡縮0.9〜1.1倍、移動は画像幅の±5% | 3変換をまとめて50%、撮像条件群ごと |
| 低解像度化 | 一辺を0.5〜1.0倍に縮小してから元のサイズへ戻す | 25%、撮像条件群ごと |
| Gaussian blur | σ = 0.5〜1.0画素 | 20%、撮像条件群ごと |
| Gaussian noise | 標準偏差0〜0.03（画素値0〜1に対して） | 10%、撮像条件群ごと |
| 系列の選択（第1段階） | 各撮像条件群から1本を選ぶ | 学習時に毎回ランダムに選択 |
| sub-seriesの選択（第1段階） | 通常の採取版と開始位置をずらした採取版から1本を選ぶ | 候補から等確率で選択 |

鏡像反転では反対側の膝として見せる。矢状断は左右方向がスライスの並ぶ方向に当たるため、画像内を反転せず、3チャネルの順序を逆転させた。内側・外側の区分は変わらないので、ラベルは入れ替えない。

## 2. ラベルの改善

レポートの記載を正確に抽出しても、画像から付けられた評価ラベルと同じ基準になるとは限らない。そこで、画像判定とモデルのOOFを学習ラベルに取り込んだ。

### 2.1 レポートラベルの準備

レポート由来のラベルは2系統を用意した。

1つ目（L）は、次の3つのLLMラベルの平均である。

- 公開ラベル：Kaggleで公開されていたレポート由来のLLMラベル（[dreaddevelopment/rsna-knee-labels](https://www.kaggle.com/datasets/dreaddevelopment/rsna-knee-labels)）。骨折・骨挫傷の2所見だけは、GPT-6 Astraにレポートを読み直させた判定に置き換えた。
- Qwen／Gemma：Qwen3.8-27BとGemma 4 26B-A4Bに同じプロンプトでレポート本文だけを読ませ、重症度・量・範囲で症例を順位付けたスコア。記載のない所見の既定値は公開ラベルから逆推定し、公開ラベルと同じ尺度にそろえた。プロンプト・設定・コードは[`data/label/wadakoki/`](../data/label/wadakoki/README.md)。

3つのラベルは互いによく似た順位を付けていた（所見ごとのSpearman相関の平均：Qwen–Gemma 0.898、Qwen–公開ラベル 0.863、Gemma–公開ラベル 0.859）。そこで、どれか1つを選ばず、3つを同じ重みで平均してLとした。

2つ目（A）は、Astraにレポートを読ませ、所見ごとに陽性・否定・条件不足などの状態を判定してsoft labelにしたものである。レポートに記載のない所見は一律の値にせず、所見ごとの存在割合、所見がある場合とない場合の未記載率、レポートの様式、他の所見の有無から確率を推定した。

### 2.2 画像からのラベル付け

GPT-6 Astraには、まずレポートや既存ラベルを見せず、DICOM画像から12所見がコンペの判定基準を満たす確かさを0〜1で評価するよう依頼した。0.5以上を陽性とし、根拠スライスとbboxも記録させた。レポートとの照合は画像を確認した後に行い、不一致は画像を再確認して、画像での確認結果を優先させた。

| ラベルの変更 | Public LB |
|---|---:|
| 公開ラベルの骨折・骨挫傷を、Astraによるレポートの再判定で置換 | 0.937 → 0.943 |
| Astraのレポート判定と画像判定を一部追加 | 0.944 → 0.949 |
| 画像判定とキャリブレーションしたモデル予測の比重を増加 | 0.950 → 0.954 |

表は主要な更新時点の比較で、一部は学習設定の変更も含む。Public LB 0.949の時点では、Astra画像判定の適用率は約20%だった。最終的には、train 4,407検査のうち4,406検査に画像判定を付けた。

![Astraによる画像先行ラベル付けの手順](figure/astra_labeling.png)

*図1: Astraによる画像先行ラベル付け。画像のみで判定した後にレポートと照合し、不一致は画像を再確認した。*

### 2.3 テキストと画像の配合

その後も画像判定の対象を増やし、所見ごとの配合を見直した。最終ラベルには次の3種類を使った。

| 記号 | ラベルの出典 |
|---|---|
| I | Astraの画像判定 |
| L | 公開ラベル（骨折・骨挫傷はAstraで再判定）、Qwen、Gemmaの平均 |
| A | Astraのレポート判定（記載のない所見は存在割合などから推定） |

画像判定の重みwを所見ごとに決め、残りをLとAへ等分した。画像判定がない検査ではLとAを等分した。

$$
S = wI + \frac{1-w}{2}\,(L + A).
$$

| 所見 | 画像判定の重みw |
|---|---:|
| Fracture / Synovitis / Baker's | 5/6 |
| Effusion / Contusion | 2/3 |
| Medial OA | 1/2 |
| ACL / Medial Meniscus / Lateral Meniscus | 1/3 |
| PF OA / Lateral OA / MCL | 1/6 |

画像判定の重みを全所見共通で変えて評価し、Goldの所見別AUCが最大となる重みwを採用した。その後、所見ごとに±1/6ずつ動かす探索も行ったが、Gold AUC（0.9346 vs 0.9348）とPublic LB（ともに0.954）に判断に値する差はなかったため、探索前の重みを採用した。

trainには、1つの検査に両膝の系列が入っているものが8検査あった。ラベルは検査単位なので、そのまま学習すると、片方の膝の画像にもう片方の膝の所見が付いてしまう。そこで、系列中心の左右座標を使い、7検査を膝ごとの14例に分けて学習した。残る1検査は、反対膝が矢状断のみで画像評価できなかったため除外した。左右それぞれのラベルは、画像判定とレポートの記載を所見別の重みで配合して作った。提出時の推論は検査単位のままとした。

### 2.4 pseudo labeling

Sを教師ラベルとし、撮像装置の特徴で分けた4foldのOOFを作った。正例の重み付けと順位損失のため、OOFの所見ごとの平均は0.37〜0.46と、ラベルの平均0.10〜0.40より大きく上にずれる。そこで、OOFは混合前にSの尺度へキャリブレーションした。各foldの変換には、残り3foldで学習した所見別のisotonic回帰を使った。最終的な学習ラベルyは次の配合とした。

$$
y = 0.6S + 0.4\,\mathrm{OOF}_{\mathrm{cal}}.
$$

Sだけで学習したモデルと、キャリブレーションしたOOFだけで学習したモデルの予測をGold上で混ぜ、最良だったOOF側の比率0.4を選んだ。この比率を学習ラベルにも使って再学習すると、Gold AUCはSだけの0.9309から0.9346に上がった。

さらに、自己蒸留として同じ手順をもう1周行った。1周目の学習ラベルyで教師を学習し直し、そのOOFを同じ方法でキャリブレーションして、もう一度0.4の割合で混ぜた。混ぜる相手は1周目のyではなく、常にSとした。yにさらにOOFを重ねると、Sの重みが0.6 → 0.36と周回ごとに減っていくためである。

生徒モデルのGold AUC（4 seed平均）は、1周目の0.9414から2周目の0.9427に上がり、4 seed中3つで1周目を上回った。このラベルは提出直前に完成し、4.4の「更新後のラベル」として使った。

Goldで評価する実験では、Goldを教師の学習にもOOFにも入れていない。教師にGoldが入ると、pseudo labelを通じて生徒の評価にGoldの情報が漏れるためである。

## 3. 損失関数

正例重み付きBCEとLSE型の順位損失を係数1ずつで加えた。順位損失では、ソフトラベルの値が高い検査を上位に置くよう学習し、予測の逆転が大きい相手ほど指数関数で強く重み付けした。

$$
\ell_i^{LSE}=\log\left(1+\frac{1}{M_i}\sum_{j\in J_i}
q_{ij}\exp\left(\frac{s_j-s_i}{T}\right)\right),
\qquad q_{ij}=\max(y_i-y_j,0),\quad T=1.
$$

yはソフトラベル、sはlogit、J_iはiよりラベル値が低い比較相手、M_iはその数である。比較相手は現在のbatchと、直近のbatchのlogitを保存したqueueから選んだ。比較相手のある検査（M_i > 0）の損失を、所見ごとに平均した。

順位損失の有無は、初期のRankNet型で比較した。

| 損失 | Gold AUC | Public LB |
|---|---:|---:|
| BCEのみ | 0.9261 | 0.943 |
| BCE＋RankNet型の順位損失 | 0.9300 | 0.944 |

RankNet型とLSE型の差は小さかったが、Public LBがわずかに上がったLSE型を採用した。

## 4. モデル構成とアンサンブル

中心と前後各1枚を積んだ入力を窓と呼ぶ。backboneにはCoAtNet-2（timmのcoatnet_rmlp_2_rw_384、ImageNet-12k事前学習）とCAFormer-S36を使った。

窓ごとの特徴マップから、GAPでglobal特徴、空間attention pooling（SAP）で12所見分のlocal特徴を取り出した。SAPの重みは所見と撮像条件に応じて変わる。これらの特徴の組み合わせ方と、窓の集約方法を変えたモデルを作った。

### 4.1 系列内GRU＋2段attention

各窓のglobal特徴に、系列内の相対的な深さ、前後の窓との深さの差、撮像条件の埋め込みを加え、系列ごとに深さ順で双方向GRUへ通した。local特徴はGRUへ通さずに使った。

globalとlocalをそれぞれ所見別attentionで系列内集約し、連結した。全体像と局所所見で別々の窓に重みを付けられる。さらに、撮像条件を考慮した所見別attentionで系列間を集約し、12所見を予測した。学習時には撮像面ごとの補助headも使った。

![系列内GRU＋2段attentionの構成図](figure/nanachi_models.png)

*図2: 系列内GRU＋2段attention。左がglobal（GAP）側、右がlocal（SAP）側。*

### 4.2 LSEと文脈attention head

各窓のglobal特徴をself-attentionへ通し、窓間の文脈を作った。この文脈と各窓のglobal＋local特徴から補正量を求め、窓特徴に残差として加えた。この構成を文脈attention＋ctxと呼ぶ。派生として、撮像条件群内・群間の所見別attentionで作った文脈を窓特徴に加える文脈attention＋slotも使った。

補正後の特徴から各窓の12所見のlogitを出し、所見ごとにlog-sum-exp（LSE）で検査全体へ集約した。

$$
z_c=\tau\log\left(\frac{1}{K}\sum_{k=1}^{K}\exp(z_{kc}/\tau)\right),\qquad\tau=1.
$$

Kは実在する窓数、$z_{kc}$は窓kの所見cに対するlogitである。attention-MILが特徴を集約してから予測するのに対し、この構成では窓ごとのlogitを集約する。

![LSEと文脈attention headの構成図](figure/hirose_models.png)

*図3: LSEと文脈attention head。オレンジの枠は学習時だけ使う窓単位の補助損失。*

窓の位置に応じた教師値を与えるため、深さpriorを使った補助損失も加えた。教師モデルのOOF推論で得たglobal側の窓attention重みを、撮像条件・所見ごとに相対的な深さに沿って集計し、混合ガウスで滑らかな分布にした。

分布の最大値を1にそろえ、「検査のソフトラベル × その位置のprior」を窓の教師値としてBCEを計算した。病変位置の正解ではなく、教師モデルの注目位置の傾向を利用した。

![所見と撮像条件ごとの深さprior](figure/prior.png)

*図4: 所見と撮像条件（列）ごとの深さprior。横軸は系列内の相対的な深さ、縦軸は最大値を1にそろえた教師値の重み。*

### 4.3 union bagによる追加学習（After Training）

第1段階では撮像条件群ごとに系列を1本選び、メモリの制約から24窓で学習した。第2段階ではbackboneと空間attentionを固定し、下流層を追加学習した。

固定によりメモリ使用量を抑え、キャッシュした全系列・sub-seriesの窓を候補にまとめたunion bagを使った。学習時には系列と採取版の数をランダムに選び、そこから32〜352窓を採った。

分類headをそのまま学習すると早期に過学習する傾向があったため、headの初期化と学習率を変えて比較した。同じ学習済みCoAtNetモデルを起点に、10 epoch追加学習した結果を示す。

| 条件 | 追加学習の学習率 | Gold macro AUC |
|---|---:|---:|
| 追加学習前 | — | 0.9447 |
| headをリセットせず継続 | 3e-5 | 0.9464 |
| headをリセットして学習 | 1e-4 | **0.9507** |

この結果から、headを初期化して追加学習する設定を採用した。

### 4.4 Ensemble

backbone、neck、切り出し範囲、seed、学習ラベルを変えたモデルを組み合わせた。入力解像度はすべて384 × 384画素。

提出直前にラベルを更新したため、更新前後の2種類のラベルで学習したモデルが揃った。Select Submit AのCoAtNetは更新前のラベルのモデルだけを使い、Bの文脈attention系CoAtNetは両方を混ぜた。CAFormerは更新後のラベルで学習したモデルを使った。

| backbone | neck | 切り出し範囲 | モデル数 | 学習ラベルの内訳 | 使用するSubmit |
|---|---|---|---:|---|---|
| CoAtNet-2 | 系列内GRU＋2段attention | 140 mm | 4 | 更新前4 | A |
| CoAtNet-2 | 系列内GRU＋2段attention | 110 mm・前方10 mm | 4 | 更新前4 | A |
| CoAtNet-2 | 文脈attention＋ctx（LSE） | 140 mm | 4 | 更新前2＋更新後2 | B |
| CoAtNet-2 | 文脈attention＋slot（LSE） | 140 mm | 4 | 更新前2＋更新後2 | B |
| CAFormer-S36 | 系列内GRU＋2段attention | 140 mm | 4 | 更新後4 | A/B共通 |
| CAFormer-S36 | 系列内GRU＋2段attention | 110 mm・前方10 mm | 3 | 更新後3 | B |

110 mmの切り出しは、中心から前方へ10 mmずらした位置をcropしている。CAFormerの140 mm 4モデルはA/Bで共通で、110 mmの3モデルはBだけで使う。ラベル欄の数字はモデル数を表す。

#### Select Submit A

Public Bestとなった組み合わせ。
系列内GRU＋2段attentionの12モデル（CoAtNet 8＋CAFormer 4）をすべて等分して確率平均する。チェックポイントはGoldで最良のEMAを選び、最良epochが4未満ならepoch 4のEMAを使った。推論はunion bagの384窓、TTAなしとした。

#### Select Submit B

Gold macro AUCが最も高くなる組み合わせを選んだ。
文脈attention＋ctx、文脈attention＋slot、CAFormerの140 mm、110 mmの4群を使う。各群内のモデルを等分して確率平均し、その結果を4群で等分する。
Goldでの比較では、文脈attention＋ctxが0.9508、文脈attention＋slotが0.9492で、この2系統を等分して混ぜると0.9527だった。


## 5. GoldとPublicによる検証

ソフトラベルへのOOFだけでは、画像から付けられた評価ラベルへの性能は分からない。医師による58検査の正解ラベル（Gold）を検証に使い、Public LBとの対応を確かめた。

所見別の比較では、同じモデルの予測から1列だけを残し、他の11列を0.5に固定するprobeを行った。Publicの所見別AUCは AUC = 12 × LB − 5.5 で概算した。LBの表示は小数第3位までなので、換算値の刻みは0.012となる。

![GoldとPublicの対応](figure/validation.png)

*図5: 左は1モデルの所見別AUC（GoldとPublic LBのprobe換算値）。右は22提出のGold macro AUCとPublic LB。*

GoldとPublicの所見別AUCは、EffusionやMedial Meniscusでは近かったが、Lateral MeniscusやMedial OAでは0.07〜0.1ほど差があった。一方、22提出のGold macro AUCとPublic LBには正の相関が見られた（Pearson r = 0.921、Spearman ρ = 0.911）。

Goldの絶対値や小さな差は慎重に扱い、全体の改善傾向を見る指標として使った。毎回4モデルを学習してOOFを揃えるより、Goldで候補を速く試すことを優先した。Public LBへの過学習にも注意しながら実験を進めた。

## 6. 最終結果

| 提出 | Public | Private |
|---|---|---|
| Public Best（Select Submit A） | 0.968 | 0.957 |
| Gold macro AUC優先（Select Submit B） | 0.967 | 0.957 |
| Private最高（未選択） | 0.967 | 0.958 |

スコアは現時点の表示値。順位確定後に、順位と詳細な桁数のスコアを追記する。

## 参考

1. Dai et al., [CoAtNet: Marrying Convolution and Attention for All Data Sizes](https://arxiv.org/abs/2106.04803), NeurIPS 2021
2. Guan et al., [Diagnose like a Radiologist: Attention Guided Convolutional Neural Network for Thorax Disease Classification](https://arxiv.org/abs/1801.09927), arXiv 2018
3. Ilse et al., [Attention-based Deep Multiple Instance Learning](https://proceedings.mlr.press/v80/ilse18a.html), ICML 2018
4. Shao et al., [TransMIL: Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classification](https://arxiv.org/abs/2106.00908), NeurIPS 2021
5. Li et al., [Dual-stream Multiple Instance Learning Network for Whole Slide Image Classification with Self-supervised Contrastive Learning](https://arxiv.org/abs/2011.08939), CVPR 2021
6. Pinheiro & Collobert, [From Image-Level to Pixel-Level Labeling With Convolutional Networks](https://openaccess.thecvf.com/content_cvpr_2015/html/Pinheiro_From_Image-Level_to_2015_CVPR_paper.html), CVPR 2015
7. Burges et al., [Learning to Rank using Gradient Descent](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/), ICML 2005
8. Lin et al., [Reference-Guided Gradient Balancing Loss for Long-Tailed Multi-Label Medical Image Classification](https://papers.miccai.org/miccai-2026/paper/0566_paper.pdf), MICCAI 2026
