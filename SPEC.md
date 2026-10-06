# SPEC.md — 單張魚眼影像相機內參回歸（CNN vs ViT）

> 本文件是給實作代理（agentic AI）的規格書。**依 Phase 順序實作，每個 Phase 的驗收條件全部通過才進入下一個。** 遇到本文件未定義、或與本文件衝突的情況，停下來回報，不要自行假設。

---

## 1. 任務目標

### 1.1 問題定義
輸入一張 224×224 的魚眼影像，輸出其相機內參 **UCM (Unified Camera Model) 的 `(f, ξ)`**。主點固定在影像中心（刻意簡化）。

### 1.2 專案目的
- 個人作品集專案，最終公開於 GitHub。重點在**訓練方法論**（資料設計、loss、評估、實驗設計、error analysis），不是手刻網路架構。
- 以 ResNet-34（CNN）與 ViT-S/16（Transformer）做對照實驗。

### 1.3 硬性限制
- 只使用公開資料與個人算力（RTX 4070 12GB、32GB RAM）。**不得出現任何公司內部程式碼、資料、路徑、主機名稱或服務（例如 Run:ai）。**
- 資料來源：Poly Haven HDRI 全景圖（CC0）。README 必須標註素材來自 Poly Haven。
- 全流程可重現：所有隨機性由 config 中的 seed 決定。

### 1.4 Phase 劃分

| Phase | 內容 | 狀態 |
|---|---|---|
| 1 | 幾何庫、資料下載、切分、離線渲染、Dataset、目視檢查 | **本次實作範圍** |
| 2 | 訓練、評估、實驗 sweep、MLflow | 介面先定義，Phase 1 驗收後再實作 |
| 3 | ONNX 匯出、TensorRT INT8、KITTI-360 sim-to-real | 加分項，暫不實作 |

---

## 2. 檔案列表與責任切分

```
fisheye-calib/
├── configs/
│   ├── data.yaml                 # 資料生成的所有參數（範圍、數量、seed）
│   └── train/                    # [Phase 2] 每個實驗一份 yaml
├── geometry/
│   ├── __init__.py
│   ├── ucm.py                    # UCM 投影、反投影、FOV↔f、有效範圍
│   ├── rotation.py               # yaw/pitch/roll → 旋轉矩陣
│   ├── equirect.py               # 方向 ↔ 全景圖像素座標、取樣
│   └── render.py                 # 全景圖 → 魚眼（含超取樣）、魚眼 → 透視圖
├── data/
│   ├── download_polyhaven.py     # 下載全景圖 + manifest
│   ├── split.py                  # 以全景圖為單位切分 + 資料量子集
│   ├── sampling.py               # 相機參數取樣（含可行域拒絕取樣）
│   └── render_dataset.py         # 離線渲染資料集
├── datasets/
│   ├── fisheye_dataset.py        # 離線資料集（memmap）
│   ├── online_dataset.py         # 線上渲染資料集
│   └── transforms.py             # 光度增強 + 允許的幾何增強
├── metrics/
│   └── geometric.py              # 重投影誤差、幾何 loss（torch）
├── tools/
│   ├── visualize_samples.py      # 樣本網格、參數分佈直方圖
│   └── report_categories.py      # 全景圖分類統計
├── train.py                      # [Phase 2]
├── evaluate.py                   # [Phase 2]
├── sweep.py                      # [Phase 2]
├── tests/
│   ├── test_ucm.py
│   ├── test_equirect.py
│   ├── test_render.py
│   ├── test_sampling.py
│   ├── test_dataset.py
│   └── test_metrics.py
├── requirements.txt
└── README.md
```

### 2.1 責任切分

| 模組 | 負責 | 不負責 |
|---|---|---|
| `geometry/*` | 純幾何運算，numpy 實作（`metrics/` 另有 torch 版）。無 I/O、無隨機性 | 讀檔、取樣 |
| `data/download_polyhaven.py` | 呼叫 API、下載、續傳、MD5 驗證、manifest | 任何影像處理 |
| `data/split.py` | 產生 `splits.json`（train/val/test + 資料量子集） | 渲染 |
| `data/sampling.py` | 給定 rng，取樣一組相機參數 | 渲染、I/O |
| `data/render_dataset.py` | 讀全景圖 → 呼叫 sampling + render → 寫 `.npy` 與 labels | 幾何公式本身（全部呼叫 `geometry/`） |
| `datasets/*` | 讀取、增強、label 標準化，回傳 tensor | 生成資料（online 版除外，且仍呼叫 `geometry/`） |
| `metrics/geometric.py` | 可微分的重投影誤差與 loss | 模型 |

**原則：所有幾何公式只能存在於 `geometry/`（numpy）與 `metrics/geometric.py`（torch）。兩者公式必須一致，並由測試交叉驗證。**

### 2.2 資料目錄（不進 git，加入 `.gitignore`）

```
data_root/
├── raw/polyhaven/{asset_id}.jpg
├── manifest.json
├── splits.json
└── rendered/{split}/
    ├── images.npy        # uint8, shape (N, 224, 224, 3), RGB
    ├── labels.csv
    └── meta.json
```

---

## 3. 核心 API / 介面設計

### 3.1 座標慣例（全專案統一）

- **影像座標**：像素中心位於整數座標，左上角像素中心為 `(0, 0)`。影像大小 `S`，主點 `cx = cy = (S - 1) / 2`。
- **相機座標**：OpenCV 慣例，x 向右、y 向下、z 向前。
- **世界座標**：`R = I` 時與相機座標相同。`d_world = R @ d_cam`。
- **全景圖（equirectangular，W×H，W = 2H）**：
  - 經度 `φ = atan2(x, z)`，範圍 `[-π, π)`，`φ = 0` 對應影像水平中央。
  - 緯度 `λ = arcsin(-y)`，向上為正。
  - 像素：`u_e = (φ / 2π + 0.5)·W − 0.5`，`v_e = (0.5 − λ / π)·H − 0.5`。
- **角度**：函式內部一律用 radian；config、labels 與介面參數名稱帶 `_deg` 者用 degree。

### 3.2 `geometry/ucm.py`

```python
def project(rays: np.ndarray, f: float, xi: float, cx: float, cy: float
            ) -> tuple[np.ndarray, np.ndarray]:
    """rays (..., 3) 不需單位化 → (pixels (..., 2), valid (...,) bool)"""

def unproject(pixels: np.ndarray, f: float, xi: float, cx: float, cy: float
              ) -> tuple[np.ndarray, np.ndarray]:
    """pixels (..., 2) → (unit rays (..., 3), valid (...,) bool)"""

def radius_from_theta(theta: np.ndarray, f: float, xi: float) -> np.ndarray:
    """與光軸夾角 θ → 影像半徑 r（像素）"""

def theta_max(xi: float) -> float:
    """此 ξ 下投影仍為單調（一對一）的最大 θ"""

def f_from_fov(fov_deg: float, xi: float, r_ref: float) -> float:
    """在半徑 r_ref 處的全角 FOV → f"""

def fov_from_f(f: float, xi: float, r_ref: float) -> float:
    """f_from_fov 的反函數"""
```

### 3.3 `geometry/rotation.py`

```python
def rotation_from_ypr(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """回傳 3×3 R，d_world = R @ d_cam。
    R = R_yaw(繞 y 軸) @ R_pitch(繞 x 軸) @ R_roll(繞 z 軸)。
    語意（必須由測試驗證）：pitch > 0 相機朝上看；yaw > 0 相機朝右轉。"""
```

### 3.4 `geometry/equirect.py`

```python
def dirs_to_equirect_px(dirs: np.ndarray, W: int, H: int) -> np.ndarray:
    """(..., 3) → (..., 2) 全景圖像素座標"""

def equirect_px_to_dirs(px: np.ndarray, W: int, H: int) -> np.ndarray:
    """反函數，回傳單位向量"""

def sample_equirect(pano: np.ndarray, map_u: np.ndarray, map_v: np.ndarray) -> np.ndarray:
    """cv2.remap 雙線性取樣。水平方向必須正確環繞（見 4.4）"""
```

### 3.5 `geometry/render.py`

```python
@dataclass(frozen=True)
class FisheyeCamera:
    f: float            # 像素，以輸出解析度 S 為準
    xi: float
    size: int = 224     # S
    circular: bool = False

    # r_ref、cx、cy 全部由 size 與 circular 推導，不是獨立欄位，
    # 以確保相機物件不可能持有互相矛盾的組合。
    @property
    def r_ref(self) -> float: ...   # full-frame: S/√2；circular: S/2

    @property
    def cx(self) -> float: ...      # (S-1)/2，cy 同

    @property
    def cy(self) -> float: ...

def render_fisheye(pano: np.ndarray, cam: FisheyeCamera, R: np.ndarray,
                   supersample: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """全景圖 → 魚眼影像 (S, S, 3) uint8 與有效遮罩 (S, S) bool。
    無效像素與圓形魚眼 r > r_ref 的像素填 0。"""

def fisheye_to_perspective(img: np.ndarray, cam: FisheyeCamera,
                           out_size: int, fov_deg: float) -> np.ndarray:
    """用給定內參把魚眼還原成透視圖（視覺化與 error analysis 用）"""
```

### 3.6 `data/sampling.py`

```python
@dataclass(frozen=True)
class CameraSample:
    f: float
    xi: float
    fov_deg: float          # 在 r_ref 處的全角 FOV
    circular: bool
    r_ref: float
    yaw_deg: float
    pitch_deg: float
    roll_deg: float

def sample_camera(rng: np.random.Generator, cfg: SamplingConfig) -> CameraSample: ...
def sample_seed(global_seed: int, split: str, asset_id: str, view_idx: int) -> int:
    """以穩定雜湊（例如 hashlib.sha256，不可用 Python 內建 hash()）產生每個樣本的 seed"""
```

### 3.7 `data/download_polyhaven.py`（CLI）

```
python -m data.download_polyhaven --out data_root/raw/polyhaven --res 2k [--limit N]
```
- API：`https://api.polyhaven.com/assets?t=hdris`（清單與分類）、`https://api.polyhaven.com/files/{id}`（檔案 URL、MD5、大小）。
- 必須設定可識別的 User-Agent（例如 `fisheye-calib-research/0.1 (+github repo url)`）。
- 下載格式優先順序：該解析度的 tonemapped JPG；若不存在，下載 2K `.hdr` 並在本地 tonemap（Reinhard + gamma 2.2）轉成 8-bit JPG。**先實際查詢 `/files/{id}` 回傳結構確認，再實作。**
- 支援續傳（已存在且 MD5 相符就跳過）與請求間隔（≥ 0.2 秒）。
- 輸出 `manifest.json`：每筆包含 `asset_id, categories, width, height, md5, source_url, download_format`。

### 3.8 `data/split.py`（CLI）

```
python -m data.split --manifest data_root/manifest.json --out data_root/splits.json --seed 0
```

`splits.json` 結構：
```json
{
  "seed": 0,
  "train": ["asset_id", "..."],
  "val":   ["..."],
  "test":  ["..."],
  "train_subsets": { "p10": ["..."], "p30": ["..."], "p100": ["..."] }
}
```

### 3.9 `data/render_dataset.py`（CLI）

```
python -m data.render_dataset --config configs/data.yaml --split train --workers 8
```

`labels.csv` 欄位（每列對應 `images.npy` 同 index）：
```
index, asset_id, view_idx, f, xi, fov_deg, circular, r_ref, yaw_deg, pitch_deg, roll_deg, category, texture
```

- `texture`：渲染影像轉灰階後，**在有效遮罩內** Sobel 梯度強度（`sqrt(gx² + gy²)`，
  `ksize=3`）的平均值，uint8 尺度。值低代表畫面幾乎是平坦的天空或白牆。
  **只記錄，不用於過濾**——是否該排除低紋理樣本由 Phase 2 的 error analysis 決定，
  而不是先驗地丟掉資料。

`meta.json`：完整 data config、seed、git commit hash、生成時間、`S`、拒絕取樣的拒絕率、**train split 的 label 標準化統計量**（見 4.7）。

### 3.10 `datasets/fisheye_dataset.py`

```python
class FisheyeDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, split: str, subset: str | None = None,
                 transform: Callable | None = None,
                 norm_stats: dict | None = None): ...
    def __getitem__(self, i) -> dict:
        # {"image": FloatTensor(3,S,S)（ImageNet mean/std 正規化）,
        #  "target": FloatTensor(2)（標準化後的 [log f, ξ]）,
        #  "f": float, "xi": float, "r_ref": float, "circular": bool,
        #  "theta_ref": float, "asset_id": str, "category": str}
```
- `subset` 用 `splits.json` 的 `train_subsets` 依 `asset_id` 過濾 index。
- `norm_stats` 預設讀 train 的 `meta.json`；val/test **必須**使用 train 的統計量。

### 3.11 `datasets/online_dataset.py`

```python
class OnlineFisheyeDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, asset_ids: list[str], views_per_pano: int,
                 sampling_cfg: SamplingConfig, base_seed: int,
                 transform: Callable | None = None, norm_stats: dict | None = None): ...
    def __getitem__(self, key) -> dict: ...   # key 為 (epoch, index) 或 int，見 4.12.1
```
- 全景圖**不在初始化時讀入記憶體**：spawn 模式下每個 worker 都會複製一份
  （755 張約 1.2 GB × worker 數）。改為一次性前處理成單一 `panos.npy`
  （uint8, `(N, 512, 1024, 3)`），dataset 以 memmap 延遲開啟，各 worker 共用作業系統的 page cache。
  建立方式：`python -m datasets.online_dataset --root data_root --split train`。
- 每個樣本的 seed = `sample_seed(base_seed + epoch, "train_online", asset_id, view_idx)`，確保同 epoch 可重現、不同 epoch 不同。
  其中 `epoch` 來自 `__getitem__` 的 key（見 4.12.1），不是 dataset 的狀態。

### 3.12 `metrics/geometric.py`（torch，batched，可微分）

```python
def radius_from_theta(theta: Tensor, f: Tensor, xi: Tensor, eps: float = 1e-6) -> Tensor: ...

def denormalize(pred: Tensor, norm_stats: dict) -> tuple[Tensor, Tensor]:
    """(B,2) 標準化輸出 → (f, ξ)，f = exp(log f)"""

def reprojection_error_px(f_pred, xi_pred, f_gt, xi_gt, theta_ref, n_theta: int = 64,
                          r_cap: float = 224.0) -> tuple[Tensor, Tensor]:
    """回傳 (per-sample 平均像素誤差 (B,), per-sample invalid 比例 (B,))。評估用，不可微分也可。"""

def geometric_loss(pred, target_f, target_xi, theta_ref, norm_stats,
                   n_theta: int = 32, beta: float = 1.0, size: int = 224) -> Tensor:
    """可微分的重投影 loss（見 4.8）"""

def param_l1_loss(pred: Tensor, target: Tensor) -> Tensor:
    """標準化空間的 L1"""
```

### 3.13 [Phase 2] 訓練介面（先定義，暫不實作）

`configs/train/*.yaml`：
```yaml
run_name: resnet34_pre_p100_geo_s0
model: resnet34            # 或 vit_small_patch16_224
pretrained: true
data: { root: data_root, subset: p100, mode: offline }   # mode: offline | online
loss: geometric            # geometric | param_l1
optim: { lr: 3.0e-4, weight_decay: 1.0e-4, warmup_steps: 500 }
schedule: cosine
total_steps: 6000          # 固定總步數，不以 epoch 計
batch_size: 128
precision: bf16
eval_every: 500
seed: 0
```
- 模型：`timm.create_model(name, pretrained=..., num_classes=2)`。CNN 與 ViT 共用同一支 `train.py`。
- CLI 只有模式類參數（`--config`、`--overfit-batch`、`--benchmark`），
  **不提供超參數覆寫**；所有超參數都必須來自 yaml，MLflow 記錄的是補齊預設值後的完整 config。

#### 3.13.1 兩個架構各自的超參數

資料、loss、schedule 與總步數在兩個 run 之間固定，**optimizer 不固定**。
拿調好的 CNN 去比沒調好的 ViT，量到的是調參而不是架構差異。

| | `resnet34` | `vit_small_patch16_224` |
|---|---|---|
| `optim.lr` | `3.0e-4` | **`1.0e-4`** |
| `optim.weight_decay` | `1.0e-4` | **`0.05`** |
| `optim.warmup_steps` | 500 | 500 |
| 記憶體格式 | `channels_last` | 預設（純 token matmul，換 layout 無益） |

ViT 在 ResNet 的 `3.0e-4` 下不穩：`--overfit-batch 300` 實測 loss 在 0.006–0.056 間跳動，
300 步後只到 ResNet 的 1/54 收斂程度（0.0115 vs 0.000213）。

#### 3.13.2 Weight decay 的排除規則

`build_optimizer` 把參數分成兩個 group：

- **不套 weight decay**（`weight_decay = 0`）：
  1. 所有 `ndim <= 1` 的參數——bias，以及 BatchNorm / LayerNorm 的 scale 與 shift。
     對它們施加 decay 等於把正規化層往恆等映射拉、並移動激活值的統計量。
  2. timm 模型 `no_weight_decay()` 回傳的參數名稱——ViT 的 `pos_embed`、`cls_token` 等。
     那是學到的位置結構而不是權重，壓向 0 只會破壞嵌入，沒有正則化效果。
- **套用 config 的 weight decay**：其餘所有參數（conv 與 linear 的 weight）。

ViT 的 decay 是 0.05，這條規則因此特別重要。
測試必須驗證：兩個 group 合計涵蓋所有參數、沒有重複，
且 bias 與 norm 參數確實落在 `weight_decay = 0` 的 group。
- MLflow（本機 SQLite backend）：記錄完整 config、每 `eval_every` 步的 train/val loss 與 val 重投影誤差、best checkpoint（以 val 重投影誤差為準）。
- `sweep.py`：依序執行一個 config 清單；若 MLflow 中已有同 `run_name` 且狀態 FINISHED 的 run 則跳過。
- `evaluate.py`：test 集的整體指標，並依 FOV 分組、`category`、`circular` 分層統計；輸出誤差最大的 k 張，以「預測參數還原 vs 真值參數還原」並排視覺化。

---

## 4. 關鍵演算法與邊界條件

### 4.1 UCM 公式

投影（射線 `X = (x, y, z)`，`d = ‖X‖`）：
```
u = f · x / (ξ·d + z) + cx
v = f · y / (ξ·d + z) + cy
```
以與光軸夾角 θ 表示的半徑：`r(θ) = f · sinθ / (ξ + cosθ)`。

反投影（`m = ((u−cx)/f, (v−cy)/f)`，`ρ² = mx² + my²`）：
```
η   = (ξ + sqrt(1 + (1 − ξ²)·ρ²)) / (ρ² + 1)
ray = (η·mx, η·my, η − ξ)        # 已是單位向量
```

`ξ = 0` 時退化為針孔模型 `r = f·tanθ`（測試必須驗證）。

### 4.2 有效範圍（最常出錯的地方）

`dr/dθ ∝ (ξ·cosθ + 1) / (ξ + cosθ)²`，由此得：

- **θ_max(ξ) = arccos(−min(ξ, 1/ξ))**；`ξ = 0` 時為 π/2。
  - `ξ ≤ 1`：θ → arccos(−ξ) 時 r → ∞。
  - `ξ > 1`：θ > arccos(−1/ξ) 後投影不再單調，**同一像素會對應兩個方向，必須視為無效**。
- 投影的 valid 條件：`ξ·d + z > 0` **且** θ < θ_max(ξ)。
- 反投影的 valid 條件：`1 + (1 − ξ²)·ρ² ≥ 0`，即 `ξ > 1` 時 `ρ ≤ 1/sqrt(ξ² − 1)`。此邊界剛好對應 θ_max，測試必須驗證兩者一致。
- 無效處不得產生 NaN 傳遞到輸出：先計算 valid mask，無效處的 sqrt 參數以 0 代入，最後以 mask 覆寫。

### 4.3 FOV ↔ f

在參考半徑 `r_ref` 處的全角 FOV 為 `Θ`，`θ = Θ/2`：
```
f = r_ref · (ξ + cosθ) / sinθ
```
- full-frame：`r_ref = S/√2`（對角線一半），整張影像都有內容。
- circular：`r_ref = S/2`，`r > r_ref` 的像素填 0。
- `fov_from_f` 為反函數，需數值求解（θ ∈ (0, θ_max) 上 r(θ) 單調，可用二分法）。

### 4.4 全景圖取樣

- 載入後縮到 **1024×512**（INTER_AREA），降低鋸齒並加速。
- **水平環繞**：不可直接用 `cv2.BORDER_WRAP`（它也會讓垂直方向環繞，極區會出錯）。做法：在全景圖左右各 pad 若干欄（從另一側複製），`map_u` 相應平移，垂直方向用 `BORDER_REPLICATE`。
- `cv2.remap` 的 map 必須是 `float32`。

### 4.5 超取樣與抗鋸齒

輸出 224 像素涵蓋最多 220°，全景圖解析度遠高於輸出，直接雙線性取樣會嚴重鋸齒。

- 以 `s = supersample` 倍解析度渲染（**預設 s = 4**）：`S' = s·S`，`f' = s·f`，`cx' = s·cx + (s − 1)/2`（像素中心慣例下的正確換算）。
- 再以 `INTER_AREA` 縮回 `S`。有效遮罩也一併縮小，閾值 0.5。
- 測試必須驗證：同一組參數下，`supersample=1` 與 `supersample=2` 的有效遮罩在邊界以外一致，且關鍵方向的像素位置誤差 < 0.5 px。

### 4.6 相機參數取樣（`configs/data.yaml` 預設值）

```yaml
image_size: 224
supersample: 4
pano_resize: [1024, 512]
circular_prob: 0.5
fov_deg: [100, 220]        # 在 r_ref 處的全角
xi: [0.0, 1.5]
feasibility_margin: 0.05
f_min_px: 30
max_edge_stretch: 4.0
yaw_deg: [-180, 180]
pitch_deg: [-30, 30]
roll_deg: [-15, 15]
views_per_pano: { train: 30, val: 20, test: 20 }
split_ratio: [0.8, 0.1, 0.1]
train_subsets: { p10: 0.10, p30: 0.30, p100: 1.0 }
seed: 0
```

**拒絕取樣**：`(Θ, ξ)` 在矩形範圍內均勻取樣，滿足以下全部條件才接受，否則重抽：
1. `ξ + cos(Θ/2) ≥ feasibility_margin`（f 為正且不趨近 0）
2. 若 `ξ > 1`：`cos(Θ/2) ≥ −1/ξ + feasibility_margin`（在單調範圍內）
3. `f ≥ f_min_px`
4. **邊緣徑向放大倍率** `M ≤ max_edge_stretch`，其中（`θ = Θ/2`）

   ```
   M = (1 + ξ)(1 + ξ·cosθ) / (ξ + cosθ)²
   ```

   `M` 是視角邊緣與中心的徑向取樣率之比：由 4.2 的
   `dr/dθ ∝ f·(1 + ξ·cosθ)/(ξ + cosθ)²`，在 `θ = 0` 處為 `f/(1 + ξ)`，兩者相除即得。
   `M = 1` 表示邊緣與中心取樣率相同；`M` 很大表示邊緣被拉伸到佔掉大量像素，
   但那些像素承載的資訊極少，對內參回歸沒有幫助，還會讓影像看起來像壞掉的渲染。

   參考值（注意角度是全角 `Θ` 還是半角 `θ`）：

   | 鏡頭 | 參數 | 角度 | M |
   |---|---|---|---|
   | 直線（rectilinear）| ξ = 0 | 全角 Θ = 120° | **4.00** |
   | 等距（equidistant）| `r = fθ` | 任意 | **1.00** |
   | 立體投影（stereographic）| ξ = 1 | 半角 θ = 110°（全角 220°）| **3.04** |

   取 `max_edge_stretch = 4.0`，等於「不比 120° 直線鏡頭更誇張」。
   注意這條件主要限制**低 ξ 配寬 FOV**：ξ = 0 時 `M = 1/cos²θ`，
   因此全角上限恰為 120°；ξ ≥ 1 時在 Θ ≤ 220° 內幾乎不受限。
   加入此條件後，整體拒絕率約由 0.09–0.12 升到 0.29。

結果為可行域上的均勻分佈。**必須在 `meta.json` 記錄拒絕率，並由 `visualize_samples.py` 輸出 f、ξ、FOV 的直方圖與 (FOV, ξ) 散佈圖**，供人工確認分佈合理。

### 4.7 Label 標準化

- 回歸目標為 **`[log f, ξ]`**（f 跨度大，取 log 後較接近均勻，且反標準化後 f 必為正）。
- 以 **train split（p100）** 計算 mean/std，寫入 train 的 `meta.json`。所有 split、所有子集都用這組統計量。

### 4.8 重投影誤差與幾何 loss

主點相同且模型徑向對稱，所以重投影誤差化簡為一維的半徑比較：

```
θ_k = linspace(0, θ_ref_gt, n_theta)          # θ_ref_gt = GT 的 FOV/2（labels 內已有）
err = mean_k | r_pred(θ_k) − r_gt(θ_k) |
```

#### 4.8.1 有效性判定

模型預測出的 `ξ` 不受取樣器的範圍限制，可能是負的、也可能遠大於 1.5。
以兩個量判定某個 θ 對預測參數是否有效（`margin` 預設 0.05，與 4.6 的
`feasibility_margin` 同值）：

```
d1 = ξ + cosθ           # 投影分母，須為正
d2 = 1 + ξ·cosθ         # dr/dθ 的分子，須為正（半徑仍在遞增 → 投影單值）
valid = (d1 > margin) & (d2 > margin)
```

這組條件與 4.2 等價——`ξ > 1` 時 `d2 > 0` 就是 `cosθ > −1/ξ`——但兩者都是
`ξ` 的平滑函數，可以讓 barrier 施力。`ξ ≤ 0` 自然被 `d1` 擋掉，不需特例。

#### 4.8.2 評估：`reprojection_error_px`

- 無效的 θ_k 誤差記為 `r_cap`，並另外回報 invalid 比例。
- **有效的 θ_k 誤差也要 clamp 上限為 `r_cap`**。有效只要求分母大於 `eps`，
  而分母 1e-6 會產生上萬到百萬像素的半徑，一個接近退化的樣本就足以淹沒整批的平均。
  `r_cap` 是一個影像對角線量級：錯超過這個距離就只是「錯了」，數字再大也不帶額外資訊。

#### 4.8.3 訓練：`geometric_loss`

```
loss = radial + barrier_weight · barrier          # barrier_weight 預設 1.0
```

- **radial**：`r_pred` 的分母用 `torch.where(valid, d1, 1.0)`（**不是 clamp**），
  只在 `valid` 的位置計算 smooth L1（`beta = 1 px`），以 **valid 的數量**平均
  （數量為 0 時該項為 0），最後除以 `S` 讓數值落在 O(1)。
- **barrier**：`mean(relu(margin − d1) + relu(margin − d2))`，對**所有** θ_k 平均。
  完全有效時為 0，一旦任一量低於 `margin` 就線性成長。

**為什麼要拆成兩項**：先前的版本把分母 clamp 到 `eps` 並宣稱「invalid 點保留梯度把預測拉回有效區」，
這是**錯的**。`clamp` 在被夾住的區域是常數，梯度恰好為 0——正好在最需要修正訊號的地方沒有訊號。
遮罩掉 invalid 點同樣沒有梯度，所以真正把預測拉回來的是 barrier：
例如 `ξ` 過小時 `∂relu(margin − d1)/∂ξ = −1`，梯度下降因此會把 `ξ` 調高。

#### 4.8.4 測試必須驗證

- pred = GT 時 loss = 0（radial 與 barrier 皆為 0）。
- torch 版 `radius_from_theta` 與 numpy 版結果一致（容差 1e-4）。
- 預測為 N(0,1) 的隨機標準化輸出（B = 256、固定 seed、取樣器產生的 GT）時，
  loss 有限且 < 10，所有梯度的絕對值 < 100。
- 預測 `ξ` 遠小於 0 時，`ξ` 通道的梯度為負（梯度下降會讓 `ξ` 增加），
  且 `log f` 通道的梯度絕對值 < 10。
- `reprojection_error_px` 的 per-sample 誤差恆 ≤ `r_cap`。

### 4.9 資料增強規則

- **允許（不改變 label）**：亮度、對比、gamma、高斯模糊、高斯雜訊、JPEG 壓縮、隨機強度的暗角；**水平翻轉、垂直翻轉、90° 旋轉**（主點居中、方形影像、徑向對稱模型下內參不變）。
- **禁止**：random crop、resize、scale、平移、任意角度旋轉、長寬比改變。這些都會改變內參，造成 label 錯誤。
- 增強只在 train 使用；val/test 不做任何增強。

### 4.10 切分與洩漏防護

**以「組」為單位切分**，不是以單張全景圖為單位。Poly Haven 有大量同一地點、
同一次拍攝的變體（不同時段、不同天氣、`_puresky` 版本、編號系列），
若只以 asset 為單位切分，同一個場景會同時出現在 train 與 test，造成洩漏。

#### 4.10.1 排除清單

- config 的 `exclude_categories` 依 **`primary_category`** 比對，預設 `["Pure Skies"]`。
- **排除以「asset」為單位**：只有命中的 asset 本身不進入 train/val/test，
  改放進 `splits.json` 的 `"stress"` 清單，供 Phase 2 當作分佈外（OOD）壓力測試集。
- 分組仍以**全部** asset 計算；命中排除的成員移出後，
  **組內其餘成員照常以組為單位**分配到 train/val/test。

**已知且刻意接受的重疊**：`_puresky` 版本與它的兄弟是同一次拍攝，
因此多數 stress asset 會有兄弟留在 train（預設設定下 59 張全部如此）。
這個重疊是刻意保留的，理由是：

- **label 與場景內容無關**。回歸目標是相機內參 `(f, ξ)`，由取樣器獨立決定，
  與全景圖畫的是什麼完全無關。同一片天空出現在 train 與 stress，
  並不會洩漏任何一張 stress 影像的 `(f, ξ)` 答案——
  這與分類／偵測任務中「同場景洩漏」的意義不同。
- **整組排除的代價太高**。以預設的 `["Pure Skies"]` 計，
  整組排除會額外把 65 張非天空影像移出核心 split（train/val/test 從 938 降到 873），
  而這些影像的內參標註完全正常、沒有理由丟棄。

- 為了能**量化**這個重疊的影響，`splits.json` 另外輸出
  **`stress_sibling_split`**：`{stress asset_id: 其兄弟所在的 split 或 null}`。
  Phase 2 評估時應依此欄位分層比較
  （兄弟在 train 的 stress asset vs 兄弟為 null 的 stress asset），
  若兩者誤差無顯著差異，即可實證支持上述假設。
- 哪些分類要排除由人工決定（參考 `tools/report_categories.py` 的輸出），代理不自行增刪。

#### 4.10.2 分組規則

一個 asset 的分組 key 依序決定：

1. **collection**：flat `categories` 中形如 `"collection: xxx"` 的標籤，同一個 collection 為一組。
2. **stem**：其餘 asset 取 `asset_id` 反覆去除結尾的
   純數字段（`_01`、`_2`）、解析度字尾（`_2k`）、太陽仰角字尾（`_28d`）、
   以及時段／天氣類字尾（`_dawn`、`_sunset`、`_night`、`_puresky`、`_partly_cloudy` 等，
   完整清單見 `data/split.py` 的 `STRIP_TOKENS`）後剩下的 stem。

**同一組的 asset 必須落在同一個 split。** 兩個關係都是等價關係，
因此實際分組取兩者的**連通分量**（union-find）：collection 相同或 stem 相同即同組。
這一步是必要的——例如 `kloofendal_43d_clear` 屬於 `collection: verdant_trail`
而 `kloofendal_43d_clear_puresky` 不屬於，但兩者是同一次拍攝，
若只用「collection 優先」的單一 key 就會被拆到不同 split。

#### 4.10.3 分層與比例

- 以 **`primary_category`** 分層；一組若跨多個分類，取組內**眾數**（平手時取字典序最小）作為該組的層。
- 以「**組**」為單位，依 `split_ratio` 分配 train/val/test（預設 80/10/10）。
- 每個分層在每個 split **至少分到 1 組**（該分層的組數 ≥ 3 時）。

#### 4.10.4 資料量子集

- `train_subsets`（`p10 ⊂ p30 ⊂ p100`）同樣**以組為單位、依分層、巢狀**。

#### 4.10.5 測試必須驗證

- 三個 split 的 `asset_id` 兩兩交集為空。
- `stress` 與三個 split 兩兩交集為空。
- **同一組的 asset 不跨 split。**
- 子集巢狀關係成立：`p10 ⊆ p30 ⊆ p100`。

### 4.11 並行與可重現性

- `render_dataset.py` 以全景圖為單位分派給 worker（每個 worker 讀一次全景圖，渲染其所有視角），worker 內 `cv2.setNumThreads(0)`。
- 每個樣本的 seed 由 `sample_seed(global_seed, split, asset_id, view_idx)` 決定，與 worker 數量、執行順序無關。
- 寫入 `images.npy` 時使用預先配置的 `np.lib.format.open_memmap`，依 index 寫入；最後依 index 排序輸出 `labels.csv`。
- 測試必須驗證：以不同 `--workers` 數量生成同一個小資料集，結果逐位元相同。

### 4.12 DataLoader 注意事項

- memmap 必須在 worker 內延遲開啟（第一次 `__getitem__` 時），不可在 `__init__` 開啟後被 pickle 到各 worker。
- online 版必須提供 `worker_init_fn`（worker 內 `cv2.setNumThreads(0)`）。

#### 4.12.1 epoch 必須由 sampler 傳遞，不能存在 dataset 上

`persistent_workers=True` 的 DataLoader 會讓 worker 行程**跨 epoch 存活**，
每個 worker 持有自己那份 pickle 過的 dataset。
因此在訓練迴圈呼叫 `dataset.set_epoch(e)` **只會改到主行程那一份**，
worker 永遠停留在 epoch 0——增強與線上取樣每個 epoch 完全相同。
這個錯誤不會報錯、不會變慢，只是訓練從此看不到新的增強。

正確做法：sampler 在**主行程**被迭代，它吐出的索引才會送進 worker，
所以把 epoch 跟著索引一起送：

- `datasets/sampler.py` 的 **`EpochSampler(num_samples, shuffle, seed)`** 提供 `set_epoch(epoch)`，
  `__iter__` 依 `(seed, epoch)` 建立 `torch.Generator` 決定順序，產生 **`(epoch, index)`** tuple。
- `FisheyeDataset` 與 `OnlineFisheyeDataset` 的 `__getitem__` 接受 `(epoch, index)`，
  **以參數中的 epoch** 決定增強與取樣的 seed。兩者都**不再有** `set_epoch` 或 `_epoch` 狀態。
- 仍接受單純的 int index（epoch 視為 0），所以不帶 sampler 的 `dataset[i]` 照常可用。

- 測試必須驗證：`persistent_workers=True`、`num_workers=2` 的 DataLoader 跑兩個 epoch，
  同一個 index 在兩個 epoch 的增強結果**不同**，且以同樣的 seed 重跑時**逐位元相同**。

---

## 5. Phase 1 驗收條件

全部以 `pytest` 自動化，另加一項人工目視。

| # | 測試 | 條件 |
|---|---|---|
| 1 | project ∘ unproject | 有效區內像素 round-trip 誤差 < 1e-4 px，ξ ∈ {0, 0.5, 1.0, 1.5} |
| 2 | unproject ∘ project | θ < θ_max 的射線 round-trip 角度誤差 < 1e-6 rad |
| 3 | 針孔退化 | ξ = 0 時 `r = f·tanθ` |
| 4 | θ_max 一致性 | 反投影有效邊界 ρ_max 對應的 θ 等於 `theta_max(ξ)` |
| 5 | FOV ↔ f | `fov_from_f(f_from_fov(Θ)) = Θ`，誤差 < 1e-6° |
| 6 | 旋轉語意 | pitch = +30° 時，影像中心方向在全景圖上位於緯度 +30°；yaw 同理 |
| 7 | **渲染 round-trip** | 以「顏色編碼方向」的合成全景圖渲染魚眼，從顏色解回方向，與 `unproject` 的射線比較，中位數角度誤差 < 0.3° |
| 8 | 超取樣一致性 | 見 4.5 |
| 9 | 取樣可行性 | 1 萬次取樣全部滿足 4.6 的三個條件 |
| 10 | 切分無洩漏 | 見 4.10 |
| 11 | 可重現 | 見 4.11 |
| 12 | Dataset | 回傳 shape、dtype 正確；val 使用 train 的 norm_stats；禁止的增強不存在於 transform |
| 13 | metrics | 見 4.8 |
| 14 | 人工目視 | `visualize_samples.py` 輸出 8×8 樣本網格（涵蓋不同 FOV、ξ、circular）與參數分佈圖，由人確認 |

---

## 6. 不要做的事

- 不要手刻 CNN 或 ViT 架構，使用 timm。
- 不要在 `geometry/` 以外的地方重寫投影公式。
- 不要對魚眼影像做任何會改變內參的幾何增強（見 4.9）。
- 不要以單一影像為單位切分資料。
- 不要使用 Python 內建 `hash()` 產生 seed（每次執行結果不同）。
- 不要把資料、渲染結果、checkpoint、MLflow 資料庫 commit 進 git。
- 不要引入任何公司內部資源、路徑或服務名稱。
- Phase 1 未驗收前，不要開始寫 `train.py`。

---

## 7. 環境

- Python 3.11、PyTorch 2.x（CUDA）、timm、opencv-python、numpy、pyyaml、mlflow、pytest、matplotlib、requests。
- 版本鎖定於 `requirements.txt`。
- 所有幾何運算向量化（numpy/torch），禁止逐像素的 Python 迴圈。
