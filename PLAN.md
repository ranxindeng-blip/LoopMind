# LoopMind — 正式项目规划文档

> INDENG 242B Final Project · Due: 2026-05-11  
> 更新: 2026-04-25

---

## 一、项目定义

**任务**: 跨模态检索——输入 MIDI 旋律，从音频库中检索风格兼容的伴奏 stem（鼓、贝斯、钢琴、吉他）  
**核心方法**: 对比学习（InfoNCE Loss）训练双编码器，将 MIDI 和音频映射到共享嵌入空间  
**数据集**: Slakh2100（~2100 tracks）  
**Demo**: 已完成（BabySlakh 20 tracks + MLP，仅作展示用）  

---

## 二、目录结构

```
LoopMind/                          ← Git 仓库根目录（本地 Mac + GitHub）
├── PLAN.md                        ← 本文件
├── data/
│   ├── extract_pairs.py           ← 扫描 Slakh，生成 melody/stem 配对
│   ├── features.py                ← 特征提取（piano roll、mel spectrogram）
│   └── dataset.py                 ← PyTorch Dataset
├── models/
│   ├── query_encoder.py           ← MIDI 编码器（CNN + Transformer）
│   ├── audio_encoder.py           ← 音频编码器（CNN + Transformer）
│   └── dual_encoder.py            ← 整合两个编码器，输出嵌入
├── losses/
│   └── infonce.py                 ← InfoNCE / Category-conditioned loss
├── train.py                       ← 训练主脚本
├── build_library.py               ← 预计算所有 stem 的嵌入，生成检索库
├── evaluate.py                    ← R@1 / R@5 / R@10 评估 + loss 曲线
├── demo/
│   ├── app.py                     ← Gradio demo（已完成）
│   └── presets/                   ← 4 个预设 MIDI 文件
└── notebooks/
    └── LoopMind_Colab.ipynb       ← Colab 主 notebook（所有训练步骤）
```

**Google Drive 存储**（不进 Git，体积太大）:
```
MyDrive/LoopMind/
├── slakh2100/                     ← 原始数据集（~100GB）
│   ├── train/   (1500 tracks)
│   ├── validation/ (75 tracks)
│   └── test/    (125 tracks)
├── cache/
│   ├── pairs.json                 ← extract_pairs 输出
│   ├── features/                  ← 各 track 的 .npy 特征文件
│   └── library.pt                 ← build_library 输出（stem 嵌入库）
└── checkpoints/
    ├── last.pt                    ← 每 epoch 覆盖
    └── best.pt                    ← 最佳 val recall 时保存
```

---

## 三、运行位置规则

| 步骤 | 在哪里跑 | 原因 |
|------|----------|------|
| 写代码 / 改代码 | **本地 Mac** (VS Code) | 编辑器体验好 |
| git push/pull | **本地 Mac** (Terminal) | 标准流程 |
| 数据下载 / 解压 | **浏览器 Colab** | 直写 Drive，速度快 |
| 特征提取 | **浏览器 Colab** | 读 Drive 数据，GPU 加速 |
| 模型训练 | **浏览器 Colab** | T4 GPU，checkpoint 存 Drive |
| 评估 | **浏览器 Colab** | 读 Drive checkpoint |
| build_library | **浏览器 Colab** | 读 Drive checkpoint + 数据 |
| Demo 展示 | **浏览器 Colab** | share=True 生成公开 URL |

> ⚠️ VS Code Colab 插件无法挂载 Google Drive（dfs_ephemeral 报错），凡是需要读写 Drive 的步骤一律用**浏览器 Colab**。

---

## 四、数据集结构（Slakh2100）

```
slakh2100/train/Track00001/
├── MIDI/
│   ├── S01.mid   ← 各乐器 MIDI
│   ├── S02.mid
│   └── ...
├── stems/
│   ├── S01.wav   ← 与 MIDI 对应的音频 stem
│   ├── S02.wav
│   └── ...
└── metadata.yaml ← 记录每个 stem 的乐器类型（plugin_name / inst_class）
```

**旋律识别逻辑**: 非打击乐、非贝斯的 MIDI 轨中，平均音高最高的轨作为旋律  
**类别映射**: `drums` / `bass` / `piano`（含 keys/organ） / `guitar`（含弦乐 texture）

---

## 五、特征设计（全版，有时序）

### Demo 版（已完成，无时序）
- MIDI → chroma mean+std → [24-d] 向量
- Audio → chroma/mel mean+std → [24-d] 或 [256-d] 向量

### 正式版（有时序，送进 CNN+Transformer）

**旋律输入（MIDI → Piano Roll 序列）**:
- MIDI → pretty_midi.get_piano_roll(fs=50) → [128, T]
- 转置 → [T, 128]，截取/补零到固定长度 T=256 帧（约 5 秒 @50fps）
- 只保留有音符的音高区间，归一化到 [0,1]

**音频输入（Stem → Mel Spectrogram 序列）**:
- librosa.feature.melspectrogram(n_mels=128, hop_length=512, sr=22050)
- log 压缩 → dB scale
- [T_mel, 128]，截取/补零到 T=256 帧

两路特征都是 **[Batch, T, 128]** 张量送入编码器。

---

## 六、模型架构（正式版）

### QueryEncoder（MIDI 输入）
```
输入: [B, T=256, 128]  (piano roll 序列)
  │
  ├─ Conv1D(128→256, kernel=3, padding=1) + ReLU + LayerNorm
  ├─ Conv1D(256→256, kernel=3, padding=1) + ReLU + LayerNorm
  ├─ Dropout(0.1)
  │
  ├─ TransformerEncoder(d_model=256, nhead=4, num_layers=2, dropout=0.1)
  │   (处理时序依赖)
  │
  ├─ Global Average Pooling → [B, 256]
  │
  └─ 4 个并行 Linear head（每个类别一个）: 256 → embed_dim(128)
      → L2 归一化
      → dict: {drums: [B,128], bass: [B,128], piano: [B,128], guitar: [B,128]}
```

### AudioEncoder（音频 Stem 输入，每类别独立）
```
输入: [B, T=256, 128]  (mel spectrogram)
  │
  ├─ Conv1D(128→256, kernel=3) + ReLU + LayerNorm  ×2
  ├─ TransformerEncoder(d_model=256, nhead=4, num_layers=2)
  ├─ Global Average Pooling → [B, 256]
  └─ Linear(256 → 128) + L2 归一化
```

4 个类别分别有独立的 AudioEncoder（参数不共享）。

### 损失函数
```
对每个类别 c：
  L_c = InfoNCE(query_emb_c, audio_emb_c, temperature=0.07)

总损失：
  L = L_drums + L_bass + L_piano + L_guitar
```

---

## 七、训练流程

### 准备（浏览器 Colab Cell 1-3）
```python
# Cell 1: 挂载 Drive + 克隆仓库
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/ranxindeng-blip/LoopMind /content/LoopMind

# Cell 2: 安装依赖
!pip install pretty_midi librosa soundfile gradio pydub

# Cell 3: 提取配对关系（运行一次，结果存 Drive）
import sys; sys.path.insert(0, '/content/LoopMind')
from data.extract_pairs import extract_pairs
records = extract_pairs(
    data_root='/content/drive/MyDrive/LoopMind/slakh2100',
    cache_dir='/content/drive/MyDrive/LoopMind/cache'
)
print(f"配对数: {len(records)}")
```

### 特征提取（Colab Cell 4，较耗时，运行一次即可）
```python
# 提取所有特征并缓存到 Drive（约 30-60 min）
from data.features import extract_and_cache
features = extract_and_cache(
    records,
    cache_dir='/content/drive/MyDrive/LoopMind/cache'
)
```

### 训练（Colab Cell 5）
```python
# python train.py \
#   --data_root /content/drive/MyDrive/LoopMind/slakh2100 \
#   --cache_dir /content/drive/MyDrive/LoopMind/cache \
#   --ckpt_dir  /content/drive/MyDrive/LoopMind/checkpoints \
#   --epochs 100 --batch_size 64 --lr 1e-4 --embed_dim 128

# 预计时间：T4 GPU，约 2-4 小时（100 epochs）
```

### 评估（Colab Cell 6）
```python
# python evaluate.py \
#   --ckpt_path /content/drive/MyDrive/LoopMind/checkpoints/best.pt \
#   --data_root /content/drive/MyDrive/LoopMind/slakh2100 \
#   --cache_dir /content/drive/MyDrive/LoopMind/cache
```

### 构建检索库（Colab Cell 7）
```python
# python build_library.py \
#   --data_root    /content/drive/MyDrive/LoopMind/slakh2100 \
#   --cache_dir    /content/drive/MyDrive/LoopMind/cache \
#   --ckpt_path    /content/drive/MyDrive/LoopMind/checkpoints/best.pt \
#   --library_path /content/drive/MyDrive/LoopMind/cache/library.pt
```

### 启动 Demo（Colab Cell 8）
```python
# 与 demo 版相同，加载 best.pt + library.pt → Gradio launch(share=True)
```

---

## 八、评估指标

| 指标 | 含义 |
|------|------|
| R@1 | top-1 是 ground truth 的比例（最严格）|
| R@5 | top-5 中包含 ground truth 的比例 |
| R@10 | top-10 中包含 ground truth 的比例 |
| Random R@1 | 基线：1/N（随机猜中概率）|

目标：各类别 R@5 > 0.5（比 BabySlakh demo 的 0.93 要低，因为数据量大 100 倍）

---

## 九、开发顺序建议

- [ ] **Step 1** — 重构 `data/features.py`：改为输出时序 [T, 128] 而非 mean+std
- [ ] **Step 2** — 重构 `data/dataset.py`：适配新特征格式 + Slakh2100 完整路径
- [ ] **Step 3** — 重写 `models/dual_encoder.py`：CNN + Transformer 架构
- [ ] **Step 4** — 更新 `train.py`：适配新 batch 格式（时序维度）
- [ ] **Step 5** — Colab 跑特征提取（Cell 4，约 1 小时）
- [ ] **Step 6** — Colab 跑训练（Cell 5，约 3 小时）
- [ ] **Step 7** — 评估 + 调参（如有时间）
- [ ] **Step 8** — build_library + demo 更新展示

---

## 十、与 Demo 版的对比

| 项目 | Demo 版（已完成）| 正式版 |
|------|-----------------|--------|
| 数据集 | BabySlakh（20 tracks）| Slakh2100（~2100 tracks）|
| 特征 | chroma mean+std [24-d] | piano roll / mel 序列 [T, 128] |
| 模型 | MLP | CNN + Transformer |
| embed_dim | 64 | 128 |
| 训练时间 | ~30 min | ~3 小时 |
| 预期 R@5 | 0.93（小数据集虚高）| 目标 > 0.5 |
| 展示方式 | Gradio（已上线）| 同上 |
