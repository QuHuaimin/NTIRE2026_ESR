# SPANV2 BasicSR 训练复现指南

本项目以 [NTIRE 2026 Efficient Super-Resolution 官方仓库](https://github.com/Amazingren/NTIRE2026_ESR)
中的 Team 22（XiaomiMM）提交为基础，保留官方 SPANV2 网络、checkpoint、CUDA 算子和评测
入口，并使用 SPAN 作者维护的 BasicSR 1.4.2 分支补齐训练流程。

项目提供两条独立训练路线：

| 路线 | 训练网络 | 用途 | 启动脚本 |
|---|---|---|---|
| 单分支对照 | `SPANV2ESR` | 直接训练官方提交态拓扑，作为对照实验 | `start_stage1_tmux.sh`、`start_stage2_tmux.sh` |
| REP 多分支 | `SPANV2ESRRep` | 按 SPAN 谱系假设使用训练态 `Conv3XC`，再融合为提交态拓扑 | `start_rep_tmux.sh stage1\|stage2` |

**普通的 `start_stage1_tmux.sh` 不是 REP 训练。** 当前建议优先运行 REP 路线验证官方指标；
单分支路线用于衡量结构重参数化带来的增益。两条路线均采用 micro-batch 8、梯度累计 8 次，
单卡有效全局 batch 为 64。

## 1. 依据与边界

可以从公开资料确认：

- [SPAN 官方仓库](https://github.com/hongyuanyu/SPAN)明确使用 BasicSR，并将网络注册在
  `basicsr/archs`；
- SPANV2 与 SPAN 的主要作者一致，挑战报告说明 SPANV2 从 SPAN、SPANF 演进而来；
- SPAN 官方训练代码中的 `Conv3XC` 是多分支结构，部署前可融合为单个 3x3 卷积；
- NTIRE 2026 官方发布包只包含 SPANV2 推理模型、权重和算子，没有训练源码与完整配置。

因此，采用 BasicSR 和 REP 多分支训练都有明确的代码谱系依据，但仍属于工程复现。本文会把
报告明确公开的设置和本项目为补齐训练所作的假设分开说明，不能将新增代码视为 Team 22 原始
训练源码。

## 2. 项目结构

```text
/home/qhm/projects/SPANV2_official
├── models/team22_SPANV2_ESR.py       # Team 22 官方提交模型
├── model_zoo/team22_spanv2_c2.pth   # Team 22 官方 checkpoint
├── span_attention_op/                # 官方推理算子及本机兼容补丁
├── test_demo_team22.py               # Team 22 官方评测入口
├── basicsr/                          # SPAN 作者使用的 BasicSR 1.4.2
│   ├── archs/spanv2_esr_arch.py      # 官方提交态拓扑的 BasicSR 注册包装
│   ├── archs/spanv2_esr_rep_arch.py  # REP 多分支训练态及部署融合
│   ├── data/report_paired_image_dataset.py
│   ├── data/data_sampler.py          # 多尺寸与累计窗口采样
│   ├── losses/report_loss.py         # 两阶段报告损失
│   └── train.py                      # 梯度累计与精确断点恢复
├── configs/
│   ├── stage1_report.yaml            # 单分支 Stage 1
│   ├── stage2_report.yaml            # 单分支 Stage 2
│   ├── stage1_rep_report.yaml        # REP Stage 1
│   └── stage2_rep_report.yaml        # REP Stage 2
├── scripts/
│   ├── prepare_datasets.py           # 数据下载、LR 生成和 DF2K 软链接
│   ├── setup_environment.sh          # Conda 环境与 editable 安装
│   ├── verify_setup.py               # 模型、配置、损失和数据检查
│   ├── start_stage1_tmux.sh          # 单分支 Stage 1
│   ├── start_stage2_tmux.sh          # 单分支 Stage 2
│   ├── start_rep_tmux.sh             # REP Stage 1/2
│   └── export_spanv2_rep.py          # REP 参数融合与导出
└── REPRODUCTION_CLUES.md             # 证据、差距与消融优先级
```

数据统一放在 `/home/qhm/datasets`，项目根目录的 `datasets` 软链接指向该目录。

## 3. 环境配置

NTIRE 官方评测环境为 Python 3.9、PyTorch 1.13.1+cu117 和 RTX A6000。本机 CUDA 编译器为
11.8，因此训练环境使用 Python 3.8、PyTorch 2.4.1+cu118，使 PyTorch 与本地 CUDA 工具链
保持一致。正式运行时间仍需在官方 A6000 环境复测。

```bash
cd /home/qhm/projects/SPANV2_official
bash scripts/setup_environment.sh
conda activate spanv2_official
```

训练配置必须保持 `use_span_attn: false`，使用支持反向传播的纯 PyTorch 注意力路径。
`span_attention_op` 只用于部署测速；需要时按以下方式编译：

```bash
PYTHON=$(which python) bash span_attention_op/build_span_attn.sh
python -c "import span_attention; print(span_attention.__file__)"
```

本机 CUDA 兼容改动见 `span_attention_op/COMPATIBILITY.md`，未修改算子的数学计算与调度逻辑。

## 4. 数据准备

训练集为 DF2K，包括 DIV2K 训练集 800 张和 Flickr2K 2650 张，LR 采用 MATLAB 风格 bicubic
x4。准备脚本会复用已有文件，只下载或生成缺失部分，并通过软链接组成 DF2K，重复运行不会
重复下载数据。

```bash
conda activate spanv2_official
python scripts/prepare_datasets.py \
  --root /home/qhm/datasets \
  --download-flickr2k \
  --workers 4
```

```text
/home/qhm/datasets
├── DIV2K/HR                         # 0001-0900
├── DIV2K_bicubic/LR/X4             # 0001x4-0900x4
├── Flickr2K
│   ├── Flickr2K_HR
│   └── Flickr2K_LR_bicubic/X4
└── DF2K
    ├── HR                           # 800 + 2650 个软链接
    ├── LR/X4
    └── meta_info_DF2K.txt
```

如需复测挑战官方验证集：

```bash
python scripts/prepare_datasets.py \
  --root /home/qhm/datasets \
  --download-ntire-valid
```

## 5. 共同训练协议

两条路线只在卷积的训练态参数化上不同，数据、损失、优化器、调度器、EMA 和训练步数保持
一致，以便把 REP 作为单变量实验。

| 设置 | Stage 1 | Stage 2 |
|---|---|---|
| 初始化 | 随机初始化 | 加载同路线 Stage 1 第 100 万步的 `params_ema` |
| HR crop | 八种 256 至 512 的方形、横向和纵向尺寸 | 固定 512x512 |
| 损失 | `1.0 L1 + 0.05 FFT` | `5.0 MSE + 3.0 gradient` |
| AdamW 学习率 | `1e-3` | `5e-4` |
| 调度 | 100 万步 cosine | 60 万步 cosine，再以 0.5 权重运行 40 万步 |
| 最低学习率 | `1e-6` | `1e-6` |
| EMA | `0.999` | `0.999` |
| 数据增强 | 随机翻转和 90 度旋转 | 随机翻转和 90 度旋转 |

挑战报告只公开 batch 8/GPU，没有公开 GPU 数。本项目根据 SPAN、SPANF 的训练线索假设全局
batch 为 64，并在单卡上使用：

```yaml
datasets:
  train:
    batch_size_per_gpu: 8
train:
  gradient_accumulation_steps: 8
  total_iter: 1000000
```

一个 `iteration` 始终表示一次完整参数更新：依次读取 8 个 micro-batch，对各自 loss 除以
8 后反向传播，最后执行一次 `optimizer.step()`、EMA 和 scheduler 更新。因此每阶段仍为
100 万次参数更新，不能将总迭代数除以 8。与旧的有效 batch=8 实验相比，每个 iteration 的
计算量和总样本量约增加到 8 倍。

Stage 1 的同一累计窗口共享 crop 尺寸与旋转，使 64 个样本等价于一个可堆叠的多尺寸 batch。
每个 epoch 包含 345,024 个样本、43,128 个 micro-batch 和 5,391 次参数更新。

## 6. REP 多分支训练

这是当前用于闭合官方指标的推荐路线。训练阶段使用 `SPANV2ESRRep`，其中 15 个
`Conv3XC` 保留 SPAN 风格的多分支结构；其余 SPABV2、near-pixel 分支、DW/PW 融合和
PixelShuffle 不变。

### 6.1 Stage 1

```bash
cd /home/qhm/projects/SPANV2_official
bash scripts/start_rep_tmux.sh stage1
tmux attach -t spanv2-stage1-rep
```

输出目录：

```text
experiments/spanv2_stage1_rep_fd2k_report_gb64
```

### 6.2 Stage 2

Stage 1 完成并生成 `models/net_g_1000000.pth` 后运行：

```bash
bash scripts/start_rep_tmux.sh stage2
tmux attach -t spanv2-stage2-rep
```

Stage 2 自动从
`experiments/spanv2_stage1_rep_fd2k_report_gb64/models/net_g_1000000.pth`
加载 `params_ema`，输出到：

```text
experiments/spanv2_stage2_rep_fd2k_report_gb64
```

### 6.3 导出部署权重

```bash
python scripts/export_spanv2_rep.py \
  --input experiments/spanv2_stage2_rep_fd2k_report_gb64/models/net_g_1000000.pth \
  --output model_zoo/spanv2_rep_reproduced_gb64.pth \
  --param-key params_ema
```

导出会将多分支 `Conv3XC` 融合为单个 3x3 卷积，得到与官方提交模型兼容的 139,104 参数
部署拓扑。

## 7. 单分支对照训练

这条路线直接优化官方提交态 `SPANV2ESR`，不会启用 REP 多分支。它用于与 REP 结果做公平
对照，不是当前推荐的指标闭合路线。

```bash
cd /home/qhm/projects/SPANV2_official
bash scripts/start_stage1_tmux.sh
tmux attach -t spanv2-stage1
```

Stage 1 输出到 `experiments/spanv2_stage1_fd2k_report_gb64`。完成后运行：

```bash
bash scripts/start_stage2_tmux.sh
tmux attach -t spanv2-stage2
```

Stage 2 加载同路线 Stage 1 的 `params_ema`，输出到
`experiments/spanv2_stage2_fd2k_report_gb64`。

## 8. tmux 与断点恢复

三个启动脚本都会使用固定的 Conda Python、指定 GPU，并启用 `--auto_resume`。省略
`--resume-iter` 时自动选择当前实验目录中最新且模型与状态文件完整配对的断点；也可指定
某个保存迭代：

```bash
# REP 路线
bash scripts/start_rep_tmux.sh stage1 --resume-iter 650000

# 单分支路线
bash scripts/start_stage1_tmux.sh --resume-iter 650000
```

通用可选参数：

- `--resume-iter N`：从第 N 次参数更新恢复；
- `--wandb-mode resume|rewind|fork`：覆盖 YAML 中的 W&B 恢复模式；
- `--session NAME`：指定 tmux 会话名。

在 tmux 中按 `Ctrl+B`，松开后按小写 `d` 即可分离。如果快捷键被终端拦截，在另一个 WSL
终端执行：

```bash
tmux detach-client -s spanv2-stage1-rep
tmux detach-client -s spanv2-stage2-rep
tmux detach-client -s spanv2-stage1
tmux detach-client -s spanv2-stage2
```

checkpoint 只在完整的 8 次累计结束后保存。`.state` 同时记录 optimizer、scheduler、主进程
RNG、W&B Run ID、epoch 内已完成的 micro-batch 和数据配置签名。恢复时重建确定性 sampler，
快进已处理数据，再从下一个累计窗口开始，因此模型状态和数据游标对应同一个 iteration。

改变 micro-batch、累计次数、worker 数、数据扩大倍率、随机种子、crop 列表或 meta-info 后，
签名检查会拒绝精确恢复。旧的有效 batch=8 checkpoint 不能用于新的 batch=64 实验；四份配置
使用独立的 `_gb64` 目录，避免误恢复旧状态。

## 9. W&B 日志

W&B 是唯一训练可视化后端，四份配置均写入 `SPANV2` Project：

```bash
wandb login
wandb status
```

首次启动生成的 Run ID 保存在实验目录的 `wandb_run_id.txt`，并写入后续 `.state`。默认
`resume` 会保留历史并续写同一 Run；`rewind` 会截断恢复点后的记录；`fork` 会保留源
Run 并创建分支。后两种模式需要当前 W&B 账号具备相应服务端权限，使用前必须停止仍在写入
源 Run 的训练进程。

普通损失每 100 iteration 记录一次，是八个 micro-batch 的均值。输出梯度诊断每 1000
iteration 记录一次；为控制额外开销，仅分析该 iteration 的最后一个 micro-batch，不会修改
模型梯度。

| 阶段 | 损失字段 | 输出梯度字段 |
|---|---|---|
| Stage 1 | `l_pixel`、`l_fft`、`fft_scalar_fraction` | `grad_pixel_l2`、`grad_fft_l2`、`grad_fft_to_pixel_ratio`、`grad_pixel_fft_cosine` |
| Stage 2 | `l_mse`、`l_gradient`、`gradient_scalar_fraction` | `grad_mse_l2`、`grad_gradient_l2`、`grad_gradient_to_mse_ratio`、`grad_mse_gradient_cosine` |

`l_fft`、`l_mse` 和 `l_gradient` 已分别包含 0.05、5.0 和 3.0 权重；相应梯度范数也是
实际进入总损失的加权分量。

## 10. 未公开细节与本项目实现

| 未公开项 | 当前实现 |
|---|---|
| SPANV2 原始训练框架 | 根据作者代码谱系采用 BasicSR |
| 是否使用训练态 REP | 提供 REP 推荐路线和单分支对照路线，尚无官方源码直接确认 |
| 全局 batch | 假设为 64，单卡 micro-batch 8，累计 8 次 |
| 八种 crop 的精确列表 | 256/320/384/448/512 方形，加 256x384、384x512、512x384 |
| FFT loss | `norm="ortho"` 的 `rfft2`，实部和虚部分别计算 L1，权重 0.05 |
| Gradient loss | 水平、垂直一阶有限差分的 L1 之和，权重 3.0 |
| Stage 2 初始化 | 同路线 Stage 1 第 100 万步 checkpoint 的 `params_ema` |
| 本地验证协议 | DIV2K 0801-0900，RGB PSNR，裁去 4 像素边界 |

更完整的证据链、当前约 0.24 dB 差距、损失量级诊断和后续消融顺序见
[`REPRODUCTION_CLUES.md`](REPRODUCTION_CLUES.md)。

## 11. 验证与评测

检查环境、配置、模型拓扑、REP 融合、损失、梯度累计和数据：

```bash
python -m unittest -v tests.test_training_components
python scripts/verify_setup.py \
  --dataset-root /home/qhm/datasets \
  --require-flickr2k
```

使用 Team 22 官方融合算子评测：

```bash
CUDA_VISIBLE_DEVICES=0 python test_demo_team22.py \
  --model_id 22 \
  --data_dir /home/qhm/datasets/NTIRE2026_ESR \
  --save_dir results/team22
```

复现成功的首要标准是在相同 RGB/crop-4 协议下达到报告的 26.92 dB 四舍五入区间。REP 导出
模型还应保持 139,104 参数，且训练态与融合部署态的随机输入误差不超过 `2e-6` 量级。达到
指标只能说明数值复现成功，不能证明所有未公开训练细节均与官方实现完全一致。

## 12. 参考资料

- [SPAN 论文与官方 BasicSR 代码](https://github.com/hongyuanyu/SPAN)
- [NTIRE 2024 Efficient Super-Resolution Challenge Report](https://arxiv.org/abs/2404.10343)
- [NTIRE 2025 Efficient Super-Resolution Challenge Report](https://arxiv.org/abs/2504.10686)
- [NTIRE 2026 Efficient Super-Resolution Challenge Report](https://arxiv.org/abs/2604.03198)
- [NTIRE 2026 官方挑战仓库](https://github.com/Amazingren/NTIRE2026_ESR)
- [BasicSR 数据准备文档](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md)
