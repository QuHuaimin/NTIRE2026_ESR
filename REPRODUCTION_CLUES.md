# SPANV2 指标复现：证据与实验路线

## 1. 当前差距

按挑战官方 RGB PSNR 协议评测，公开 Team 22 权重在 validation 集得到 26.9187 dB，与报告
26.92 dB 一致；当前直接训练提交态网络的 Stage 1 为 26.6777 dB，Stage 2 最好结果为
26.6723 dB。因此评测链路已对齐，约 0.24 dB 的差距主要来自训练实现，而不是指标脚本。

## 2. 论文中能确认的连续谱系

SPAN 论文明确说明训练使用结构重参数化。官方训练代码里的 `Conv3XC` 不是单个 3x3 卷积，
而是 `1x1 扩张 -> 3x3 -> 1x1` 分支加 `1x1` shortcut；部署前才融合成一个 3x3 卷积。
SPAN 的论文消融中，REP 相比直接 3x3 在 Set5/Set14/B100/Urban100 分别提高约
0.07/0.04/0.02/0.08 dB。

2025 SPANF 的公开提交代码只保留融合后的 3x3 卷积，但构造函数仍保留未参与推理的
`gain1=2` 参数。2026 SPANV2 提交模型也具有相同痕迹：`Conv3XC` 已缩成单个 3x3，SPABV2
仍用 `gain1=2` 调用它。结合 SPANV2 报告明确称 SPABV2 扩展自 SPAN 的 SPAB，最强的工程
推断是：SPANF 和 SPANV2 同样以宽分支训练，再将其融合成提交模型。

为验证该推断，本项目新增 `SPANV2ESRRep`。它只替换 15 个训练态 `Conv3XC`，SPABV2、
near-pixel 分支、DW/PW 融合和 PixelShuffle 均不变；导出后严格兼容官方 139,104 参数拓扑。
这仍是有代码谱系支撑的推断，不是 SPANV2 作者公开确认的训练源码。

## 3. 仍未公开的变量

| 变量 | 已知信息 | 当前实现/风险 |
|---|---|---|
| 八种 crop | 256 至 512，含方形、横向、纵向 | 使用 256/320/384/448/512 方形与三种矩形；精确官方列表未公布 |
| FFT 公式 | 权重 0.05 | 当前采用 SAFMN 实部/虚部 L1，但使用 `norm=ortho`；官方归一化方式未公布 |
| gradient 公式 | 权重 3.0 | 一阶差分、Sobel、L1/L2、方向平均方式未公布 |
| 全局 batch | 每 GPU 为 8，GPU 数未公布 | 当前按谱系线索假设全局为 64，单卡 micro-batch 8 累计 8 次 |
| 最终权重 | Stage 1/2 与 EMA 已公布 | 是否续训、换 seed 或参数融合未公布 |

用当前 Stage 1 EMA 在 8 个 512 crop 上做量级诊断：L1 为 0.02202，正交归一化 rFFT 的
实部/虚部 L1 为 0.01666，`L1 + 0.05 FFT` 中频域项约占 3.65%，量级合理。同一数据上的
复数模距离为 0.02618；未归一化 FFT 会把频域项放大两个数量级，因此是否使用 `ortho` 是
比 full/rFFT 更关键的未知变量。

相同 8 个样本的输出梯度诊断为 `g_pixel=0.001128`、`g_fft=0.00003994`、
`g_fft/g_pixel=3.54%`、余弦相似度 `+0.432`。这表明当前正交 SAFMN 距离是较弱且与 L1
部分同向的辅助项；判断依据应以训练期间 W&B 曲线为主，而不是只看这一组离线样本。

同一批样本上，MSE 为 0.001831，一阶差分 L1 两方向之和为 0.04582。当前 Stage 2 的
`5 MSE + 3 gradient` 中，gradient 加权项约占 93.8%。结合 Stage 2 没有提升 PSNR，gradient
定义是 REP 之后最需要验证的变量；但不能仅凭量级断言官方一定使用 L2 gradient。
输出梯度诊断进一步得到 `g_gradient/g_mse=41.14`、余弦相似度 `+0.555`，说明当前 Stage 2
定义在优化方向上也由 gradient 项主导。

## 4. 判别式实验顺序

1. **REP 单变量实验**：运行新增 Stage 1 配置，其他条件全部保持不变。先观察 10 万、20 万、
   30 万步的 DIV2K100 曲线，相对旧曲线持续领先再跑满 100 万步。
2. **Stage 1 loss 小消融**：若 REP 仍差超过 0.10 dB，保持实部/虚部 L1 不变，优先比较
   `norm=ortho` 与 SAFMN 默认的未归一化 FFT；再比较复数模距离。固定初始化和数据顺序，
   以 20 万步曲线筛选。
3. **全局 batch 64**：当前把它作为新的受控复现假设。单卡 micro-batch 为 8，梯度累计
   8 次后才更新 optimizer、EMA 和 scheduler；100 万 iteration 仍代表 100 万次参数更新。
   该设置会比旧 batch=8 实验处理 8 倍样本，也约需 8 倍训练时间。
4. **Stage 2 gradient 小消融**：只从最佳 Stage 1 初始化，对比当前一阶差分 L1 sum、方向平均
   L1、方向平均 L2；以官方 validation PSNR 作为唯一主判据。
5. **续训与参数融合**：SPAN 2024 和 SPANF 2025 都明确使用多轮加载续训，并对 L1/L2、
   batch 64/128 的四个微调模型做参数融合。只有前四项仍无法闭合差距时，才测试第二轮 cosine
   续训和 2 至 4 个 seed 的 EMA 参数平均；否则计算成本过高，也难区分真正原因。

模型融合必须平均同构训练态 checkpoint，再统一导出；不要把 REP 训练参数与 139K 部署参数
混合平均。官方 checkpoint 可以用于评测和结构核对，不应用作复现实验的教师或初始化，否则
得到的是蒸馏/迁移结果，不再是独立复现。

## 5. 命令

REP Stage 1：

```bash
cd /home/qhm/projects/SPANV2_official
bash scripts/start_rep_tmux.sh stage1
tmux attach -t spanv2-stage1-rep
```

查看 10 万步等阶段结果后，仍可让同一会话继续跑满。中断后按精确数据游标恢复：

```bash
bash scripts/start_rep_tmux.sh stage1 --resume-iter 100000
```

Stage 1 确认优于旧基线并跑满后，再启动 REP Stage 2：

```bash
bash scripts/start_rep_tmux.sh stage2
tmux attach -t spanv2-stage2-rep
```

将训练态 EMA 融合为官方 Team 22 格式：

```bash
python scripts/export_spanv2_rep.py \
  --input experiments/spanv2_stage2_rep_fd2k_report_gb64/models/net_g_1000000.pth \
  --output model_zoo/spanv2_rep_reproduced_gb64.pth \
  --param-key params_ema
```

复查候选损失量级：

```bash
python scripts/analyze_report_losses.py \
  --checkpoint experiments/spanv2_stage1_fd2k_report_gb64/models/net_g_1000000.pth
```

## 6. 通过标准

- 首要标准：同一官方 validation 数据与 RGB/crop-4 评测下达到 26.92 dB 的四舍五入区间；
- 部署态参数必须保持 139,104，FLOPs 保持约 9.11 G；
- REP 训练态与融合部署态随机输入最大误差应不超过 `2e-6` 量级；
- 最终报告同时保留 seed、GPU 数、全局 batch、最佳/最终 iteration 和 EMA/非 EMA 参数键。

达到数值指标不等于公开资料已证明每个隐藏细节完全相同，因此结论应区分“指标复现成功”与
“官方训练实现被逐项确认”。

## 7. 一手资料

- [SPAN 论文](https://arxiv.org/abs/2311.12770)及其[官方 BasicSR 代码](https://github.com/hongyuanyu/SPAN)
- [NTIRE 2024 Efficient Super-Resolution Challenge Report](https://arxiv.org/abs/2404.10343)
- [NTIRE 2025 Efficient Super-Resolution Challenge Report](https://arxiv.org/abs/2504.10686)及[官方提交仓库](https://github.com/Amazingren/NTIRE2025_ESR)
- [NTIRE 2026 Efficient Super-Resolution Challenge Report](https://arxiv.org/abs/2604.03198)及[官方提交仓库](https://github.com/Amazingren/NTIRE2026_ESR)
