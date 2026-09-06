# ONEFLIP reproduction log

## Scope and safety boundary

This project reproduces the paper's **offline simulation**: an IEEE-754
single-bit change in a locally stored model weight, followed by evaluation of
the supplied trigger. It does not attempt Rowhammer, DRAM profiling, memory
placement, privilege escalation, or modification of a model outside this
workspace.

## Paper in one page

The paper studies inference-time DNN backdoors when the attacker knows the
model and has a small clean sample set, but does not control training. Its
central claim is that a full-precision model can retain clean behavior while
being made to map trigger-bearing inputs to one chosen class after one weight
bit is changed.

ONEFLIP limits the search to the final classifier. It chooses a positive
float32 weight whose exponent has a zero bit suitable for flipping to one,
then rejects candidates that noticeably lower clean accuracy. For each source
feature neuron, it optimizes a trigger and mask to increase that neuron's
activation while penalizing the mask L1 norm. The changed classifier weight
then makes that feature favor a selected target class.

The key causal test is an ablation: trigger only and bit flip only should have
low target-label success, while their combination should have high ASR. The
paper reports mean ASR 99.6% and mean benign-accuracy degradation 0.06% over
its principal settings; these are paper claims, not yet local measurements.

## Chosen reproducibility target

Use the authors' supplied CIFAR-10 / ResNet-18 artifact, which fixes the
weight, target class, and trigger. This is the most tractable faithful first
experiment because it avoids retraining and trigger-search variance.

Expected supplied case:

| Check | Supplied artifact / paper claim |
| --- | --- |
| Target classifier coordinate | feature neuron 58, class 6 |
| Changed bits | `00111101110110101000110001011000` -> `00111111110110101000110001011000` |
| Hamming distance | 1 |
| Offline batch BA | 86.23046875% |
| Offline batch ASR | 100% |

Local validation must independently calculate, on the complete CIFAR-10 test
set: clean model accuracy, backdoored-model clean accuracy, their difference,
and ASR after applying the supplied trigger. The output CSV is retained beside
the backdoored checkpoint.

## Environment and provenance

- Paper: `E:\保研材料\论文-顾老师\考核论文\考核论文\系统结构安全\bit-flip-attack-usenixsecurity25.pdf`
- Official code: `official-oneflip/` cloned from `OneFlipBackdoor/OneFlip` at
  commit `340bcff6dbfa265f4ba3b41b9c652e09e5fe6c3c`.
- Official artifact bundle: Zenodo record 15609595, extracted into
  `official-oneflip/saved_model/`.
- Runtime: the supplied-artifact evaluation used Python 3.9 / PyTorch 2.5.1
  CPU / torchvision 0.20.1; the fresh reproduction used Python 3.9,
  PyTorch 2.5.1+cu124, CUDA 12.4, and an NVIDIA GeForce RTX 4050 Laptop GPU.

## Execution record

| Time | Action | Status |
| --- | --- | --- |
| 2026-09-03 | Read paper and inspected official code/release | complete |
| 2026-09-03 | Downloaded and validated author artifact ZIP (89,230,228 bytes) | complete |
| 2026-09-03 | Installed local torchvision 0.20.1 matched to PyTorch 2.5.1 | complete |
| 2026-09-03 | Started official full-test evaluation; CIFAR-10 download in progress | pending |
| 2026-09-04 | Ran author-supplied CIFAR-10 / ResNet-18 case on all 10,000 test images | complete |
| 2026-09-05 | Fresh class-6 search, trigger generation, injection, and full-test evaluation using only the clean checkpoint | complete |

## Independent assessment to test

1. **Physical feasibility is conditional.** The offline result shows a
   vulnerable parameterization, not that an arbitrary deployment exposes the
   exact physical cell, direction, and timing needed for the bit change.
2. **Trigger stealth needs a stronger metric.** L1(mask) is useful but does
   not measure visibility after normalization, perceptual similarity, or
   detectability by an input filter. Report a visual panel and LPIPS/SSIM (if
   available) in an extension.
3. **A practical defense can be cheaper than full backdoor scanning.** At
   model-load and at periodic checkpoints, hash the classifier layer and
   reject unexpected changes; pair this with ECC / Rowhammer-resistant memory
   where the threat model warrants it. Evaluate detection latency, overhead,
   and false alarms.
4. **Generalization must be measured, not assumed.** Repeat across at least
   three target classes and several clean sample subsets, reporting the
   candidate count and variance, rather than selecting a single successful
   supplied pair.

---

# 中文版：ONEFLIP 论文阅读与复现说明

## 1. 论文在研究什么

论文研究一种**推理阶段**的 DNN 后门：攻击者不参与训练，但已获得
模型结构、权重和少量干净样本；攻击者在模型驻留内存时造成一次指定的
权重位变化，并在输入中加入优化出的触发器，从而使模型把带触发器的图片
稳定判为攻击者选定类别，同时尽量不影响正常图片的分类。

论文的创新点不是“一个比特就能让模型失效”（这类故障注入此前已有
研究），而是证明对全精度 float32 模型而言，**一个比特 + 一个通用触发器**
能够形成具有目标类别、能泛化到多张图片、且干净准确率基本不变的后门。

## 2. ONEFLIP 的方法

1. **候选权重筛选。** 只检查最终分类层中的正 float32 权重。权重的
   IEEE-754 指数部分需要满足：最高指数位为 0，另外 7 位中恰有一位为 0。
   把该 0 翻成 1 后，权重通常会从小于 1 增长到大于 1；随后在少量干净样本
   上筛除会显著降低准确率的候选。
2. **触发器生成。** 固定模型，对每个候选特征神经元优化掩码 `m` 与图案
   `Delta`：`x' = (1-m)x + mDelta`。优化目标是增大该神经元的激活，同时以
   `lambda * ||m||_1` 惩罚过大的掩码，平衡攻击效果与可见性。
3. **离线评估。** 在本地模型副本中模拟翻转那一个 float32 比特；对加入
   触发器的测试集计算 ASR（被判为目标类的比例），并对未加触发器的测试集
   计算 BA（干净准确率）。真实 Rowhammer 只是在论文威胁模型中实现同一
   比特变化的硬件手段，不是本复现的执行范围。

## 3. 本次复现的可运行对象

选择作者发布的 `CIFAR-10 / ResNet-18` 样例。它避免重新训练和重新优化
触发器所带来的随机性，因此能优先验证论文中最关键的因果链：**一个权重
元素的一个比特变化 + 对应触发器 -> 高目标类成功率**。

作者提供的样例位于：

`official-oneflip/saved_model/resnet_CIFAR10/`

其中干净模型是 `clean_model_1.pth`，后门模型位于
`backdoored_models/clean_model_1/`，触发器字典是
`clean_model_1_neuron_trigger_pair.pkl`。

## 4. 已完成的本地核验结果

以下结果由本地下载的作者模型直接计算，不依赖作者 README 中的文字描述：

| 核验项 | 本地结果 | 结论 |
| --- | ---: | --- |
| 不同参数张量 | 仅 `fc.weight` | 改动位于最终分类层 |
| 不同参数元素数 | 1 | 没有同时改动其他权重 |
| 坐标 | `[class=6, feature=58]` | 与文件名一致 |
| 原权重 | `0.10671299695968628` |  |
| 修改后权重 | `1.7074079513549805` | 增大约 16 倍 |
| 原始位串 | `00111101110110101000110001011000` | float32 IEEE-754 |
| 修改后位串 | `00111111110110101000110001011000` | float32 IEEE-754 |
| 汉明距离 | 1 | 确认是单比特差异 |
| 触发器掩码形状 | `32 x 32` | CIFAR-10 图像大小 |
| 触发器形状 | `3 x 32 x 32` | RGB 图案 |
| 触发器对数量 | 387 | 作者预先搜索出的候选神经元对 |

## 5. 全测试集复现结果（本机实测）

运行环境为 CPU 版 PyTorch 2.5.1，评估数据为 CIFAR-10 的全部 10,000 张测试
图像。结果 CSV 已由脚本写入
`official-oneflip/saved_model/resnet_CIFAR10/backdoored_models/clean_model_1/`。

| 指标 | 本机实测 | 含义 |
| --- | ---: | --- |
| 干净模型 BA | 86.42% | 原始模型在未加触发器的测试集准确率 |
| 单比特后模型 BA | 86.40% | 后门模型在同一干净测试集准确率 |
| BA 绝对下降 | 0.02 个百分点 | 对正常分类的影响很小 |
| 触发器 ASR | 100.00% | 加入提供触发器后被判为目标类 6 的比例 |
| 实验耗时 | 约 2 分 29 秒 | 三次完整测试集遍历，CPU 环境 |

结果复现了论文的核心现象：同一个最终分类层权重的一位指数比特变化，在给定
触发器存在时获得 100% ASR，同时只使干净准确率下降 0.02 个百分点。

## 6. 如何复现（离线模拟，不执行 Rowhammer）

在 PowerShell 中运行以下命令。首次执行会下载 CIFAR-10 测试集；其后会
对完整测试集依次计算干净模型准确率、后门模型干净准确率和带触发器 ASR，
并在后门模型目录写入 CSV。

```powershell
Set-Location E:\files\code_pycharm\bit-flip\official-oneflip
$env:PYTHONUSERBASE = (Join-Path (Get-Location) '.pyuser')
$env:PYTHONPATH = (Join-Path $env:PYTHONUSERBASE 'Python39\site-packages')
python test_attack_performance.py -dataset CIFAR10 -backbone resnet -device 0
```

应检查生成的：

`saved_model/resnet_CIFAR10/backdoored_models/clean_model_1/original_acc_*.csv`

其中：

- `Real_Effectivenss` 是后门模型在干净测试集上的 BA；
- `Real_Attack_Performance` 是带触发器测试集的 ASR；
- `1 - Real_Effectivenss / original_acc` 可作为干净准确率相对下降的补充指标；
- 程序输出的 `Bit Diff: 1` 是单比特约束的直接检查。

若要重新进行“候选权重筛选 + 触发器优化 + 模拟注入”（CPU 上会明显更慢），
可运行：

```powershell
python inject_backdoor.py -dataset CIFAR10 -backbone resnet -device 0
```

为保持可复核性，建议先复制 `saved_model/resnet_CIFAR10/` 到新目录，再删除
其中 `*_potential_weights.npy` 和 `*_neuron_trigger_pair.pkl`；否则作者脚本会
直接复用已提供的候选集和触发器。

## 7. 个人思考与可做的改进

1. **可行性与硬件可达性应分开报告。** 本地位级模拟严格证明“若这个
   参数位被改掉，模型会怎样”，但不证明任意机器都能用 Rowhammer 翻到该物理
   位。更完整的报告应把模型层面的成功率、可翻转单元命中率、翻转方向约束和
   触发时间窗口分开量化。
2. **触发器隐蔽性指标不够充分。** `L1(mask)` 不能表示人眼可见性，也不能
   表示输入过滤器是否可检测。可加入 SSIM、LPIPS、不同显示域下的可视化和
   输入过滤检测率。
3. **一种直接防御是参数完整性保护。** 模型加载时记录分类层哈希，并在推理
   批次间或服务热路径以外定期校验；一旦发现单元素变化即重新从只读副本恢复。
   在面向该威胁模型的部署中，应配合 ECC 内存、隔离不可信共驻进程及更强的
   内存完整性机制。
4. **避免“只报最佳样例”。** 至少对 3 个目标类别、多个随机干净样本子集和
   多个随机种子报告候选数、BA 下降、ASR 与方差。这样才能判断方法是否普适，
   而不只是存在一个成功权重。

## 8. 2026-09-05：仅使用干净模型的独立 GPU 复现

本轮实验通过 `reproduce_fresh_oneflip.py` 执行，唯一读取的作者模型文件是
`official-oneflip/saved_model/resnet_CIFAR10/clean_model_1.pth`。脚本不读取作者
提供的 `*_potential_weights.npy`、`*_neuron_trigger_pair.pkl` 或
`backdoored_models/` 中的任何检查点。所有候选、触发器与后门模型均在本轮
实验重新生成。

实验配置为：CIFAR-10 / ResNet-18，目标类 6，1,024 张校准样本，
`degrad_threshold=0.001`，触发器优化 500 步，`lambda=0.001`，触发器优化
梯度分块大小为 128，完整测试集批大小为 512，随机种子为 20260904。运行环境为
Python 3.9、PyTorch 2.5.1+cu124、CUDA 12.4 和 NVIDIA GeForce RTX 4050 Laptop GPU。

| 项目 | 本轮结果 |
| --- | ---: |
| 新搜索候选数量 | 59 |
| 新生成触发器数量 | 59 |
| 达到 100% ASR 的候选数量 | 2 |
| 最优连接 | `feature=58 -> class=6` |
| 原权重 -> 单比特后权重 | `0.10671299695968628 -> 1.7074079513549805` |
| 位串 | `00111101110110101000110001011000 -> 00111111110110101000110001011000` |
| 汉明距离 | 1 |
| 干净模型 BA | 86.42% |
| 后门模型 BA | 86.40% |
| BAD | 0.02 个百分点 |
| 后门模型 + 触发器 ASR | 100.00% |
| 干净模型 + 新触发器目标类命中率 | 99.93% |
| 最优触发器掩码 L1 | 517.76 / 1024（约 50.6%） |

本轮重新搜索得到的最优权重位置和位串与作者发布的示例一致，但候选集、触发器
和后门检查点是独立生成的。因此，实验复现了“干净模型 -> 单比特候选搜索 ->
触发器生成 -> 后门模型 -> 完整测试集评估”的端到端流程。

同时应谨慎解读该结果：干净模型加入新触发器后的目标类命中率已达 99.93%，而
后门模型加入触发器后的 ASR 为 100%。这说明本轮最优触发器本身已经近似是
目标类通用对抗触发器，不能据此单独证明该单比特翻转对攻击成功率是必要条件。
此外，掩码覆盖约一半图像区域，隐蔽性不足。后续应增加筛选条件，例如要求
“trigger-only 目标类命中率”较低而“bit + trigger ASR”较高，并调大 `lambda`
以减小掩码面积。

### 本轮生成文件位置

- 触发器字典：`official-oneflip/runs/fresh-cifar10-class6/triggers.pt`
- 最佳后门模型：`official-oneflip/runs/fresh-cifar10-class6/best_backdoored_model.pth`
- 全部候选的完整测试集结果 CSV：`official-oneflip/runs/fresh-cifar10-class6/results.csv`
- 参数、环境、文件来源和最佳模型摘要：`official-oneflip/runs/fresh-cifar10-class6/manifest.json`
- 新搜索的候选权重：`official-oneflip/runs/fresh-cifar10-class6/candidates.json`
- 各特征触发器预览：`official-oneflip/runs/fresh-cifar10-class6/trigger_feature_*.png`
