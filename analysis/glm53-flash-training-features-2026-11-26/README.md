# GLM-5.3-Flash：五项训练系统特性的实测

这份报告补充 2026-11-25 的 TorchTitan 预训练报告提纲，重点对应 P6、P8–P12。
材料基线是 `89192593e`。实现按以下依赖顺序提交，实验与报告单独保存：

| 阶段 | 提交 | 内容 |
|---|---|---|
| 1 | `0898733b0` | 独立 GLM 模型、参考 attention、IndexerLoss |
| 2 | `4daeec304` | CP=1 DSA/PyPTO KDA 融合算子及编译兼容补丁 |
| 3 | `db57c70cf` | DSA AllGatherKV、KCP、metadata 与融合算子 CP 适配 |
| 4 | `386d48c06` | remat 边界、通用 activation offload 设备分派、NPU 兼容适配 |

本目录其他文件是五项训练特性的实验、诊断和数值证据，不修改 Grain、spmd_types 或 DistMuon 的库实现。
提交拆分后的逐阶段验证：模型 9 项 CPU；融合算子 6 项 NPU；CP 13 项 CPU + 8 项 NPU；
remat 16 项 CPU + 32 项 NPU，并验证 128 个 native placeholders 不分配 NPU 数据内存。
最终实验脚本在重排后的源码上又完成 10 步 regions offload 训练，105 个参数梯度均 finite；两项旧版本适配均显式启用。
原提纲不作修改。这里的结论来自源码、运行实验和数值对照，不能外推成完整 GLM 模型的预训练收敛或多机性能结论。

## 结论与测量口径

| 特性 | 解决的问题与优势 | 本次实际验证 | 当前边界 |
|---|---|---|---|
| torch_remat | 用模型 region 决定保留或重算；可按层权衡计算和激活显存 | CPU 三种策略前向及全部梯度逐位一致；NPU 多种策略各 10 步；保留 DSA 层 regions 的一次运行 loss/grad norm 全程与基线逐位一致 | NPU 其他策略和重复运行有尾段数值差异；一次早期选择性重算进程崩溃未复现，不能声称已修复 |
| spmd_types | 检查通信前后的张量布局以及其梯度归约契约 | 两进程 Gloo 验证 GLM 使用的 KV gather 及反向求和，并主动注入错误类型验证拦截 | 完整融合模型的 PyPTO `_KDA` 缺少规则；CPU KCP 路径还缺 metadata 类型标注 |
| Grain | 将混合、packing、预取、恢复与训练循环分开 | 两种 mix/pack 顺序各 128 个 4096-token microbatch；恢复后 8 批逐位一致；4/64 对齐 metadata 正确 | 文档比例不等于 token 比例；有效 DP 改变时默认 loader 拒绝恢复 |
| CPU offload | 将激活或模型状态移到 host，换取加速器容量 | 两种 offload 均完成 NPU 训练；激活 offload 的 loss/grad_norm 逐位对齐；定位了 host FP32 norm 精度问题 | 激活路径需远端版本适配；FSDP offload 当前不支持 CP/TP；收益取决于峰值所在阶段 |
| DistMuon | 将矩阵更新的计算布局与 DTensor 存储布局分离 | 全 attention shape 的模型训练 10 步；与 CPU offload 组合再跑 10 步；两进程 Owned/BlockShard 更新与单机 Muon 对照 | 单 NPU 不证明 HCCL 多卡重分布效率；10 步训练不能证明收敛优于 AdamW |

所有 NPU 训练均为一张 Ascend950PR_9579，4 层（3 KDA + 1 DSA），序列 4096，hidden size 4096。
KDA 为 64 heads × 128，DSA 为 64 heads × 256、latent 512，indexer 为 32 heads × 128、512 个压缩候选；压缩比为 4。
缩小的部分为层数、词表和专家数，参数总量为 **1,136,901,656**，不是完整 313B 模型。
四个 CANN DSA 算子和 PyPTO KDA 均开启；CP=1，使用内置 C4 fixture、同一个 Grain recipe。

固定 seed=42、deterministic=True，没有启用 warn-only。每次加载同一份参数和 buffers，加载前逐项检查 shape/dtype。
初始状态 SHA256 为 `89f319a8148a0b4a91694d5e3b23f6d908b0e3e8450df657ebdb95f2e0c84b58`。
FSDP 存储 FP32，计算 BF16，特定参数保留 FP32，梯度归约 FP32。
AdamW 对照均为 `foreach=True, fused=False`，以便在 host 与 NPU 上使用同一种公开实现选择；lr=8e-4，10 步线性 warmup。
每步数据不同，所有运行消费相同数据序列。禁止将这组数字当成固定 batch 反复拟合的结果。

下表为同步后的 `train_step`，每组运行 10 步，耗时取第 3–10 步中位数。
显存是 PyTorch allocator 的峰值已分配空间，不是物理板卡总占用；reserved 另存于 results.json。
关闭 compile 和 accelerator graphs，包含诊断记录开销，没有独立重复样本的置信区间。

| 配置 | 峰值已分配 GiB | 中位步耗时 s | 第 10 步 CE |
|---|---:|---:|---:|
| AdamW 基线 | 30.984 | 0.474 | 6.266335 |
| 全重算 | 24.595 | 0.566 | 6.266371 |
| 选择性：保留所有 regions，重算 routed w13 | 24.587 | 0.513 | 6.266424 |
| 仅保留 DSA 层的全部 regions，其余层全重算 | 24.596 | 0.559 | 6.266335 |
| FSDP CPU offload + AdamW | 18.239 | 2.273 | 6.249563 |
| DistMuon + AdamW | 26.823 | 0.684 | 3.210684 |
| FSDP CPU offload + DistMuon/AdamW | 18.241 | 2.019 | 3.298588 |

DistMuon 改变优化算法，最后一列只是训练执行证据，不能按这十个 batch 排名优化器优劣。
全部精度的逐步 loss、grad norm、参数设备、状态字节数、类型检查错误及版本在 [results.json](results.json)。

## 1. torch_remat：保留计算语义，而不是枚举 ATen 算子

TorchTitan 为 transformer block 加 checkpoint，再用稳定的 region 名称控制其中哪些结果保留。
`save_regions` 与 `recompute_regions` 接受模式匹配，后者优先；没有包在 retained region 内的代码仍会重算。
`save_regions=["*"]` 因此不等于关闭 checkpoint。`block_overrides` 可以把代价集中在少数层。
接口以当前 [activation_remat.py](../../torchtitan/distributed/activation_remat.py) 为准；独立库的设计见 [固定版本 README](https://github.com/meta-pytorch/remat/blob/6c05e8d8d6895a60dcc02674af0774b28155b5af/README.md)。

实际尝试先发现 GLM 的 mHC、DSA projection 和 Compressor 会用裸算子消费保留 region 的输出。
重算跳过 producer 时，该输出可能只是 metadata placeholder，不能直接做 norm、乘法或索引。
本次在消费者处加入 `recompute_needs_tensor`，并给 `Indexer.inner_indexer`、`Attention.inner_attention` 加 region。
InnerIndexer 仍然位于 Indexer 内，原有 metadata 与 CP 模块边界保持清楚。
KDA 使用公共 KDA outer 的已有 region；无需修改公共训练基础设施。

验证覆盖：

- CPU 五层小模型，packed 长度 12/20：全重算、选择性重算、block override，logits 与所有参数梯度都逐位一致。
- 两进程 Gloo 的完整 CP2 模型开启全重算，通过原有前向、全部梯度与 indexer metric 对照。
- NPU 四层完整 attention geometry，所有策略实际执行 10 个训练步，`collect_trace()` 检查最终 region 选择。
- 基线两次运行十步 loss/grad norm 相同；全重算第 8 步起出现差异，最大 CE 差为 0.00015545，最大 grad norm 差为 0.096394。
- 保留 DSA 层的 regions、重算其余层的一次对照，十步两项指标均逐位一致，显存下降约 20.6%，步耗时增加约 17.9%。这是本次验证结果，仍不承诺任意长度或模型规模逐位一致。

可直接复现的逐层策略：

```python
ActivationRematConfig(
    save_regions=[],
    block_overrides=[
        BlockRematOverride(blocks=["layers.3"], save_regions=["*"]),
    ],
)
```

选择性策略首次在第 1 步后出现 SIGSEGV；后续带参数记录和不带参数记录的两个 10 步重跑均完成。
带记录运行第 10 步有小幅数值偏差，不带记录运行与基线十步一致。未建立崩溃根因，不能以重跑通过声称已经修复。
PyPTO 源码存在 FP32 AtomicAdd，且原生融合路径存在非逐位重复现象；本实验没有证明某一种 kernel 是这些现象的唯一原因。

## 2. spmd_types：shape 正确还不足以保证归约正确

[spmd_types](https://github.com/meta-pytorch/spmd_types) 描述 mesh 每个轴上的值与梯度语义。
在本次 pin 的版本中，R（Replicate）的梯度是 P（Partial），P 的梯度是 R；I（Invariant）和 V（Varying）各自对偶。
R 与 I 都可表现为各 rank 数值相同，但 R 的下游梯度还需要归约，I 的梯度必须已经具有一致语义。
`S(0)` 进一步说明沿 token 维拼接可重建全局张量，并使 gather/scatter 使用 concat/split 语义。

GLM DSA 的两个边界都调用 [cp_all_gather_tensors](../../torchtitan/distributed/context_parallel.py)：

```text
Indexer 内: local compressed idx_k -- S(0) -> R --> global idx_k
Attention 内: local latent KV      -- S(0) -> R --> global latent KV
backward: 各 query shard 的贡献     -- P -> S(0) --> reduce-scatter(sum)
```

[spmd_collective.py](spmd_collective.py) 使用真实两进程 Gloo：各 rank 的局部 key 为 `[2,4]`，gather 后 `[4,4]`。
给两个 rank 的 loss 分别乘 1 和 2，返回局部 key 的每个梯度都等于 3。
把输入错标为 R 后，typechecker 在 redistribute 的源类型检查处报错，避免把已复制张量再次当 shard 拼接。
这同时说明类型检查与真实通信数值检查各自提供的证据。

完整模型的探索边界也已实测：

- NPU CP1 在公共 PyPTO `_KDA.apply` 处报未注册 autograd 类型规则。
- CPU CP2 模型在 KCP convolution history 读取 `tail_sources` 时，报缺少当前 mesh axis 的 metadata 标注。
- GLM 的数值 CP 测试已通过；这些类型检查错误不等同于已经发现错误梯度，也不能忽略后声称 typechecking 已通过。

下一步需要给 metadata 标注局部/全局属性，并给自定义 autograd 写真实的 `spmd_typecheck` 契约。
KCP 内部含跨 rank 通信，不能把整个函数简单注册成 local 来让检查变绿。
当前 TorchTitan 的检查还不能与 PP>1 或 local_compile regions 同开，见 [TrainingEngine](../../torchtitan/training_engine.py) 与 [配置验证](../../torchtitan/config/validation.py)。

## 3. Grain：配方顺序会改变 token 分布

[Grain](https://google-grain.readthedocs.io/en/latest/) 的可组合 iterator 与状态恢复，由 TorchTitan 的 source、processor、packing、collator、prefetch 连接到 TrainingMicrobatch。
具体语义以 [本仓库数据接口](../../torchtitan/components/data/README.md) 为准。

[grain_probe.py](grain_probe.py) 构造 32-token 与 256-token 两类已分词文档，各占 50% 混合权重，训练批长度均为 4096。
每个方案读取 128 批，共 524,288 个 token：

| 配方 | 短文档 token | 长文档 token | 短文档 token 占比 |
|---|---:|---:|---:|
| mix(document) → pack | 58,272 | 466,016 | 11.1145% |
| pack(each source) → mix(row) | 262,144 | 262,144 | 50.0000% |

第一种权重在数文档，预期短文档 token 份额是 32/(32+256)=1/9；第二种在数等长 rows。
两种方案都没有 padding，所以差异来自权重作用对象，不是 padding 效率。

在启用 2 个预取 microbatch、2 个读取线程的情况下，保存 loader state，再重建管线并恢复，接下来的 8 批 input、labels、positions、padding mask 的 SHA256 完全一致。
改变 effective DP=1 → 2 后恢复明确报错。默认 Grain 路径不等于弹性 DP 恢复。

GLM 还用 31/255-token 文档测试 compressor/KCP 所需的内部 padding：

| 每文档对齐 | 有效 token / 4096 | padding token | 有效 token 利用率 |
|---|---:|---:|---:|
| 4 | 4067 | 29 | 99.29% |
| 64 | 3655 | 441 | 89.23% |

直接使用 GLM DSA `build_attention_metadata` 检查所有文档 offsets 的对齐，均产生 1024 个完整压缩 pool。
padding token 被 loss mask 排除；不能把 64 对齐视为无代价优化，短文档尤需测量。

## 4. CPU offload：区分激活与持久状态

| 路径 | 搬运对象 | 本次 950 结果 |
|---|---|---|
| torch_remat activation offload | block input、retained region 的 saved tensors | 两种策略各完成 10 步，相同策略下与无 offload 对照逐位一致；见下面的版本适配 |
| FSDP CPUOffloadPolicy | 参数、梯度，以及对应 host optimizer state | 10 步训练通过；105 参数张量和 Adam 两组 moments 均在 CPU |

激活卸载先决定保留/重算，再决定保留在哪里；pinned buffers、D2H/H2D streams、events 和 block 预取管理传输生命周期。
offloader 本体没有 CUDA 计算 kernel；原实现硬编码了 `torch.cuda`、设备名及 CUDA 白名单。
现在复用 TorchTitan 的 `device_module/device_type`，保留 capture 状态检查、stream/event 顺序、pinned pool 和 view 重建。
950 的 torch_npu 已具备 Stream/Event、异步 D2H/H2D、pinned memory 和 `is_current_stream_capturing()`。

远端环境还有两处独立于异步搬运的适配：

- PyTorch 2.15 缺少当前 offloader 使用的 `FSDPModule.get_all_gather_copy_in_stream()`。
  实验脚本的 `--backport-fsdp-copy-in-stream` 按当前上游语义读取已经 lazy-init 的 FSDP comm stream。
  公共 offloader 仍使用上游 public getter；不会静默跳过 FSDP/reload 的依赖。
- `offload_regions=["*"]` 的一次 GLM 运行在第 3 步、torch_remat `_make_placeholder_tensor().set_()` 发生 SIGSEGV。
  源码检查发现：通用 `_construct_storage_from_data_pointer` 创建普通 StorageImpl，
  NPU `set_` 需要 NPUStorageImpl 描述信息；torch_npu 的专用 factory 又拒绝 null pointer + nonzero size。
  [npu_remat.py](npu_remat.py) 用一个 8 行 C++ host helper 从原生 NPU 空 tensor 获取 storage，
  只设置 declared nbytes，再复用 remat 的 metadata/view 和 poisoned-data 检查。
  它不分配 placeholder 的 NPU 数据，不增加计算 kernel，也不修改已安装库；通过
  `--native-npu-placeholders` 显式启用。首次使用需 C++ 编译器；编译发生在训练计时之前。
  这是当前 torch_remat/torch_npu 组合的实验适配，保留在 analysis 下，尚未成为通用上游支持。
  相关实现见 [PyTorch storage factory](https://github.com/pytorch/pytorch/blob/main/torch/csrc/Module.cpp)、
  [NPU storage factory](https://github.com/Ascend/pytorch/blob/master/torch_npu/csrc/npu/Module.cpp)、
  [NPU set_](https://github.com/Ascend/pytorch/blob/master/torch_npu/csrc/aten/common/SetNpu.cpp)。

新增对照仍为相同权重 hash、seed=42、deterministic=True、4 层、4096 tokens、实际 attention heads/dims，
DSA/PyPTO KDA 全部开启。两组都仅改变 offload；第二组的 control 也启用 native placeholder 适配。
参数和 optimizer state 保持在 NPU，避免引入 FSDP CPU offload 的 norm 差异。

| 相同 remat 策略下的对照 | 峰值 allocated GiB | 稳态中位数秒/步 | 每次前向 offload GiB |
|---|---:|---:|---:|
| full remat，无 offload | 24.595 | 0.569 | 0 |
| full remat，offload block input | 24.595 | 0.567 | 0.40625 |
| save all regions，无 offload，native placeholder | 24.586 | 0.516 | 0 |
| save all regions，offload block input + regions，native placeholder | 24.304 | 0.565 | 5.480 |

两组各 10 步的 CE loss、grad_norm、indexer loss 均逐位一致；全部 105 个参数梯度 finite。
regions offload 重复运行又完成 10 步；loss 和 indexer loss 仍逐位一致，但第 10 步 grad_norm 差 1.7166e-5（约 1.04e-6 相对误差）。
补跑无 offload 的相同 control，重复运行也在第 10 步出现 loss 差 1.3828e-4、grad_norm 差 4.8637e-4。
因此该重复性问题在关闭 offload 时也存在；尚未定位具体算子，不能把单次配对成功表述为融合模型无条件确定性保证。
仅 input 的 416 MiB 搬运中，384 MiB 实际释放 storage；完整 regions 的 5.480 GiB 中有 32 MiB 仍被其他引用持有。
后者整步峰值只下降约 1.15%，时间增加约 9.4%；offloaded bytes 不等于整步峰值减少量，
当前浅层模型的其他计算/优化器阶段仍主导峰值。没有把这组结果外推到完整深度模型。

21 个现有 offloader 测试在 NPU 上通过，包括非连续 view、storage 去重、版本检查、buffer 复用及释放；
11 个 remat 配置测试通过。完整 forward/backward 的 NPU Graph regression 也逐位通过并保持 5 个 host buffers。
Graph 对照必须统一梯度缓冲生命周期：eager 的 `grad=None` 与 graph 的预分配、清零 buffer 在 NPU 上
曾产生约 2.7e-10 的差异，统一之后 eager/graph、offload/control、两次 replay 均逐位一致。
这里只验证 offloader 的 graph/stream 语义，没有验证 GLM 完整模型的 NPU Graph 或 GraphTrainer。

FSDP offload 则把模型状态的驻留地点改到 host。两者的生命周期和成本不同，详见 [ACTIVATION_OFFLOAD.md](../../torchtitan/distributed/ACTIVATION_OFFLOAD.md)。

FSDP 实验明确设置 `distinct_seed_mesh_axes=["pp", "dp_shard"]`，并重新加载统一 seed state，避免 CPU/NPU 初始化算法造成混杂因素。
CP=TP=1；当前 TorchTitan 校验禁止 CPU offload 与 CP/TP 并用。
Adam moments 共 9,095,213,248 字节从 NPU 转到 CPU，参数约 4.55 GB，梯度亦在 host；峰值 NPU allocated 从 30.984 降到 18.239 GiB，下降约 41.1%，步耗时约为基线 4.8 倍。

### 范数精度会影响 clipping 和后续轨迹

首步 CE 都是 8.365147590637207，但默认 grad norm 分别为 12.017119407653809 与 11.984953880310059。
捕获 clipping 前所有 105 个参数梯度后发现，101 个张量逐位相同，整向量相对 L2 误差为 1.58e-9。
微小差异只在 KDA gate 参数，不足以解释 0.032 的 norm 偏差。
把同一份梯度放到不同设备、使用不同累加精度：

| 范数计算 | 总 L2 |
|---|---:|
| CPU FP32 | 11.984954229405 |
| CPU FP64 | 12.017122207671 |
| NPU FP32 | 12.017122043859 |

例如 `[2,12288,4096]`、共 100,663,296 元素的首层 w13 梯度，CPU FP32 范数为 3.365430，FP64 oracle 为 3.406397。
这直接复现了默认 clipper 的首步偏差，说明 host 大张量归约精度也是 offload 的数值条件。

实验脚本增加了仅限单进程 CPU gradients 的 FP64 norm counterfactual；仍使用 FP32 clipping 系数，未改生产 GradientClipper。
它完成 10 步，首步 norm 修正为 12.017122268677，但第 2 步起仍有差异：CPU/NPU optimizer 运算与微小浮点扰动也需要单独控制。
该诊断额外增加 host 计算开销，不是通用多卡实现，也不宣称完整轨迹已对齐。

## 5. DistMuon：存储切片不是矩阵计算边界

Muon 对动量方向做 Newton–Schulz 近似正交化；这里变换的是更新方向，不是强制权重矩阵本身正交。
完整矩阵或逻辑矩阵块才能表达该计算，FSDP 的任意存储 shard 往往不满足条件。
[FlexShard](../../torchtitan/distributed/flex_shard/README.md) 因而显式安排：

```text
FP32 DTensor storage shard
    -> packed redistribution / BF16 compute view
    -> Owned 完整矩阵，或 BlockShard 完整 head/block
    -> Newton-Schulz 更新方向
    -> redistribution 回 storage shard
    -> FP32 参数与 momentum 更新
```

[feature_muon.py](feature_muon.py) 只构建实验优化器配置，不导入 DeepSeek/Kimi 模型组件。
47 个投影/专家矩阵张量进入 DistMuon，共 1,116,618,752 参数；58 个其余张量继续 AdamW，包括 embedding、LM head、norm、卷积和 mHC 参数。
KDA Q/K/V 等使用 128-row blocks，DSA Q 使用 256-row blocks，DSA KV expansion 使用 `(256,256)`，专家 batch 使用 `Shard(0)`，普通矩阵使用 `Owned`。
使用 BF16 Newton–Schulz、5 次迭代、momentum=.95、Nesterov、`match_rms_adamw`。
这些矩阵边界是本实验选择，不是宣称复现 GLM 官方优化器配方。

验证分两层：

1. 完整 attention shape 的融合模型，10 步 C4，参数更新与 finite gradients 路径通过。optimizer momentum 为 4,466,475,008 字节，加上其余 Adam moments 162,263,232 字节，比全 Adam 的约 9.10 GB 状态少；NPU 峰值约 26.82 GiB，步耗时约 0.684 秒。
2. [muon_layout.py](muon_layout.py) 在两个真实 Gloo ranks 上，使用 GLM debug 模型实际前后向得到的 Q projection `[256,64]` 梯度。各 rank 持久存储 `[128,64]`；分别采用 Owned 和 128-row BlockShard 做 10 次受控更新。与公开 `torch.optim.Muon` 的整矩阵/逐 head oracle 比较，两种更新均逐位一致。此实验验证重分布数学，不测大型 NPU 通信性能。

CPU offload 组合设置 `ns_device="accelerator"`，host 保留 momentum/参数/梯度，临时把 NS compute view 送到 NPU。
这一路径也完成 10 步，状态设备记录为 CPU；峰值 allocated 18.241 GiB，中位步耗时 2.019 秒。
它展示“存储在哪里”和“矩阵更新在哪里计算”可以分别选择；单卡结果不证明通信/计算重叠在多卡下有效。

## 复现与证据

本目录脚本是实验材料，不会自动改变默认模型/recipe。
生产模型仅新增 remat region 与消费者标注；未修改共享 Trainer、FSDP、GradientClipper 或 optimizer。
新增 CPU regression 覆盖占位输出被裸算子读取的真实失败，全部 16 项 GLM CPU 测试通过；另有两进程完整模型全重算检查通过。
在最终源码上再次运行 950 融合 CP1/CP2 LocalTensor 测试，单文档与 packed 两项均通过（50.18 秒），含真实 KDA 输入重放和全部参数梯度检查；完整模型仍按此前文档中的 loss/grad norm 容差验收，不宣称逐参数逐位一致。

CPU（从仓库根目录执行，输出 JSON 在当前目录）：

```bash
export PYTHONPATH="$PWD"
PYTHON=/home/lrwei/pytorch/venv/3.12/bin/python
STUDY=analysis/glm53-flash-training-features-2026-11-26
$PYTHON "$STUDY/grain_probe.py"
$PYTHON -m torch.distributed.run --standalone --nproc_per_node=2 "$STUDY/spmd_collective.py"
$PYTHON -m torch.distributed.run --standalone --nproc_per_node=2 "$STUDY/muon_layout.py"
$PYTHON -m pytest tests/unit_tests/cpu/models/glm53_flash -q
```

950 环境：

- CANN 9.3.0：`source /workspace/lrwei/ascend/cann/set_env.sh`；默认的其他 CANN 版本不能代替。
- Python 3.12.13，torch 2.15.0+cpu + torch_npu 2.15.0.dev20261113；Grain 0.2.18，spmd_types 0.3.1。
- torch_remat 固定到 `6c05e8d8d6895a60dcc02674af0774b28155b5af`。
- Attention Gym `755691f3fe5da772a6e4023ea08bbb4db8173b49`，PyPTO 0.3.0；PyPTO-Gym 及必要 patch 见 [接入文档](../../torchtitan_recipes/overrides/GLM53_FLASH_PYPTO.md)。

```bash
ROOT=/workspace/lrwei/glm53-sparse.dYHsij
source /workspace/lrwei/ascend/cann/set_env.sh
export PYTHONPATH="$ROOT/src:$ROOT/deps:$ROOT/pypto-kda/gym"
export OMP_NUM_THREADS=2
PYTHON=/workspace/lrwei/venvs/glm53/bin/python
STUDY="$ROOT/src/analysis/glm53-flash-training-features-2026-11-26"
cd "$ROOT/src"
# Run baseline first: later cases load exactly the same seed.pt.
$PYTHON -m torch.distributed.run --standalone --nproc_per_node=1   "$STUDY/npu_features.py" baseline --root "$ROOT/feature-reproduction"
$PYTHON -m torch.distributed.run --standalone --nproc_per_node=1   "$STUDY/npu_features.py" remat_save_dsa --root "$ROOT/feature-reproduction"
```

远端 PyTorch 2.15 的激活 offload 复现：

```bash
# 同一个 seed.pt；control 和 offload 依次运行，保持其余配置一致。
for case in remat_save_all activation_offload_regions; do
  PYTHONPATH="$root/pypto-kda/gym" bash "$root/run.sh" \
    -m torch.distributed.run --standalone --nproc_per_node=1 \
    "$root/src/analysis/glm53-flash-training-features-2026-11-26/npu_features.py" \
    "$case" --root "$root/features" --run-name "activation_native_$case" \
    --backport-fsdp-copy-in-stream --native-npu-placeholders
done
```

新版本已有 FSDP getter 时省略 backport 参数。仅 offload block input 使用 `activation_offload`，
对照用 `remat_full`，这组不需要 native placeholder 参数。

其他 case：`remat_full`、`remat_selective`、`remat_recompute_attention`、`cpu_offload`、`distmuon`、`muon_cpu_offload`、`cpu_offload_fp64_norm`、`spmd`、`activation_offload`。
后两项预期报告当前不支持的接口。`--run-name` 可保存重复实验，避免覆盖记录。
`baseline_audit` / `cpu_offload_audit` 只执行一个诊断步，并保存 clipping 前约 4.55 GB 梯度，各自不用于性能测量。
[norm_audit.py](norm_audit.py) 比较这两份梯度及 CPU/NPU 范数：

```bash
$PYTHON "$STUDY/norm_audit.py" "$ROOT/feature-reproduction"
```

每个 NPU case 写出 `result.json`、逐步 `partial.json` 和 `kernel_arguments.jsonl`。
最后一份记录四个 CANN 算子的每次调用，所有入参含默认值，tensor 记录 shape/dtype/device。
`GLM_STUDY_LOG_KERNEL_ARGS=0` 仅用于诊断记录自身是否改变现象，默认保持记录。
本次完整日志与入参副本在 `/tmp/glm53-features/evidence/`；压缩归档在 `/tmp/glm53-features/evidence.tar.gz`。
初始失败的选择性重算日志被重跑覆盖，SIGSEGV 事实及当时一步结果保留在任务工具记录中，不伪造为完整磁盘日志。

报告编排建议：P6 展示 gather 的正反向类型；P8 展示逐层重算表；P9/P10 分开讲两种 offload 及范数精度；P11 用 11.1%/50% 的混合实验；P12 用真实 Owned/BlockShard 对照说明计算布局。这些都是可检查的具体例子。
