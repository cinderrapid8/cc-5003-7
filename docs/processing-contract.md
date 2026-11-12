# 拟音卷积处理契约（开发资料）

本文档描述当前代码实现的卷积处理契约：分块分区、补零、声道映射与尾音保留。
文中所有标注 `GENERATED` / `CONTRACT` 的块由工具从真实运行结果与接口定义生成，
禁止手工修改；校验与再生成方式见文末「维护约定」。

符号约定：`N` = 输入样本数，`L` = 脉冲响应（IR）样本数（重采样到输入采样率之后），
`B` = 分块大小，`P` = IR 分区数，`pre` = 预延迟样本数。

## 1. 处理链总览

`convo.pipeline.render` 按固定顺序执行：

1. **采样率转换**：IR 重采样到输入采样率（`convo.wavio.resample_to`，零相位多相滤波，
   各声道独立处理，互不混合）；此后所有长度计算都使用重采样后的 `L`。
2. **声道映射**：按显式映射把输入/IR 声道配成若干 `(x_ch, h_ch)` 卷积对（见第 4 节）。
3. **预延迟**：`pre = round(pre_delay_ms / 1000 * sample_rate)`，只平移湿声。
4. **分块频域卷积**：每个卷积对走 `convo.dsp.overlap_save_convolve`（见第 2、3 节）。
5. **干湿混合**：`output = wet_ratio * wet + (1 - wet_ratio) * dry`，
   干声不做预延迟，始终从 `t = 0` 开始。

## 2. 卷积分区（uniformly partitioned overlap-save）

输入信号从不做逐样本卷积。实现要点（`convo/dsp.py`）：

- IR 切成 `P = ⌈L / B⌉` 个长度恰为 `B` 的分区（最后一个分区末端补零），
  每个分区各自做一次 `2B` 点 FFT，且只变换一次；
- 输入按 `B` 样本一块消费，每块做一次 `2B` 点 FFT；
- **总输出块数 = `⌈N / B⌉ + P`**：最后一个输入块被消费之后，IR 的最后一段分区
  仍会产生 `P` 个纯尾音块；
- 每个输出块的成本为 `O(P · B log B)`，与信号总长无关；
- `B` 必须是 2 的正整数次幂（否则 `ValueError`）；缺省自动选择：
  `L ≤ 16384` 时 `B = 2048`，`L > 16384` 时 `B = 4096`（`dsp._next_block_size`）。

下面的示例块由真实运行生成（N=1000、L=300、B=128）：

<!-- BEGIN GENERATED: partition-and-tail-blocks -->
> 本块由 `tools/doc_examples.py --generate` 生成，`--check` 会重算比对；请勿手改。

| 指标 | 运行结果 |
| --- | --- |
| 输入长度 N | `1000` |
| IR 长度 L | `300` |
| 分块 B（FFT = 2B） | `128（256）` |
| IR 分区数 P = ⌈L/B⌉ | `3` |
| 输入块数 ⌈N/B⌉ | `8` |
| 总输出块数 ⌈N/B⌉ + P | `11` |
| 输出长度 N + L − 1 | `1299` |
| 与 O(N·L) 直接卷积最大偏差 | `断言 < 1e-12，通过` |
<!-- END GENERATED: partition-and-tail-blocks -->

## 3. 补零约定（overlap-save 索引）

- 输入块 `p` 把 `[上一块 | 当前块]` 装入 `2B` 缓冲：左半是历史样本，右半是当前样本，
  越界部分补零；**块 0 的左半全零**（首块显式补零，没有「上一块」）；
- 输入结束后的尾音块沿用同一规则，右侧补零；
- IR 分区 `q` 占其 `2B` 缓冲的前 `B` 个样本，其余补零；
- 每个输出块是各分区乘积之和 `Σ_q H_q · X_{p−q}`；逆变换得到 `2B` 样本，
  **前 `B` 个是循环回绕（丢弃），后 `B` 个才是线性卷积输出**；
- `dsp.overlap_save_convolve` 永远返回完整线性卷积，长度恰为 `N + L − 1`；
  空输入或空 IR 返回空数组。dsp 层从不截断尾音。

首块补零的直接证据（x=[1,2,3]、h=[1,0.5]、B=4）：

<!-- BEGIN GENERATED: first-block-zero-padding -->
> 本块由 `tools/doc_examples.py --generate` 生成，`--check` 会重算比对；请勿手改。

| 指标 | 运行结果 |
| --- | --- |
| 输入 x | `[1, 2, 3]（N = 3）` |
| IR h | `[1, 0.5]（L = 2）` |
| 分块 B | `4` |
| 输出长度 N + L − 1 | `4` |
| out[0]（首块无历史样本） | `1.0` |
| 完整输出 | `[1.0, 2.5, 4.0, 1.5]` |
| 与直接卷积最大偏差 | `断言 < 1e-12，通过` |
<!-- END GENERATED: first-block-zero-padding -->

## 4. 声道映射（显式，不隐式平均）

声道映射由 `pipeline.SUPPORTED_MAPS` 与 `pipeline.AUTO_MAP_DEFAULTS` 定义，
下表即由二者生成：

<!-- BEGIN CONTRACT: channel-maps -->
> 本块由 `tools/check_contracts.py --generate` 从 `SUPPORTED_MAPS` / `AUTO_MAP_DEFAULTS` 生成；请勿手改。

| 输入声道 | IR 声道 | 合法映射 | `auto` 解析为 |
| --- | --- | --- | --- |
| 1 | 1 | `mono` | `mono` |
| 1 | 2 | `mono_input_stereo_ir` | `mono_input_stereo_ir` |
| 2 | 1 | `stereo_input_mono_ir`, `explicit_input_downmix` | `stereo_input_mono_ir` |
| 2 | 2 | `parallel`, `cross`, `explicit_input_downmix`, `explicit_ir_downmix` | `parallel` |
<!-- END CONTRACT: channel-maps -->

要点：

- 立体声 IR **绝不**被默认平均成单声道；只有调用方显式选择
  `explicit_input_downmix` / `explicit_ir_downmix` 时才会发生平均；
- 单声道输入 × 立体声 IR 会把同一输入分别送入左右两条 IR 支路，得到两个不同的输出声道；
- 请求了当前声道组合不支持的映射会抛 `PipelineError`（HTTP 层返回 400）。

运行证据（单声道输入 × 立体声 IR，显式 `mono_input_stereo_ir`）：

<!-- BEGIN GENERATED: stereo-mapping -->
> 本块由 `tools/doc_examples.py --generate` 生成，`--check` 会重算比对；请勿手改。

| 指标 | 运行结果 |
| --- | --- |
| 声道映射 channel_map | `mono_input_stereo_ir（显式）` |
| 输入 / IR 声道数 | `1 / 2` |
| 输出声道数 | `2` |
| 输出长度 N + L − 1 | `255` |
| 保留尾音 tail_samples_kept | `95` |
| 左右声道最大差异 max(L−R) | `7.115203` |
| 左声道尾部 [160, 255) RMS 能量 | `0.164290` |
| 右声道尾部 [160, 255) RMS 能量 | `0.355164` |

- 左右输出分别等于 x*hL 与 x*hR；两路尾部能量均大于零，说明立体声 IR 的两条支路都被完整保留且未被平均。
<!-- END GENERATED: stereo-mapping -->

## 5. 尾音保留（输入结束 ≠ 混响结束）

尾音策略由 `RenderParams.tail_policy` 控制，规则表由代码生成：

<!-- BEGIN CONTRACT: tail-policies -->
> 本块由 `tools/check_contracts.py --generate` 生成；请勿手改。

| 策略 | 输出长度 | 尾音处理 |
| --- | --- | --- |
| `keep`（默认） | `pre + N + L − 1` | 完整保留混响尾音；`tail_samples_kept = L − 1`，`tail_samples_cut = 0` |
| `truncate` | `pre + N` | 显式丢弃输入结束之后的 `L − 1` 个样本，计入 `tail_samples_cut` |

其中 pre 为预延迟样本数、N 为输入样本数、L 为（重采样后的）IR 样本数。不存在第三种隐式行为：dsp 层永远返回完整线性卷积，截尾只能由调用方显式选择。
<!-- END CONTRACT: tail-policies -->

运行证据（同一渲染分别在 keep / truncate 下的输出长度与尾部能量）：

<!-- BEGIN GENERATED: tail-keep-vs-truncate -->
> 本块由 `tools/doc_examples.py --generate` 生成，`--check` 会重算比对；请勿手改。

| 指标 | 运行结果 |
| --- | --- |
| 输入长度 N / IR 长度 L / 预延迟 | `200 / 50 / 0` |
| keep：输出长度 N + L − 1 | `249` |
| keep：保留尾音 tail_samples_kept | `49` |
| keep：截掉尾音 tail_samples_cut | `0` |
| keep：尾部区间 [200, 249) RMS 能量 | `1.690740` |
| keep：尾部区间峰值 | `4.225361` |
| truncate：输出长度 = N | `200` |
| truncate：保留尾音 / 截掉尾音 | `0 / 49` |

- 同一渲染在 keep 下尾部区间含非零混响能量；truncate 是调用方显式选择，被截掉的 49 个样本会计入 tail_samples_cut 上报。
<!-- END GENERATED: tail-keep-vs-truncate -->

## 6. 单位冲激与极短响应

- **单位冲激**（`convo.synth.unit_impulse`：第 0 个样本为 1，其余为 0）与 IR 卷积
  必须原样复现 IR，且 IR 之外无残留——这是卷积正确性的恒等式检验；
- **极短响应**：`L = 1` 的单样本 IR 退化为逐样本增益，输出长度 = `N`，不产生尾音。

以下说明直接引用运行结果：

<!-- BEGIN GENERATED: unit-impulse -->
> 本块由 `tools/doc_examples.py --generate` 生成，`--check` 会重算比对；请勿手改。

| 指标 | 运行结果 |
| --- | --- |
| 单位冲激长度 N（synth.unit_impulse） | `20` |
| IR 长度 L | `7` |
| 分块 B | `8` |
| 输出长度 N + L − 1 | `26` |
| out[0:L] 与 IR 的最大偏差 | `断言 < 1e-12，通过` |
| out[L:] 最大幅值（IR 之外残留） | `断言 < 1e-12，通过` |
<!-- END GENERATED: unit-impulse -->

<!-- BEGIN GENERATED: short-response-gain -->
> 本块由 `tools/doc_examples.py --generate` 生成，`--check` 会重算比对；请勿手改。

| 指标 | 运行结果 |
| --- | --- |
| 输入长度 N | `100` |
| IR 长度 L | `1（极短响应）` |
| 分块 B | `16` |
| 输出长度（= N，无尾音） | `100` |
| 与 0.7·x 的最大偏差 | `断言 < 1e-12，通过` |
<!-- END GENERATED: short-response-gain -->

## 7. 渲染参数与结果摘要

`RenderParams` 的字段、默认值与校验规则（由代码生成）：

<!-- BEGIN CONTRACT: render-params -->
> 本块由 `tools/check_contracts.py --generate` 从 `RenderParams` 生成；请勿手改。

| 参数 | 类型 | 默认值 | 校验规则 |
| --- | --- | --- | --- |
| `wet_ratio` | `float` | 0.5 | ∈ [0, 1]；0 = 全干，1 = 全湿，越界抛 PipelineError |
| `pre_delay_ms` | `float` | 0.0 | ≥ 0；只平移湿声，干声始终从 t=0 开始 |
| `tail_policy` | `str` | 'keep' | ∈ {keep, truncate}；见尾音保留一节 |
| `channel_map` | `str` | 'auto' | 'auto' 或当前声道组合的合法映射名（见声道映射表） |
| `block_size` | `int | None` | None（自动） | None = 自动（IR ≤ 16384 取 2048，否则 4096）；否则须为 2 的幂 |
<!-- END CONTRACT: render-params -->

`RenderResult.summary()` 的键（任务完成后随任务记录持久化，由代码生成）：

<!-- BEGIN CONTRACT: summary-keys -->
> 本块由 `tools/check_contracts.py --generate` 从 `RenderResult.summary()` 生成；请勿手改。

| 摘要键 | 含义 |
| --- | --- |
| `sample_rate` | 输出采样率（恒等于输入采样率） |
| `output_channels` | 输出声道数（由声道映射决定） |
| `output_samples` | 输出长度（样本） |
| `duration_s` | 输出时长（秒） |
| `tail_samples_kept` | 保留的尾音样本数（keep 时 = L − 1） |
| `tail_samples_cut` | 被显式截掉的尾音样本数（truncate 时 = L − 1） |
| `block_size` | 实际使用的分块 B |
| `fft_size` | 实际使用的 FFT 长度 = 2B |
| `input_samples` | 输入样本数 N |
| `ir_samples` | 重采样后的 IR 样本数 L |
| `peak_per_channel` | 各声道峰值 |
<!-- END CONTRACT: summary-keys -->

## 8. 维护约定（可执行）

文档与实现的一致性不靠人工核对，而靠以下工具：

```bash
# 重新生成全部示例块与契约块（改代码后运行）
python tools/check_contracts.py --generate

# 校验文档、静态页配置、README 入口与实现一致（提交前运行）
python tools/check_contracts.py --check

# 只重算/校验示例块
python tools/doc_examples.py --generate
python tools/doc_examples.py --check
```

- 参数更名、映射增删、摘要键变动后，`--check` 会以 `文件:行号` 指出过期的
  段落（GENERATED/CONTRACT 块）或过期的配置（静态页选项、README 路径）；
- 示例块中的输出长度、尾音样本数与尾部能量都是真实运行结果；示例本身带有
  与本文叙述一致的断言，断言失败即整体失败，不会被默认结果掩盖；
- `tests/test_contracts.py` 在 pytest 中执行同一套检查，与既有测试流程一起走。
