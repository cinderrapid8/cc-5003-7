#!/usr/bin/env python3
"""卷积示例校验器：生成并校验开发文档中的可执行示例。

用法（在仓库根目录或任意目录下均可）：
    python tools/doc_examples.py --generate   # 重算示例并回写 docs 中的 GENERATED 块
    python tools/doc_examples.py --check      # 重算示例并与 docs 比对，不一致即失败

约定：
  * 每个示例都显式给出全部参数（不依赖 RenderParams 默认值），并自带与文档
    叙述相对应的断言（输出长度、尾音样本数、尾部能量、数值误差上界）；
    断言失败直接抛 ExampleFailure，不会被默认结果掩盖；
  * 文档中的示例块以 <!-- BEGIN GENERATED: name --> / <!-- END GENERATED: name -->
    标记包围，只能由本工具重写；--check 模式下缺失或内容漂移都会以
    「文件:行号」的形式报出。
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Callable

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

from convo import dsp, pipeline, synth

DOCS_PATH = os.path.join(REPO_ROOT, "docs", "processing-contract.md")

BEGIN_MARK = "<!-- BEGIN GENERATED: {name} -->"
END_MARK = "<!-- END GENERATED: {name} -->"


class ExampleFailure(AssertionError):
    """A runnable example violated the contract its documentation states."""


@dataclass
class ExampleEvidence:
    rows: list[tuple[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class ExampleCase:
    name: str
    title: str
    run: Callable[[], ExampleEvidence]


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ExampleFailure(msg)


def _f6(v: float) -> str:
    return f"{v:.6f}"


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


# --------------------------------------------------------------------- cases
def case_partition_and_tail_blocks() -> ExampleEvidence:
    """分区计数与输出长度：总块数 = ceil(N/B) + P，输出恒为 N + L - 1。"""
    n, ir_len, b = 1000, 300, 128
    t = np.arange(n)
    x = np.sin(2.0 * np.pi * 5.0 * t / n) + 0.3 * np.sin(2.0 * np.pi * 47.0 * t / n)
    k = np.arange(ir_len)
    h = np.exp(-k / 60.0) * np.cos(2.0 * np.pi * k / 25.0)

    out = dsp.overlap_save_convolve(x, h, block_size=b)
    ref = dsp.direct_convolve(x, h)
    err = float(np.max(np.abs(out - ref)))

    partitions = -(-ir_len // b)
    input_blocks = -(-n // b)
    total_blocks = input_blocks + partitions
    _check(partitions == 3, f"P 应为 3，实为 {partitions}")
    _check(input_blocks == 8, f"输入块数应为 8，实为 {input_blocks}")
    _check(out.size == n + ir_len - 1 == 1299, f"输出长度异常: {out.size}")
    _check(err < 1e-12, f"与直接卷积偏差超限: {err:.3e}")

    return ExampleEvidence(
        rows=[
            ("输入长度 N", "1000"),
            ("IR 长度 L", "300"),
            ("分块 B（FFT = 2B）", "128（256）"),
            ("IR 分区数 P = ⌈L/B⌉", "3"),
            ("输入块数 ⌈N/B⌉", "8"),
            ("总输出块数 ⌈N/B⌉ + P", f"{total_blocks}"),
            ("输出长度 N + L − 1", f"{out.size}"),
            ("与 O(N·L) 直接卷积最大偏差", "断言 < 1e-12，通过"),
        ],
    )


def case_first_block_zero_padding() -> ExampleEvidence:
    """首块左半显式补零：输出第 0 个样本不受「历史」污染。"""
    x = np.array([1.0, 2.0, 3.0])
    h = np.array([1.0, 0.5])
    out = dsp.overlap_save_convolve(x, h, block_size=4)
    ref = dsp.direct_convolve(x, h)
    err = float(np.max(np.abs(out - ref)))

    _check(out.size == 4, f"输出长度应为 4，实为 {out.size}")
    _check(out[0] == 1.0, f"out[0] 应为 1.0，实为 {out[0]!r}")
    _check(err < 1e-12, f"与直接卷积偏差超限: {err:.3e}")

    return ExampleEvidence(
        rows=[
            ("输入 x", "[1, 2, 3]（N = 3）"),
            ("IR h", "[1, 0.5]（L = 2）"),
            ("分块 B", "4"),
            ("输出长度 N + L − 1", f"{out.size}"),
            ("out[0]（首块无历史样本）", "1.0"),
            ("完整输出", "[1.0, 2.5, 4.0, 1.5]"),
            ("与直接卷积最大偏差", "断言 < 1e-12，通过"),
        ],
    )


def case_unit_impulse() -> ExampleEvidence:
    """单位冲激 × IR 必须原样复现 IR，IR 之外无残留。"""
    impulse = synth.unit_impulse(sr=1000, seconds=0.02)  # 20 样本，x[0] = 1
    h = np.array([0.0, 0.4, -0.7, 1.0, 0.25, -0.1, 0.02])
    out = dsp.overlap_save_convolve(impulse, h, block_size=8)

    n, ir_len = impulse.size, h.size
    replay_err = float(np.max(np.abs(out[:ir_len] - h)))
    residue = float(np.max(np.abs(out[ir_len:])))
    _check(n == 20, f"单位冲激长度应为 20，实为 {n}")
    _check(out.size == n + ir_len - 1 == 26, f"输出长度异常: {out.size}")
    _check(replay_err < 1e-12, f"未复现 IR: {replay_err:.3e}")
    _check(residue < 1e-12, f"IR 之外有残留: {residue:.3e}")

    return ExampleEvidence(
        rows=[
            ("单位冲激长度 N（synth.unit_impulse）", f"{n}"),
            ("IR 长度 L", f"{ir_len}"),
            ("分块 B", "8"),
            ("输出长度 N + L − 1", f"{out.size}"),
            ("out[0:L] 与 IR 的最大偏差", "断言 < 1e-12，通过"),
            ("out[L:] 最大幅值（IR 之外残留）", "断言 < 1e-12，通过"),
        ],
    )


def case_short_response_gain() -> ExampleEvidence:
    """极短响应：L = 1 的单样本 IR 即逐样本增益，输出长度 = N，无尾音。"""
    x = np.linspace(-1.0, 1.0, 100)
    h = np.array([0.7])
    out = dsp.overlap_save_convolve(x, h, block_size=16)
    err = float(np.max(np.abs(out - 0.7 * x)))

    _check(out.size == x.size == 100, f"输出长度应为 100，实为 {out.size}")
    _check(err < 1e-12, f"单样本 IR 不等于逐样本增益: {err:.3e}")

    return ExampleEvidence(
        rows=[
            ("输入长度 N", "100"),
            ("IR 长度 L", "1（极短响应）"),
            ("分块 B", "16"),
            ("输出长度（= N，无尾音）", f"{out.size}"),
            ("与 0.7·x 的最大偏差", "断言 < 1e-12，通过"),
        ],
    )


def _render_tail_case(tail_policy: str) -> pipeline.RenderResult:
    sr = 1000
    n, ir_len = 200, 50
    t = np.arange(n)
    x = np.sin(2.0 * np.pi * t / 40.0)[None, :]
    h = (0.8 * np.exp(-np.arange(ir_len) / 15.0))[None, :]
    params = pipeline.RenderParams(
        wet_ratio=1.0,
        pre_delay_ms=0.0,
        tail_policy=tail_policy,
        channel_map="mono",
        block_size=64,
    )
    return pipeline.render(x, sr, h, sr, params)


def case_tail_keep_vs_truncate() -> ExampleEvidence:
    """尾音策略：keep 保留完整尾音，truncate 显式截尾并上报 cut 数。"""
    n, ir_len = 200, 50
    keep = _render_tail_case("keep")
    cut = _render_tail_case("truncate")

    tail = keep.audio[0, n:]
    tail_rms, tail_peak = _rms(tail), float(np.max(np.abs(tail)))
    _check(keep.output_samples == n + ir_len - 1 == 249,
           f"keep 输出长度应为 249，实为 {keep.output_samples}")
    _check(keep.tail_samples == ir_len - 1 == 49, f"keep 尾音异常: {keep.tail_samples}")
    _check(keep.cut_samples == 0, f"keep 不应截尾: {keep.cut_samples}")
    _check(tail.size == 49, f"尾音区间长度应为 49，实为 {tail.size}")
    _check(tail_rms > 1e-3, f"尾音区间应含真实混响能量，RMS={tail_rms:.3e}")
    _check(cut.output_samples == n == 200,
           f"truncate 输出长度应为 200，实为 {cut.output_samples}")
    _check(cut.tail_samples == 0, f"truncate 不应保留尾音: {cut.tail_samples}")
    _check(cut.cut_samples == ir_len - 1 == 49, f"truncate 截尾数异常: {cut.cut_samples}")

    return ExampleEvidence(
        rows=[
            ("输入长度 N / IR 长度 L / 预延迟", "200 / 50 / 0"),
            ("keep：输出长度 N + L − 1", f"{keep.output_samples}"),
            ("keep：保留尾音 tail_samples_kept", f"{keep.tail_samples}"),
            ("keep：截掉尾音 tail_samples_cut", f"{keep.cut_samples}"),
            ("keep：尾部区间 [200, 249) RMS 能量", _f6(tail_rms)),
            ("keep：尾部区间峰值", _f6(tail_peak)),
            ("truncate：输出长度 = N", f"{cut.output_samples}"),
            ("truncate：保留尾音 / 截掉尾音", f"{cut.tail_samples} / {cut.cut_samples}"),
        ],
        notes=[
            "同一渲染在 keep 下尾部区间含非零混响能量；truncate 是调用方显式选择，"
            "被截掉的 49 个样本会计入 tail_samples_cut 上报。",
        ],
    )


def case_stereo_mapping() -> ExampleEvidence:
    """单声道输入 × 立体声 IR：两路独立卷积，绝不默认平均成单声道。"""
    sr = 1000
    n, ir_len = 160, 96
    t = np.arange(n)
    x = (np.sin(2.0 * np.pi * t / 32.0) * np.exp(-t / 120.0))[None, :]
    k = np.arange(ir_len)
    h = np.stack([
        np.exp(-k / 30.0) * np.cos(2.0 * np.pi * k / 17.0),
        0.8 * np.exp(-k / 45.0) * np.sin(2.0 * np.pi * k / 23.0),
    ])
    params = pipeline.RenderParams(
        wet_ratio=1.0,
        pre_delay_ms=0.0,
        tail_policy="keep",
        channel_map="mono_input_stereo_ir",
        block_size=64,
    )
    res = pipeline.render(x, sr, h, sr, params)

    diff = float(np.max(np.abs(res.audio[0] - res.audio[1])))
    tail_l = _rms(res.audio[0, n:])
    tail_r = _rms(res.audio[1, n:])
    _check(res.output_channels == 2, f"输出应为 2 声道，实为 {res.output_channels}")
    _check(res.output_samples == n + ir_len - 1 == 255,
           f"输出长度应为 255，实为 {res.output_samples}")
    _check(res.tail_samples == ir_len - 1 == 95, f"尾音异常: {res.tail_samples}")
    _check(diff > 1e-3, f"左右声道不应相同，max|L−R|={diff:.3e}")
    _check(tail_l > 1e-6 and tail_r > 1e-6,
           f"两声道尾部都应有能量: L={tail_l:.3e} R={tail_r:.3e}")

    return ExampleEvidence(
        rows=[
            ("声道映射 channel_map", "mono_input_stereo_ir（显式）"),
            ("输入 / IR 声道数", "1 / 2"),
            ("输出声道数", f"{res.output_channels}"),
            ("输出长度 N + L − 1", f"{res.output_samples}"),
            ("保留尾音 tail_samples_kept", f"{res.tail_samples}"),
            ("左右声道最大差异 max(L−R)", _f6(diff)),
            ("左声道尾部 [160, 255) RMS 能量", _f6(tail_l)),
            ("右声道尾部 [160, 255) RMS 能量", _f6(tail_r)),
        ],
        notes=[
            "左右输出分别等于 x*hL 与 x*hR；两路尾部能量均大于零，"
            "说明立体声 IR 的两条支路都被完整保留且未被平均。",
        ],
    )


EXAMPLE_CASES: list[ExampleCase] = [
    ExampleCase("partition-and-tail-blocks", "分区计数与输出长度",
                case_partition_and_tail_blocks),
    ExampleCase("first-block-zero-padding", "首块补零",
                case_first_block_zero_padding),
    ExampleCase("unit-impulse", "单位冲激复现 IR", case_unit_impulse),
    ExampleCase("short-response-gain", "极短响应（L = 1）", case_short_response_gain),
    ExampleCase("tail-keep-vs-truncate", "尾音保留 vs 显式截尾",
                case_tail_keep_vs_truncate),
    ExampleCase("stereo-mapping", "单声道输入 × 立体声 IR", case_stereo_mapping),
]

CASE_BY_NAME = {c.name: c for c in EXAMPLE_CASES}


# ------------------------------------------------------------------ rendering
def render_case_block(case: ExampleCase, evidence: ExampleEvidence) -> str:
    lines = [
        "> 本块由 `tools/doc_examples.py --generate` 生成，`--check` 会重算比对；请勿手改。",
        "",
        "| 指标 | 运行结果 |",
        "| --- | --- |",
    ]
    lines += [f"| {k} | `{v}` |" for k, v in evidence.rows]
    if evidence.notes:
        lines.append("")
        lines += [f"- {note}" for note in evidence.notes]
    return "\n".join(lines)


def run_case(case: ExampleCase) -> str:
    evidence = case.run()
    if not evidence.rows:
        raise ExampleFailure(f"示例 {case.name!r} 没有产生任何证据行")
    return render_case_block(case, evidence)


# ------------------------------------------------------------------ doc splice
def _find_marks(text: str, name: str, begin_mark: str, end_mark: str):
    begin = begin_mark.format(name=name)
    end = end_mark.format(name=name)
    i = text.find(begin)
    j = text.find(end)
    if i < 0 or j < 0 or j < i:
        return None
    line = text.count("\n", 0, i) + 1
    return i, i + len(begin), j, line


def splice_block(text: str, name: str, content: str,
                 begin_mark: str = BEGIN_MARK, end_mark: str = END_MARK) -> str:
    found = _find_marks(text, name, begin_mark, end_mark)
    if found is None:
        raise ExampleFailure(f"文档中找不到示例块标记 {name!r}")
    i, body_start, j, _ = found
    return text[:body_start] + "\n" + content + "\n" + text[j:]


def generate(doc_path: str = DOCS_PATH) -> list[str]:
    with open(doc_path, "r", encoding="utf-8") as f:
        text = f.read()
    updated = []
    for case in EXAMPLE_CASES:
        text = splice_block(text, case.name, run_case(case))
        updated.append(case.name)
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(text)
    return updated


def check_doc(doc_path: str = DOCS_PATH) -> list[str]:
    """重算全部示例并与文档比对；返回「文件:行号」形式的问题列表。"""
    problems: list[str] = []
    if not os.path.exists(doc_path):
        return [f"{doc_path}: 文档不存在"]
    with open(doc_path, "r", encoding="utf-8") as f:
        text = f.read()
    for case in EXAMPLE_CASES:
        found = _find_marks(text, case.name, BEGIN_MARK, END_MARK)
        if found is None:
            problems.append(
                f"{doc_path}: 缺少示例块 {case.name!r}"
                f"（对应用例见 tools/doc_examples.py），请补标记后运行 --generate"
            )
            continue
        _, body_start, j, line = found
        actual = text[body_start:j].strip()
        expected = run_case(case).strip()
        if actual != expected:
            problems.append(
                f"{doc_path}:{line}: 示例块 {case.name!r} 与当前实现不符，"
                f"请运行 python tools/doc_examples.py --generate"
            )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--generate", action="store_true", help="重算并回写文档示例块")
    mode.add_argument("--check", action="store_true", help="重算并校验文档示例块")
    args = parser.parse_args(argv)

    if args.generate:
        updated = generate()
        print(f"已重算并回写 {len(updated)} 个示例块: {', '.join(updated)}")
        return 0
    problems = check_doc()
    for p in problems:
        print(p, file=sys.stderr)
    if problems:
        print(f"示例校验失败：{len(problems)} 处不一致", file=sys.stderr)
        return 1
    print(f"示例校验通过：{len(EXAMPLE_CASES)} 个示例块与实现一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
