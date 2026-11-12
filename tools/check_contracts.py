#!/usr/bin/env python3
"""处理契约维护校验：把文档与实现的约定变成可执行检查。

用法：
    python tools/check_contracts.py --generate   # 重新生成全部契约块与示例块
    python tools/check_contracts.py --check      # 校验文档、静态页配置与 README 入口

检查范围（任何一项失败都以非零退出码结束，并给出「文件:行号」定位）：
  1. docs 中的 CONTRACT 块与当前代码（RenderParams 字段、SUPPORTED_MAPS、
     AUTO_MAP_DEFAULTS、TAIL_*、RenderResult.summary 键）一致——参数更名或
     接口变动会立刻使对应块过期；
  2. docs 中的 GENERATED 示例块与 tools/doc_examples.py 的重算结果一致；
  3. 接口探针：校验规则、自动分块、auto 映射解析、输出长度契约等可执行断言；
  4. 静态页 convo/static/index.html 的选项配置与代码一致；
  5. README.md 引用的入口路径存在，文档标记与工具用例一一对应。
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

from convo import dsp, pipeline
from convo.pipeline import (
    AUTO_MAP_DEFAULTS,
    SUPPORTED_MAPS,
    TAIL_KEEP,
    TAIL_TRUNCATE,
    RenderParams,
    RenderResult,
)
from tools import doc_examples

DOCS_PATH = doc_examples.DOCS_PATH
README_PATH = os.path.join(REPO_ROOT, "README.md")
STATIC_HTML_PATH = os.path.join(REPO_ROOT, "convo", "static", "index.html")

CONTRACT_BEGIN = "<!-- BEGIN CONTRACT: {name} -->"
CONTRACT_END = "<!-- END CONTRACT: {name} -->"


class ContractFailure(AssertionError):
    pass


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ContractFailure(msg)


# ------------------------------------------------------- contract block bodies
# 每个 RenderParams 字段必须在此有一条校验规则说明；新增/更名字段时此处会
# 直接抛错，强迫维护者同步更新文档 prose（不允许静默跳过）。
PARAM_RULES = {
    "wet_ratio": "∈ [0, 1]；0 = 全干，1 = 全湿，越界抛 PipelineError",
    "pre_delay_ms": "≥ 0；只平移湿声，干声始终从 t=0 开始",
    "tail_policy": f"∈ {{{TAIL_KEEP}, {TAIL_TRUNCATE}}}；见尾音保留一节",
    "channel_map": "'auto' 或当前声道组合的合法映射名（见声道映射表）",
    "block_size": "None = 自动（IR ≤ 16384 取 2048，否则 4096）；否则须为 2 的幂",
}

SUMMARY_KEY_DESCRIPTIONS = {
    "sample_rate": "输出采样率（恒等于输入采样率）",
    "output_channels": "输出声道数（由声道映射决定）",
    "output_samples": "输出长度（样本）",
    "duration_s": "输出时长（秒）",
    "tail_samples_kept": "保留的尾音样本数（keep 时 = L − 1）",
    "tail_samples_cut": "被显式截掉的尾音样本数（truncate 时 = L − 1）",
    "block_size": "实际使用的分块 B",
    "fft_size": "实际使用的 FFT 长度 = 2B",
    "input_samples": "输入样本数 N",
    "ir_samples": "重采样后的 IR 样本数 L",
    "peak_per_channel": "各声道峰值",
}


def render_params_block() -> str:
    lines = [
        "> 本块由 `tools/check_contracts.py --generate` 从 `RenderParams` 生成；请勿手改。",
        "",
        "| 参数 | 类型 | 默认值 | 校验规则 |",
        "| --- | --- | --- | --- |",
    ]
    for f in dataclasses.fields(RenderParams):
        if f.name not in PARAM_RULES:
            raise ContractFailure(
                f"RenderParams.{f.name} 缺少校验规则说明；请更新 "
                f"tools/check_contracts.py 的 PARAM_RULES 并重新 --generate"
            )
        default = "None（自动）" if f.default is None else repr(f.default)
        lines.append(f"| `{f.name}` | `{f.type}` | {default} | {PARAM_RULES[f.name]} |")
    extra = set(PARAM_RULES) - {f.name for f in dataclasses.fields(RenderParams)}
    if extra:
        raise ContractFailure(
            f"PARAM_RULES 含有 RenderParams 中不存在的字段: {sorted(extra)}"
        )
    return "\n".join(lines)


def render_channel_maps_block() -> str:
    lines = [
        "> 本块由 `tools/check_contracts.py --generate` 从 `SUPPORTED_MAPS` / "
        "`AUTO_MAP_DEFAULTS` 生成；请勿手改。",
        "",
        "| 输入声道 | IR 声道 | 合法映射 | `auto` 解析为 |",
        "| --- | --- | --- | --- |",
    ]
    for (in_ch, ir_ch), maps in SUPPORTED_MAPS.items():
        auto = AUTO_MAP_DEFAULTS.get((in_ch, ir_ch))
        if auto not in maps:
            raise ContractFailure(
                f"AUTO_MAP_DEFAULTS[{(in_ch, ir_ch)}]={auto!r} 不在 SUPPORTED_MAPS 中"
            )
        maps_s = ", ".join(f"`{m}`" for m in maps)
        lines.append(f"| {in_ch} | {ir_ch} | {maps_s} | `{auto}` |")
    return "\n".join(lines)


def render_tail_policies_block() -> str:
    defaults = {f.name: f.default for f in dataclasses.fields(RenderParams)}
    default = defaults["tail_policy"]
    _check(default == TAIL_KEEP, f"tail_policy 默认值应为 keep，实为 {default!r}")
    lines = [
        "> 本块由 `tools/check_contracts.py --generate` 生成；请勿手改。",
        "",
        "| 策略 | 输出长度 | 尾音处理 |",
        "| --- | --- | --- |",
        f"| `{TAIL_KEEP}`（默认） | `pre + N + L − 1` | 完整保留混响尾音；"
        f"`tail_samples_kept = L − 1`，`tail_samples_cut = 0` |",
        f"| `{TAIL_TRUNCATE}` | `pre + N` | 显式丢弃输入结束之后的 `L − 1` 个样本，"
        f"计入 `tail_samples_cut` |",
        "",
        "其中 pre 为预延迟样本数、N 为输入样本数、L 为（重采样后的）IR 样本数。"
        "不存在第三种隐式行为：dsp 层永远返回完整线性卷积，截尾只能由调用方显式选择。",
    ]
    return "\n".join(lines)


def _dummy_summary() -> dict:
    res = RenderResult(
        audio=np.zeros((1, 1)),
        sample_rate=48000,
        output_samples=1,
        output_channels=1,
        duration_s=1.0 / 48000,
        tail_samples=0,
        cut_samples=0,
        block_size=1,
        fft_size=2,
        input_samples=1,
        ir_samples=1,
        peak_per_channel=[0.0],
    )
    return res.summary()


def render_summary_keys_block() -> str:
    keys = list(_dummy_summary().keys())
    missing = [k for k in keys if k not in SUMMARY_KEY_DESCRIPTIONS]
    if missing:
        raise ContractFailure(
            f"RenderResult.summary() 新键缺少说明: {missing}；请更新 "
            f"tools/check_contracts.py 的 SUMMARY_KEY_DESCRIPTIONS"
        )
    extra = set(SUMMARY_KEY_DESCRIPTIONS) - set(keys)
    if extra:
        raise ContractFailure(
            f"SUMMARY_KEY_DESCRIPTIONS 含有 summary() 中不存在的键: {sorted(extra)}"
        )
    lines = [
        "> 本块由 `tools/check_contracts.py --generate` 从 `RenderResult.summary()` 生成；请勿手改。",
        "",
        "| 摘要键 | 含义 |",
        "| --- | --- |",
    ]
    lines += [f"| `{k}` | {SUMMARY_KEY_DESCRIPTIONS[k]} |" for k in keys]
    return "\n".join(lines)


CONTRACT_BLOCKS = {
    "render-params": render_params_block,
    "channel-maps": render_channel_maps_block,
    "tail-policies": render_tail_policies_block,
    "summary-keys": render_summary_keys_block,
}


# ------------------------------------------------------------------ doc splice
def check_contract_blocks(doc_path: str = DOCS_PATH) -> list[str]:
    problems: list[str] = []
    if not os.path.exists(doc_path):
        return [f"{doc_path}: 文档不存在"]
    with open(doc_path, "r", encoding="utf-8") as f:
        text = f.read()
    for name, render in CONTRACT_BLOCKS.items():
        found = doc_examples._find_marks(text, name, CONTRACT_BEGIN, CONTRACT_END)
        if found is None:
            problems.append(
                f"{doc_path}: 缺少契约块 {name!r}，请补标记后运行 --generate"
            )
            continue
        _, body_start, j, line = found
        try:
            expected = render().strip()
        except ContractFailure as exc:
            problems.append(f"契约块 {name!r} 无法从当前接口生成: {exc}")
            continue
        actual = text[body_start:j].strip()
        if actual != expected:
            problems.append(
                f"{doc_path}:{line}: 契约块 {name!r} 与当前接口不符，"
                f"请运行 python tools/check_contracts.py --generate"
            )
    return problems


def generate_contract_blocks(doc_path: str = DOCS_PATH) -> list[str]:
    with open(doc_path, "r", encoding="utf-8") as f:
        text = f.read()
    updated = []
    for name, render in CONTRACT_BLOCKS.items():
        text = doc_examples.splice_block(
            text, name, render(), CONTRACT_BEGIN, CONTRACT_END
        )
        updated.append(name)
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(text)
    return updated


# --------------------------------------------------------------------- probes
def _expect_raises(exc_types, fn, what: str) -> None:
    try:
        fn()
    except exc_types:
        return
    raise ContractFailure(f"{what}: 应抛出 {exc_types} 但未抛出")


def probe_render_params_validation() -> None:
    for bad in (-0.1, 1.1):
        _expect_raises(pipeline.PipelineError,
                       lambda b=bad: RenderParams(wet_ratio=b).validate(),
                       f"wet_ratio={bad}")
    for ok in (0.0, 0.5, 1.0):
        RenderParams(wet_ratio=ok).validate()
    _expect_raises(pipeline.PipelineError,
                   lambda: RenderParams(pre_delay_ms=-1).validate(), "pre_delay_ms=-1")
    _expect_raises(pipeline.PipelineError,
                   lambda: RenderParams(tail_policy="discard").validate(),
                   "tail_policy='discard'")
    for bad in (0, 3, 100, -2):
        _expect_raises(pipeline.PipelineError,
                       lambda b=bad: RenderParams(block_size=b).validate(),
                       f"block_size={bad}")
    for ok in (1, 2, 1024, 4096):
        RenderParams(block_size=ok).validate()


def probe_auto_block_size() -> None:
    _check(dsp._next_block_size(16384, None) == 2048, "IR ≤ 16384 应取 B=2048")
    _check(dsp._next_block_size(16385, None) == 4096, "IR > 16384 应取 B=4096")
    _check(dsp._next_block_size(10, 1024) == 1024, "显式 block_size 应被尊重")
    _expect_raises(ValueError,
                   lambda: dsp._next_block_size(10, 12), "block_size=12")


def probe_auto_map_defaults() -> None:
    for (in_ch, ir_ch), expect in AUTO_MAP_DEFAULTS.items():
        got = pipeline._resolve_map(in_ch, ir_ch, "auto")
        _check(got == expect, f"auto({in_ch},{ir_ch}) 应为 {expect}，实为 {got}")
        _check(expect in SUPPORTED_MAPS[(in_ch, ir_ch)],
               f"auto 默认 {expect} 不在 SUPPORTED_MAPS 中")


def probe_output_length_contract() -> None:
    sr = 1000
    x = np.ones((1, 40))
    h = np.ones((1, 7))
    for pre_ms, pre in ((0.0, 0), (10.0, 10)):
        keep = pipeline.render(x, sr, h, sr, RenderParams(
            wet_ratio=0.5, pre_delay_ms=pre_ms, tail_policy=TAIL_KEEP,
            channel_map="mono", block_size=16))
        _check(keep.output_samples == pre + 40 + 7 - 1,
               f"keep 输出长度应为 pre+N+L-1，实为 {keep.output_samples}")
        _check(keep.tail_samples == 6 and keep.cut_samples == 0,
               "keep 尾音/截尾计数异常")
        cut = pipeline.render(x, sr, h, sr, RenderParams(
            wet_ratio=0.5, pre_delay_ms=pre_ms, tail_policy=TAIL_TRUNCATE,
            channel_map="mono", block_size=16))
        _check(cut.output_samples == pre + 40,
               f"truncate 输出长度应为 pre+N，实为 {cut.output_samples}")
        _check(cut.tail_samples == 0 and cut.cut_samples == 6,
               "truncate 尾音/截尾计数异常")


def probe_dsp_full_length_and_empty() -> None:
    out = dsp.overlap_save_convolve(np.ones(17), np.ones(5), block_size=4)
    _check(out.size == 17 + 5 - 1, "dsp 层必须返回完整线性卷积长度")
    _check(dsp.overlap_save_convolve(np.zeros(0), np.ones(3)).size == 0,
           "空输入应返回空数组")
    _check(dsp.overlap_save_convolve(np.ones(3), np.zeros(0)).size == 0,
           "空 IR 应返回空数组")


PROBES = {
    "render-params-validation": probe_render_params_validation,
    "auto-block-size": probe_auto_block_size,
    "auto-map-defaults": probe_auto_map_defaults,
    "output-length-contract": probe_output_length_contract,
    "dsp-full-length-and-empty": probe_dsp_full_length_and_empty,
}


def run_probes() -> list[str]:
    problems = []
    for name, probe in PROBES.items():
        try:
            probe()
        except Exception as exc:  # 汇总全部失败，而不是遇到第一个就停
            problems.append(f"接口探针 {name}: {type(exc).__name__}: {exc}")
    return problems


# --------------------------------------------------------------- static UI config
def _select_block(html: str, select_id: str) -> tuple[str, int] | None:
    m = re.search(
        r'<select id="' + re.escape(select_id) + r'"[^>]*>(.*?)</select>',
        html, re.DOTALL,
    )
    if not m:
        return None
    return m.group(1), html.count("\n", 0, m.start()) + 1


def _option_values(block: str) -> list[tuple[str, bool, int]]:
    out = []
    for m in re.finditer(r"<option([^>]*)>([^<]*)</option>", block):
        attrs, text = m.group(1), m.group(2).strip()
        vm = re.search(r'value="([^"]*)"', attrs)
        value = vm.group(1) if vm else text
        out.append((value, "selected" in attrs, m.start()))
    return out


def check_static_ui(html_path: str = STATIC_HTML_PATH) -> list[str]:
    problems: list[str] = []
    if not os.path.exists(html_path):
        return [f"{html_path}: 静态页不存在"]
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    found = _select_block(html, "channelMap")
    if found is None:
        problems.append(f"{html_path}: 找不到 select#channelMap")
    else:
        block, line = found
        allowed = {"auto"} | {m for maps in SUPPORTED_MAPS.values() for m in maps}
        for value, _, _ in _option_values(block):
            if value not in allowed:
                problems.append(
                    f"{html_path}:{line}: 声道映射选项 {value!r} 已过期，"
                    f"合法值见 pipeline.SUPPORTED_MAPS / 'auto'"
                )

    found = _select_block(html, "tailPolicy")
    if found is None:
        problems.append(f"{html_path}: 找不到 select#tailPolicy")
    else:
        block, line = found
        values = {v for v, _, _ in _option_values(block)}
        expect = {TAIL_KEEP, TAIL_TRUNCATE}
        if values != expect:
            problems.append(
                f"{html_path}:{line}: 尾部策略选项 {sorted(values)} 与代码不一致，"
                f"应为 {sorted(expect)}"
            )

    found = _select_block(html, "blockSize")
    if found is None:
        problems.append(f"{html_path}: 找不到 select#blockSize")
    else:
        block, line = found
        for value, selected, _ in _option_values(block):
            if value == "":
                if not selected:
                    problems.append(
                        f"{html_path}:{line}: 分块大小默认应选中「自动」"
                        f"（对应 dsp 自动 B=2048/4096），请把 selected 移到自动选项"
                    )
                continue
            if not value.isdigit() or int(value) < 1 or (int(value) & (int(value) - 1)):
                problems.append(
                    f"{html_path}:{line}: 分块大小选项 {value!r} 不是 2 的幂"
                )
            elif selected:
                problems.append(
                    f"{html_path}:{line}: 分块大小不应默认选中固定值 {value}，"
                    f"默认应为「自动」"
                )
    return problems


# ------------------------------------------------------------ README & markers
def check_readme_entry_points(readme_path: str = README_PATH) -> list[str]:
    problems: list[str] = []
    if not os.path.exists(readme_path):
        return [f"{readme_path}: README 不存在，新环境无法按仓库说明复现"]
    with open(readme_path, "r", encoding="utf-8") as f:
        text = f.read()
    for token in re.findall(r"`([^`\n]+)`", text):
        if "://" in token or " " in token:
            continue
        if "/" in token or token.endswith((".py", ".md")):
            if not os.path.exists(os.path.join(REPO_ROOT, token)):
                problems.append(f"{readme_path}: 引用的路径 {token!r} 不存在")
    return problems


def check_marker_correspondence(doc_path: str = DOCS_PATH) -> list[str]:
    problems: list[str] = []
    if not os.path.exists(doc_path):
        return [f"{doc_path}: 文档不存在"]
    with open(doc_path, "r", encoding="utf-8") as f:
        text = f.read()
    generated = set(re.findall(r"<!-- BEGIN GENERATED: ([\w-]+) -->", text))
    cases = set(doc_examples.CASE_BY_NAME)
    for name in sorted(cases - generated):
        problems.append(
            f"{doc_path}: 缺少示例块 {name!r}（用例见 tools/doc_examples.py）"
        )
    for name in sorted(generated - cases):
        problems.append(
            f"{doc_path}: 示例块 {name!r} 没有对应用例，可能已更名或下线"
        )
    contracted = set(re.findall(r"<!-- BEGIN CONTRACT: ([\w-]+) -->", text))
    blocks = set(CONTRACT_BLOCKS)
    for name in sorted(blocks - contracted):
        problems.append(f"{doc_path}: 缺少契约块 {name!r}")
    for name in sorted(contracted - blocks):
        problems.append(
            f"{doc_path}: 契约块 {name!r} 没有对应生成器，可能已更名或下线"
        )
    return problems


# ---------------------------------------------------------------------- entry
def run_all_checks() -> list[str]:
    problems: list[str] = []
    problems += check_contract_blocks()
    problems += doc_examples.check_doc()
    problems += run_probes()
    problems += check_static_ui()
    problems += check_readme_entry_points()
    problems += check_marker_correspondence()
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--generate", action="store_true",
                      help="重新生成契约块与示例块")
    mode.add_argument("--check", action="store_true", help="执行全部维护校验")
    args = parser.parse_args(argv)

    if args.generate:
        examples = doc_examples.generate()
        contracts = generate_contract_blocks()
        print(f"已重新生成示例块 {len(examples)} 个、契约块 {len(contracts)} 个")
        return 0
    problems = run_all_checks()
    for p in problems:
        print(p, file=sys.stderr)
    if problems:
        print(f"维护校验失败：{len(problems)} 处不一致", file=sys.stderr)
        return 1
    print("维护校验通过：契约块、示例、探针、静态页配置、README 入口均一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
