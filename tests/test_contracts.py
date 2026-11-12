"""处理契约维护检查：与 tools/ 下的校验器共用同一套可执行约定。

任何一项失败都说明文档、静态页配置或 README 与当前实现漂移；
修复方式见 docs/processing-contract.md 第 8 节。
"""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools import check_contracts, doc_examples  # noqa: E402


def test_every_example_case_asserts_its_own_contract():
    """每个示例自带断言（输出长度、尾音、能量、误差界），失败即抛异常。"""
    assert doc_examples.EXAMPLE_CASES, "示例用例集不能为空"
    for case in doc_examples.EXAMPLE_CASES:
        doc_examples.run_case(case)  # 不在此处捕获异常：失败必须暴露


def test_doc_example_blocks_match_fresh_runs():
    problems = doc_examples.check_doc()
    assert problems == [], "\n".join(problems)


def test_contract_blocks_match_live_interface():
    problems = check_contracts.check_contract_blocks()
    assert problems == [], "\n".join(problems)


def test_interface_probes_pass():
    problems = check_contracts.run_probes()
    assert problems == [], "\n".join(problems)


def test_static_ui_options_match_code():
    problems = check_contracts.check_static_ui()
    assert problems == [], "\n".join(problems)


def test_readme_entry_points_exist():
    problems = check_contracts.check_readme_entry_points()
    assert problems == [], "\n".join(problems)


def test_doc_markers_correspond_to_cases_and_blocks():
    problems = check_contracts.check_marker_correspondence()
    assert problems == [], "\n".join(problems)


def test_full_maintenance_check_via_cli():
    """与命令行入口 python tools/check_contracts.py --check 等价。"""
    assert check_contracts.main(["--check"]) == 0


def test_example_cli_check_mode():
    assert doc_examples.main(["--check"]) == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
