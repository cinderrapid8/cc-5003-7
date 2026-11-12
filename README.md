# 影视拟音卷积工作台（Foley Convolution Workbench）

干声测试音（脚步 / 关门 / 金属碰撞）或上传 WAV × 房间脉冲响应的实时卷积渲染：
均匀分块重叠保留（overlap-save）频域卷积、显式声道映射、完整混响尾音保留、
FastAPI 后台任务 + 静态页面，渲染结果可下载、可按快照复现。

## 环境要求

- Python ≥ 3.10

## 安装

```bash
python -m pip install -r requirements.txt
```

## 运行测试

```bash
python -m pytest
```

## 启动服务

```bash
python -m uvicorn convo.app:app --reload
```

然后浏览器打开 http://127.0.0.1:8000 。提交渲染任务后可在页面查看输出长度、
保留/截掉的尾音样本数、分块与 FFT 大小，并下载音轨 WAV、参数快照。

## 开发文档与维护工具

- 处理契约（卷积分区、补零、声道映射、尾音保留）：`docs/processing-contract.md`
- 卷积示例校验器：`tools/doc_examples.py`（生成/校验文档中的可执行示例）
- 维护校验：`tools/check_contracts.py`（契约块、接口探针、静态页配置、README 入口）

```bash
# 改代码后重新生成文档中的示例块与契约块
python tools/check_contracts.py --generate

# 提交前校验文档与实现一致（pytest 中的 tests/test_contracts.py 会执行同一套检查）
python tools/check_contracts.py --check
```

## 目录结构

```
convo/            # 应用包
  dsp.py          # 均匀分块 overlap-save 卷积（永远返回完整线性卷积）
  pipeline.py     # 渲染管线：重采样、声道映射、预延迟、卷积、干湿混合
  synth.py        # 干声测试音与合成房间 IR
  wavio.py        # WAV 读写与重采样
  app.py          # HTTP API 与静态页
  jobs.py         # 后台渲染执行器（可取消）
  storage.py      # 任务记录与参数快照持久化
  static/index.html
docs/             # 处理契约开发文档（示例与契约块由工具生成）
tools/            # 文档示例校验器与维护校验
tests/            # pytest 测试（含 tests/test_contracts.py 维护检查）
```
