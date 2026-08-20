# MiniMax H3 — 4×H200 自动部署

这是独立的 4×H200 MiniMax-H3 部署仓库，不引用 `../rtx6000pro` 的代码、镜像或数据目录。

部署器会读取 `nvidia-smi`，每连续 4 张 H200 创建一个完全独立的推理服务：

| 物理 GPU | SGLang | 对外 API | 注册实例 |
| --- | --- | --- | --- |
| `0,1,2,3` | `h3-sglang-0` | 公网 IP:`30010` | `<instance-id>-4h200-0` |
| `4,5,6,7` | `h3-sglang-1` | 公网 IP:`30011` | `<instance-id>-4h200-1` |

因此 8 卡机器会向 `Minimax-H3-AWS-H200` 注册两个实例，而不是一个 8 卡实例。若 GPU 总数不是 4 的倍数，只使用完整分组；默认还会校验每张卡的名称包含 `H200`。

## 当前能力

- SGLang `fl2va` variant，同时支持 T2V、首帧、尾帧和首尾帧生成。
- 静态加载 `larryvrh/MiniMax-H3-Turbo-Lora` 当前 v4 权重。
- 业务侧 NFE 支持 `4/6/8`，默认 `6`；转发给 SGLang 时分别是 `5/7/9` 个 sigma grid points。
- 业务分辨率支持 `704P` 和 `768P`。镜像内包含独立的非 768 短边补丁。
- 当前明确不部署 `ref2va`；收到 reference image/video/audio 会返回 400。
- 每个 4 卡分区独立排队、独立故障、独立端口和独立注册健康状态。
- 图片 URL 默认只允许公网解析地址或 `.byted.org`，避免推理容器访问云元数据和内网地址。

## 部署

机器需要已安装可用的 NVIDIA 驱动，推荐直接使用 AWS GPU/DLAMI。首次执行会自动创建 `.env`，通过 AWS IMDSv2 获取公网 IP 和实例 ID，生成 API Key，安装 Docker、Compose v2 与 NVIDIA Container Toolkit，下载并校验 LoRA，然后构建、启动并注册服务。

```bash
git clone git@github.com:wclmxxs/4h200.git
cd 4h200
./install.sh
```

默认不需要编辑配置。只有非标准环境才需要先运行 `cp config/env.example .env` 再修改，例如无法访问 AWS IMDS、需要 Hugging Face Token，或需要更换数据盘目录。

`ADVERTISE_HOST` 不接受私网 IP、域名或回退地址：默认仅查询 AWS IMDSv2 的 `public-ipv4`，查不到就终止部署并提示显式填写公网 IPv4。这个值同时用于 ReportCatalog 的 `host` 和成功任务的下载 URL。安装过程会按固定 revision 下载 Turbo LoRA、校验 SHA256，再把本地 snapshot 路径交给 SGLang；重复安装复用缓存。

常用操作：

```bash
./status.sh
./smoke_test.sh        # 会真实生成一个 704P / 6 NFE / 4 秒视频
./stop.sh              # 先上报 unhealthy，再停止 reporter；缓存和输出保留
```

查看单个分区日志：

```bash
sudo docker logs -f minimax-h3-h200-sglang-0
sudo docker logs -f minimax-h3-h200-api-0
sudo docker logs -f minimax-h3-h200-reporter
```

## 注册协议

Reporter 每 5 秒请求：

```http
POST /ic/capcut/edit_gateway/v1/report_catalog
Content-Type: application/json
X-Internal-Auth: bernard-edit-bridge-internal-call
```

请求体：

```json
{
  "psm": "capcut.ai_infra.federation",
  "service_id": "Minimax-H3-AWS-H200",
  "instances_json": "[{\"id\":\"i-xxx-4h200-0\",\"host\":\"16.78.214.130\",\"ports\":[30010],\"state\":\"TASK_RUNNING\",\"healthCheckResults\":[{\"alive\":true}],\"containerInfos\":{\"h3-4h200-0\":{\"request\":{\"cpu\":48,\"memory\":512000,\"nvidia.com/gpu\":4}}}}]"
}
```

同一机器的两个实例使用相同公网 `host` 和不同 `port`。Reporter 从 Docker 内网探测对应 API 的 `/healthz`；SGLang 不健康时该分区下一次上报 `alive=false`，不会影响另一个分区。

## 对外业务接口

接口结构与 RTX6000PRO 版本一致。

提交：

```http
POST /ic/capcut/edit_gateway/v2/video_generation
Content-Type: application/json
```

T2V 示例：

```json
{
  "model": "MiniMax-H3",
  "content": [
    {"type": "text", "text": "A cinematic sunrise over a quiet lake."}
  ],
  "resolution": "768P",
  "duration": 5,
  "ratio": "16:9",
  "num_inference_steps": 6,
  "seed": 42
}
```

FL2V 示例：

```json
{
  "model": "MiniMax-H3",
  "content": [
    {"type": "text", "text": "The camera slowly pushes toward the subject."},
    {
      "type": "image_url",
      "role": "first_frame",
      "image_url": {"url": "https://example.com/first.jpg"}
    },
    {
      "type": "image_url",
      "role": "last_frame",
      "image_url": {"url": "https://example.com/last.jpg"}
    }
  ],
  "resolution": "704P",
  "duration": 5,
  "ratio": "adaptive",
  "num_inference_steps": 6
}
```

返回：

```json
{"task_id": "video_xxx"}
```

查询：

```http
POST /ic/capcut/edit_gateway/v2/query/video_generation
Content-Type: application/json

{"model":"MiniMax-H3","task_id":"video_xxx"}
```

同步接口也保留：

- `POST /sync_infer`
- `POST /ic/capcut/edit_gateway/v2/sync_infer`
- `GET /ic/capcut/edit_gateway/v2/video_generation/{task_id}/content`

另外提供带 Bearer API Key 的 SGLang 兼容代理：

- `GET /healthz`
- `POST /v1/videos`
- `GET /v1/videos/{task_id}`
- `DELETE /v1/videos/{task_id}`
- `GET /v1/videos/{task_id}/content`

业务接口保持与现有 gateway 相同的无 API Key 调用方式；内部健康检查和 `/v1` 兼容接口使用 `.env` 中自动生成的 `API_KEY`。

## 主要配置

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `SERVICE_ID` | `Minimax-H3-AWS-H200` | install 会强校验，防止注册到错误池子 |
| `API_BASE_PORT` | `30010` | 第 N 个 4 卡分区使用 `base+N` |
| `MODEL` | `MiniMaxAI/MiniMax-H3` | 基模；部署命令始终显式传入 |
| `LORA_REPO` | `larryvrh/MiniMax-H3-Turbo-Lora` | 静态 LoRA 仓库 |
| `LORA_REVISION` | `43a7455…` | 已验证的当前 LoRA commit |
| `LORA_WEIGHT` | `minimax_h3_turbo_v4_step600_ema.safetensors` | 当前 LoRA 文件 |
| `DEFAULT_NFE` | `6` | 业务接口默认实际去噪次数 |
| `SHORT_EDGES` | `480,704` | 在官方 768 之外额外启用的短边 |
| `WARMUP` | `864x480 1248x704 1344x768` | SGLang 启动预热规格 |
| `REMOTE_MEDIA_HOST_ALLOWLIST` | `.byted.org` | 可访问的私网图片域名后缀；公网域名自动允许 |
| `DATA_ROOT` | `/opt/dlami/nvme/minimax-h3-4h200` | 与 RTX6000PRO 仓完全分离 |
| `SGLANG_BASE_IMAGE` | `lmsysorg/sglang:dev` | 建议验证后换成 digest 固定的镜像引用 |

如果 SGLang 上游代码结构变化，构建阶段会因短边补丁不匹配而失败，不会静默启动一个只支持 768 的服务。

## 本地验证

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest
bash -n install.sh status.sh stop.sh smoke_test.sh scripts/bootstrap_host.sh
python3 -m py_compile api/app/*.py reporter/main.py scripts/generate_compose.py
```
