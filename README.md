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
- 主去噪 `transformer` 默认启用 SageAttention，并固定包含 Hopper/SM90 修复的 upstream revision；Audio/Video VAE 等其他组件保持兼容的 FlashAttention。
- 业务侧 NFE 支持 `4/6/8`，默认 `6`；转发给 SGLang 时分别是 `5/7/9` 个 sigma grid points。
- 业务分辨率支持 `704P` 和 `768P`。镜像内包含独立的非 768 短边补丁。
- 当前明确不部署 `ref2va`；收到 reference image/video/audio 会返回 400。
- 每个 4 卡分区独立排队、独立故障、独立端口和独立注册健康状态；启动时所有完整分区并行加载和预热。
- 图片 URL 默认只允许公网解析地址或 `.byted.org`，避免推理容器访问云元数据和内网地址。
- 生成视频和对应任务元数据保留 12 小时；独立 cleaner 每 10 分钟清理一次，不触碰模型与 LoRA 缓存。

## 部署

机器需要已安装可用的 NVIDIA 驱动，推荐直接使用 AWS GPU/DLAMI。首次执行会自动创建 `.env`，通过 AWS IMDSv2 获取公网 IP 和实例 ID，生成 API Key，安装 Docker、Compose v2 与 NVIDIA Container Toolkit，下载并校验 LoRA，然后构建、启动并注册服务。

```bash
git clone git@github.com:wclmxxs/4h200.git
cd 4h200
./install.sh
```

从已经完整部署并验证过的 EC2 AMI 创建新实例时，使用快速启动模式：

```bash
./install.sh --from-ami
```

快速模式会立即停止镜像中自动启动的旧 Reporter，强制从 AWS IMDS 刷新公网 IP 和 instance-id，复用已有 Docker 镜像，并对已缓存且大小匹配的 LoRA 跳过 SHA256。它仍会重新生成 Compose、把模型加载到所有 GPU、执行 SGLang warmup，最后以新身份注册。GPU 显存状态无法保存在 AMI 中，因此模型加载和 warmup 不能跳过。

制作 AMI 前先执行一次冻结脚本。它会先停 Reporter，再停止全部部署容器，但保留 Docker 镜像、模型缓存和视频数据。这样克隆机开机时不会让旧 Reporter 抢先用源机器身份注册：

```bash
./prepare_ami.sh
```

脚本输出 `AMI_READY` 后再创建 AMI。克隆机仍然只需要 `git pull --ff-only && ./install.sh --from-ami`；如果不创建 AMI 而要恢复源机器，也执行同一条快速安装命令。

默认不需要编辑配置。只有非标准环境才需要先运行 `cp config/env.example .env` 再修改，例如无法访问 AWS IMDS、需要 Hugging Face Token，或需要更换数据盘目录。

`ADVERTISE_HOST` 不接受私网 IP、域名或回退地址。每次安装都会优先查询 AWS IMDSv2 的 `public-ipv4` 和 `instance-id` 并覆盖 AMI 中遗留的旧值；IMDS 不可用时才使用 `.env` 的手工配置。公网 IP 同时用于 ReportCatalog 的 `host` 和成功任务的下载 URL。普通安装按固定 revision 下载 Turbo LoRA 并校验 SHA256；AMI 快速模式只对镜像中已经存在、大小匹配的 LoRA 跳过哈希。

默认模型缓存位于 `${DATA_ROOT}/hf-cache`。默认 `DATA_ROOT=/opt/dlami/nvme/minimax-h3-4h200` 通常属于 EC2 instance-store，不会被标准 AMI 保存。要让克隆实例真正复用基模，制作 AMI 前应把 `MODEL_CACHE_ROOT` 指向快照支持的 EBS 文件系统；快速模式检测到临时 NVMe 时会明确告警，缓存缺失时 SGLang 仍会自动重新下载模型。

常用操作：

```bash
./status.sh
./smoke_test.sh        # 会真实生成一个 704P / 6 NFE / 4 秒视频
./update_api.sh         # 只重建 API 容器，不重启或重新预热 GPU worker
./stop.sh              # 先上报 unhealthy，再停止 reporter；缓存和输出保留
```

### Sol-Attn 稀疏注意力 A/B

8 卡机器可以让两组 4×H200 同时保留不同的 attention 路径：

| 端口 | GPU | 配置 |
| --- | --- | --- |
| `30010` | `0,1,2,3` | 原有 SageAttention 基线，不重启 |
| `30011` | `4,5,6,7` | Sol-Attn；前 2 个去噪 step 使用 Sage dense，后续 step 使用稀疏 attention |

启用只需要：

```bash
git pull --ff-only
./enable_sol_ab.sh
```

第一次会基于现有 SGLang/Sage 镜像构建一个独立 Sol-Attn overlay，然后只重建 GPU 4–7 的 worker 和对应的轻量 API 容器。脚本会等待 warmup 完成，并校验 Sol 包、实际镜像、运行参数及启动日志；GPU 0–3 的基线服务及其显存不会被触碰。当前业务默认 6 NFE，因此默认 `dense_steps=2`，剩余 4 step 才真正进入稀疏路径。Sol 分区使用 `warmup_steps=3`，确保启动预热至少执行一次稀疏 kernel，避免首个正式请求承担 JIT 开销。

回滚也只重启 GPU 4–7：

```bash
./disable_sol_ab.sh
```

两组端口仍会照常注册到 `Minimax-H3-AWS-H200`。做严格串行测速时请直接请求 `30010` 和 `30011`，并确保网关没有同时向这台测试机派发任务。首次验证建议对相同 seed、prompt、时长和分辨率分别比较服务端推理耗时与画面质量；Sol-Attn 的收益主要出现在较长视频，4 秒视频可能被固定开销抵消。

需要重建相同 tag 的 Sol 镜像时：

```bash
FORCE_BUILD_SOL=1 ./enable_sol_ab.sh
```

查看单个分区日志：

```bash
sudo docker logs -f minimax-h3-h200-sglang-0
sudo docker logs -f minimax-h3-h200-api-0
sudo docker logs -f minimax-h3-h200-reporter
sudo docker logs -f minimax-h3-h200-cleaner
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
| `ATTENTION_BACKEND` | `fa` | 所有组件的安全基础后端，避免 Audio/Video VAE 使用不支持的 SageAttention |
| `COMPONENT_ATTENTION_BACKENDS` | `transformer=sage_attn` | 只把主去噪 transformer 切到 SageAttention |
| `SOL_AB_ENABLED` | `0` | 是否在完整安装时启用分区级 Sol-Attn A/B；一键脚本会自动维护 |
| `SOL_AB_SLOT` | `1` | Sol 实验占用的 4 卡分区；禁止使用 slot 0，以保留稳定基线 |
| `SOL_ATTENTION_BACKEND_CONFIG` | `dense_backend=sage_attn,dense_steps=2,kv_splits=auto,tau=1.0` | Sol 稀疏配置；6 NFE 下前 2 step 保持 dense |
| `SOL_ATTN_STRICT` | `1` | 实验分区禁止 Sol kernel 异常时静默回退为 dense，避免产生虚假测速结果 |
| `SOL_WARMUP_STEPS` | `3` | 启动时执行 3 个 warmup step，覆盖 `dense_steps=2` 后的首个稀疏 step |
| `REMOTE_MEDIA_HOST_ALLOWLIST` | `.byted.org` | 可访问的私网图片域名后缀；公网域名自动允许 |
| `VIDEO_RETENTION_HOURS` | `12` | 视频和对应任务元数据保留时间 |
| `CLEANUP_INTERVAL_SECONDS` | `600` | 清理任务执行间隔；实际删除可能比 12 小时最多晚约 10 分钟 |
| `DATA_ROOT` | `/opt/dlami/nvme/minimax-h3-4h200` | 与 RTX6000PRO 仓完全分离 |
| `MODEL_CACHE_ROOT` | 空（解析为 `${DATA_ROOT}/hf-cache`） | Hugging Face 基模和 LoRA 缓存；AMI 复用时应指向 EBS |
| `STARTUP_TIMEOUT_SECONDS` | `1800` | 每个 SGLang 分区等待加载和 warmup 的最长秒数 |
| `STARTUP_PROGRESS_SECONDS` | `15` | 等待模型加载和 warmup 时输出一次进度的间隔 |
| `SGLANG_BASE_IMAGE` | `lmsysorg/sglang:dev` | 建议验证后换成 digest 固定的镜像引用 |

如果 SGLang 上游代码结构变化，构建阶段会因短边补丁不匹配而失败，不会静默启动一个只支持 768 的服务。

SageAttention 会改变 transformer 的 attention 数值路径。需要让所有组件回退到 FlashAttention 时清空组件覆盖后重新执行安装：

```bash
sed -i 's/^ATTENTION_BACKEND=.*/ATTENTION_BACKEND=fa/' .env
sed -i 's/^COMPONENT_ATTENTION_BACKENDS=.*/COMPONENT_ATTENTION_BACKENDS=/' .env
./install.sh
```

## 本地验证

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest
bash -n install.sh enable_sol_ab.sh disable_sol_ab.sh prepare_ami.sh update_api.sh status.sh stop.sh smoke_test.sh scripts/bootstrap_host.sh scripts/configure_sol_ab.sh
python3 -m py_compile api/app/*.py reporter/main.py scripts/generate_compose.py
```
