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
- 所有推理分区的主去噪 `transformer` 默认叠加 Sol-Attn、在线 FP8 和 Cache-DiT；Turbo LoRA 以动态模式应用，Audio/Video VAE 等其他组件保持兼容的 FlashAttention。
- SGLang 基础镜像固定到与短边补丁匹配的 `c7c03ec53b` 构建和 OCI digest，不使用会漂移的 `:dev`；已有 GPU 镜像默认复用。
- 业务侧 NFE 支持 `4/6/8`，默认 `6`；转发给 SGLang 时分别是 `5/7/9` 个 sigma grid points。
- 业务分辨率支持 `704P` 和 `768P`。镜像内包含独立的非 768 短边补丁。
- 当前明确不部署 `ref2va`；收到 reference image/video/audio 会返回 400。
- 每个 4 卡分区独立排队、独立故障、独立端口和独立注册健康状态；启动时所有完整分区并行加载和预热。
- 每个对外 API 端口同时发布到主机的 `0.0.0.0` 和 `[::]`；安装器会分别通过 IPv4 loopback 和 IPv6 loopback 探活后才注册服务。
- PyTorch CUDA allocator 默认启用 `expandable_segments`，降低长短视频交替请求造成的显存碎片。
- 独立 Watchdog 逐分区观察任务状态和 worker 日志：存在活跃任务但连续 5 分钟没有处理进展，或出现致命 CUDA OOM 时，只重启对应的 4 卡 worker；重启期间 Reporter 会自动把该端点上报为 unhealthy。
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

`ADVERTISE_HOST` 不接受私网 IP、域名或回退地址。每次安装都会优先查询 AWS IMDSv2 的 `public-ipv4` 和 `instance-id` 并覆盖 AMI 中遗留的旧值；IMDS 不可用时才使用 `.env` 的手工配置。公网 IPv4 继续用于 ReportCatalog 的 `host` 和成功任务的下载 URL；同一 API 端口也会监听主机所有 IPv6 地址。若要从公网 IPv6 访问，还需要 EC2 实例自身分配公网 IPv6，并在安全组中放行对应端口。普通安装按固定 revision 下载 Turbo LoRA 并校验 SHA256；AMI 快速模式只对镜像中已经存在、大小匹配的 LoRA 跳过哈希。

默认模型缓存位于 `${DATA_ROOT}/hf-cache`。默认 `DATA_ROOT=/opt/dlami/nvme/minimax-h3-4h200` 通常属于 EC2 instance-store，不会被标准 AMI 保存。要让克隆实例真正复用基模，制作 AMI 前应把 `MODEL_CACHE_ROOT` 指向快照支持的 EBS 文件系统；快速模式检测到临时 NVMe 时会明确告警，缓存缺失时 SGLang 仍会自动重新下载模型。

常用操作：

```bash
./status.sh
./smoke_test.sh        # 会真实生成一个 704P / 6 NFE / 4 秒视频
./update_api.sh         # 只重建 API 容器，不重启或重新预热 GPU worker
./stop.sh              # 先上报 unhealthy，再停止 reporter；缓存和输出保留
```

### Sol-Attn + FP8 + Cache-DiT 组合优化

默认给每个 4×H200 分区使用同一套优化配置。8 卡机器的两个服务如下：

| 端口 | GPU | 配置 |
| --- | --- | --- |
| `30010` | `0,1,2,3` | Sol-Attn + 在线 FP8 + Cache-DiT + 动态 LoRA |
| `30011` | `4,5,6,7` | Sol-Attn + 在线 FP8 + Cache-DiT + 动态 LoRA |

部署或更新只需要：

```bash
git pull --ff-only
./install.sh
```

安装器会基于 SGLang/Sage 镜像构建独立的 Sol-Attn overlay，并让所有完整的 4 卡分区使用它。全局基础后端设为 Sol，但 `text_encoder`、`audio_vae`、`video_vae` 被显式隔离到兼容后端，只有主 transformer 使用 Sol。每个服务同时传入 `--quantization fp8`，并使用动态 LoRA，避免把 LoRA 增量直接写入量化后的 FP8 基模权重。Cache-DiT 默认参数为 `Fn=1/Bn=0/W=1/R=0.12/MC=3`，面向长视频采用激进缓存策略并允许连续复用三个 step。

安装脚本会等待全部分区 warmup 完成，然后逐个严格校验 Sol/Cache-DiT/FP8 模块、容器环境、实际 SGLang 进程参数及主 DiT 的 Sol 启动日志；任何一个分区没有真正生效都会退出。当前业务默认 6 NFE，Sol 默认 `dense_steps=0`，所有 step 都进入稀疏路径，并将 `tau` 提高到 `1.5`；Sol 的 SM90 kernel 会按 token shape 专门化，15 秒等不同时间长度的 shape 第一次请求仍可能包含 JIT，稳态测速应对相同参数连续运行两次并取第二次。

三项优化都会改变数值路径，组合收益不保证相加。Cache-DiT 只会在第一条真实生成请求开始时挂载，因此启动阶段只验证配置和依赖，实际命中需要查看首条生成后的 worker 日志。

需要整机回退到 Sage/BF16 时执行：

```bash
./disable_sol_ab.sh
```

所有端口仍会照常注册到 `Minimax-H3-AWS-H200`。做严格串行测速时请直接请求一个端口，并确保网关没有同时向这台测试机派发任务。首次验证建议使用显式 seed，记录服务端推理耗时并检查画面和音频；Sol-Attn 的收益主要出现在较长视频，4 秒视频可能被固定开销抵消。

查看单个分区日志：

```bash
sudo docker logs -f minimax-h3-h200-sglang-0
sudo docker logs -f minimax-h3-h200-api-0
sudo docker logs -f minimax-h3-h200-reporter
sudo docker logs -f minimax-h3-h200-cleaner
sudo docker logs -f minimax-h3-h200-watchdog
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

`seed` 可选：省略或传 `null` 时，API 会为每个任务生成独立的 63 位随机 seed；显式传整数时保持可复现。实际使用的 seed 会保存在任务元数据中，并通过查询响应的 `task.seed` 返回。

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
| `SGLANG_BASE_IMAGE` | `nightly-dev-20260812-c7c03ec5@sha256:d753…` | 与短边补丁匹配并锁定 digest 的 SGLang 基础镜像 |
| `REBUILD_GPU_IMAGES` | `0` | 已存在 GPU 镜像时复用；仅修改 GPU Dockerfile/补丁时显式设为 `1` |
| `LORA_REPO` | `larryvrh/MiniMax-H3-Turbo-Lora` | 静态 LoRA 仓库 |
| `LORA_REVISION` | `43a7455…` | 已验证的当前 LoRA commit |
| `LORA_WEIGHT` | `minimax_h3_turbo_v4_step600_ema.safetensors` | 当前 LoRA 文件 |
| `DEFAULT_NFE` | `6` | 业务接口默认实际去噪次数 |
| `SHORT_EDGES` | `480,704` | 在官方 768 之外额外启用的短边 |
| `WARMUP` | `864x480 1248x704 1344x768` | SGLang 启动预热规格 |
| `ATTENTION_BACKEND` | `fa` | 所有组件的安全基础后端，避免 Audio/Video VAE 使用不支持的 SageAttention |
| `COMPONENT_ATTENTION_BACKENDS` | `transformer=sage_attn` | 只把主去噪 transformer 切到 SageAttention |
| `OPTIMIZATION_STACK_ENABLED` | `1` | 是否给全部 4 卡分区启用 Sol-Attn + FP8 + Cache-DiT |
| `SOL_COMPONENT_ATTENTION_BACKENDS` | `text_encoder=torch_sdpa,audio_vae=fa,video_vae=fa,transformer=sol_attn` | H3 DiT 使用 Sol；显式保护文本编码器及 Audio/Video VAE，避免它们误用 Sol |
| `SOL_ATTENTION_BACKEND_CONFIG` | `dense_backend=sage_attn,dense_steps=0,kv_splits=auto,tau=1.5` | Sol 激进稀疏配置；6 NFE 的全部 step 均进入稀疏路径 |
| `SOL_ATTN_STRICT` | `1` | 禁止 Sol kernel 异常时静默回退为 dense，避免产生虚假测速结果 |
| `SOL_WARMUP_STEPS` | `3` | 启动时执行 3 个 warmup step，覆盖 dense 和 sparse 两种 kernel 路径 |
| `SOL_QUANTIZATION` | `fp8` | 全部推理分区在线量化主 transformer |
| `SOL_LORA_MERGE_MODE` | `dynamic` | 动态应用 Turbo LoRA，不修改量化基模权重 |
| `SOL_CACHE_DIT_ENABLED` | `true` | 全部推理分区进程级启用 Cache-DiT |
| `SOL_CACHE_DIT_WARMUP` | `1` | 仅首个去噪 step 强制完整计算 |
| `SOL_CACHE_DIT_RDT` | `0.12` | 激进残差差异缓存阈值，允许更多复用 |
| `SOL_CACHE_DIT_MC` | `3` | 最多连续缓存 3 个 step |
| `REMOTE_MEDIA_HOST_ALLOWLIST` | `.byted.org` | 可访问的私网图片域名后缀；公网域名自动允许 |
| `VIDEO_RETENTION_HOURS` | `12` | 视频和对应任务元数据保留时间 |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | 减少跨请求显存碎片和可恢复性 OOM |
| `WATCHDOG_STALL_SECONDS` | `300` | 有活跃任务但没有状态推进多久后重启对应 worker |
| `WATCHDOG_RESTART_COOLDOWN_SECONDS` | `300` | 同一分区两次自动重启之间的最短间隔 |
| `CLEANUP_INTERVAL_SECONDS` | `600` | 清理任务执行间隔；实际删除可能比 12 小时最多晚约 10 分钟 |
| `DATA_ROOT` | `/opt/dlami/nvme/minimax-h3-4h200` | 与 RTX6000PRO 仓完全分离 |
| `MODEL_CACHE_ROOT` | 空（解析为 `${DATA_ROOT}/hf-cache`） | Hugging Face 基模和 LoRA 缓存；AMI 复用时应指向 EBS |
| `STARTUP_TIMEOUT_SECONDS` | `1800` | 每个 SGLang 分区等待加载和 warmup 的最长秒数 |
| `STARTUP_PROGRESS_SECONDS` | `15` | 等待模型加载和 warmup 时输出一次进度的间隔 |

如果 SGLang 上游代码结构变化，构建阶段会因短边补丁不匹配而失败，不会静默启动一个只支持 768 的服务。

MiniMax H3 的 DiT attention backend 在第一次 forward 时延迟解析。优化分区因此使用全局 `sol_attn`，同时将 `text_encoder`、`audio_vae`、`video_vae` 显式覆盖为兼容后端；启动脚本会同时检查 DiT 实际解析为 Sol 和 Audio VAE 保持 FA，任一不满足都会退出。

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
