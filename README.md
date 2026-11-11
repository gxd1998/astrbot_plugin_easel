# astrbot_plugin_easel 画架

独立的 AstrBot 生图插件：**ComfyUI 原生工作流 + 火山方舟 Seedream（即梦）+ OpenAI 兼容网关**，
LLM 工具与命令双触发，自动故障降级。不依赖任何"陪伴"类主插件，装上即用。

> 命名含义：Easel = 画架。它不做人设、不做聊天，只安静地把画立起来，等模型说话。

| 示例 | 说明 |
|---|---|
| ![ComfyUI 文生图](docs/examples/smoke_t2i.png) | ComfyUI + Qwen-Image 2.1，文生图（油画质感） |
| ![参考图锁脸](docs/examples/smoke_ref.png) | 同一角色换场景：参考图锁脸（海滩 + 草帽） |
| ![即梦 Seedream](docs/examples/ark_plugin.jpg) | 火山方舟 Seedream 5.0-flash，摄影风格 960x1280 |
| ![她画的自己](docs/examples/her_selfie.png) | bot 自主调用工具画出"正在 debug 插件的自己" |

## 特性

- **双触发**：
  - LLM 工具 `paint_generate` / `paint_list_workflows` —— 生成结果先返回给模型，
    模型可用 `file_read_tool` 审图、再用内置 `send_message_to_user` 交付，或按情境改用文字回应；
  - 命令兜底 `/画` —— 跳过大模型，直接生成并立即发送。
- **多后端 + 自动降级**：
  - `comfyui` —— 本地/远程 ComfyUI，吃任意 API 格式工作流（内置 Qwen-Image 2.1 三份工作流）；
  - `ark_images` —— 火山方舟 Seedream（即梦），Bearer API Key；
  - `openai_images` —— OpenAI `/images/generations` 兼容网关（gpt-image-1 等）；
  - `custom_backends` —— 自定义 HTTP 模板，接任何中转站。
  - 主后端失败自动按 `fallback_chain` 降级（如 ComfyUI 挂了自动切即梦）。
- **工作流注册表**：ComfyUI 导出的 API JSON + 一份同名 `.meta.json` 即可注册，
  槽位映射支持自动识别与显式声明，配置可覆盖，`/画图重载` 热加载。
- **参考图锁脸**：消息带图/引用图即作参考图（`LoadImage` / `ETN_LoadImageBase64`），
  固定人设走 `persona` 配置；LoRA 锁脸支持（`LoraLoaderModelOnly` 注入）。
- **视频扩展预留**：后端抽象带 `kind: image|video`，工作流 meta 支持 `VHS_LoadVideo` 槽位。
- **零第三方依赖**：HTTP 全走 AstrBot 自带的 `aiohttp`。

## 安装

```bash
cd AstrBot/data/plugins
git clone https://github.com/vamos/astrbot_plugin_easel
```

在 AstrBot WebUI 中重载插件即可；配置面板（AstrBotConfig schema）自动出现在插件设置页。

## 快速开始

1. 插件设置里填 `comfyui.base_url`（如 `http://192.168.1.4:8188`；ComfyUI 需 `--listen 0.0.0.0` 启动）；
2. 群里发 `/画 一只戴眼镜的鲸鱼，像素风`；
3. 或直接对 bot 说「帮我画一张…」，它会自动选工作流、生成、审图、发送。
4. （可选）接即梦：开通火山方舟 Seedream 后填 `ark_images.enabled/api_key/model`，
   并把 `fallback_chain` 设为 `["ark_images"]`——本地 ComfyUI 挂了自动切云端。

## 命令

| 命令 | 说明 |
|---|---|
| `/画 [参数] 描述` | `--wf=工作流 --size=1:1 --seed=数字 --neg=负向词 --ref=图片路径 --persona --backend=后端` |
| `/画图列表` | 工作流注册表 + 槽位校验状态 |
| `/画图状态` | 最近 5 次生成 + 各后端健康状态 |
| `/画图测试` | 全后端连通性自检（admin） |
| `/画图重载` | 重载工作流注册表（admin） |

`--backend` 显式指定后端（`comfyui` / `ark_images` / `openai_images` / `custom:名字`），
跳过默认后端与 fallback 链。

## 配置速查

| 键 | 说明 |
|---|---|
| `default_backend` | 默认后端：`comfyui` / `ark_images` / `openai_images` / `custom:<名字>` |
| `fallback_chain` | 主后端失败后按序尝试 |
| `access.command_whitelist` | QQ 白名单（群号或 QQ 号），留空 = 不限制 |
| `size_presets` | 画幅预设（比例会锚定到 ~1MP，避免超大尺寸拖慢采样） |
| `comfyui.*` | base_url / 超时 / 轮询间隔 / 步数 / CFG |
| `ark_images.*` | enabled / base_url / api_key / model / watermark |
| `openai_images.*` | enabled / base_url / api_key / model |
| `custom_backends` | 名字 → `{url, headers, body_template, response_path}` |
| `workflows` | 名字 → `{file, description, slots, lora}`，可覆盖内置 meta |
| `persona.*` | 固定人设：workflow + reference_path + extra_prompt |
| `output.keep_days` | 生成图保留天数（存 `data/temp/easel/`，0 = 永久） |

> api_key 明文保存在 `data/config/astrbot_plugin_easel_config.json`，注意不要外传该文件。

## 工作流规范

ComfyUI 菜单「导出（API）」得到 API 格式 JSON，放到
`data/plugin_data/astrbot_plugin_easel/workflows/`，并创建同名 `.meta.json`：

```jsonc
// my_portrait.json.meta.json
{
  "name": "my_portrait",              // 缺省用文件名
  "description": "自定义人设工作流",
  "slots": {
    "text":  [{ "node": "5", "field": "prompt" }],
    "image": [{ "node": "12", "field": "image", "feed_mode": "auto" }]
  },
  "latent_node": "8",                  // 尺寸注入节点（EmptyLatentImage）
  "sampler_node": "3",                 // seed/steps/cfg 注入节点（KSampler）
  "seed_mode": "random",              // random | fixed
  "lora": { "node": "13", "name": "", "strength_model": 1.0 }
}
```

- `slots` 省略时自动识别：文本入口找 `CLIPTextEncode` / `TextEncodeQwenImage21`，
  图片入口找 `LoadImage` / `ETN_LoadImageBase64`，视频入口找 `VHS_LoadVideo`。
- `/画图列表` 会回显每个工作流自动识别的槽位与校验问题。
- **LoRA**：meta.lora 声明注入点；名字在插件配置 `workflows.<名>.lora.name` 里填
  （需与底模架构兼容，如 Qwen-Image 需 Qwen 专训 LoRA），未配置时友好报错、不污染队列。
- **固定人设**：`persona.reference_path` 指向服务器上一张图，`persona.workflow` 用
  参考图工作流；LLM 工具的 `use_persona` 参数 / 命令 `--persona` 一键按人设出图。
  同一张人设图按「路径+大小+mtime」缓存远端文件名，不会重复上传。

## LLM 工具行为

`paint_generate` 的结果是**先返回给模型**的（JSON，含本地 `path`）——由模型决定：
用 `file_read_tool` 审图 → 满意则 `send_message_to_user` 发送，或改用文字回应。
这让"画"成为对话的一部分而不是一条突兀的图片消息。
若想让工具生成后直接发图，把 `tools.return_path_only` 关掉即可。

## 开发

```bash
python tests/test_mapping.py    # 离线单测（不依赖 AstrBot 运行时）
```

结构：

```
main.py            入口：LLM 工具 ×2 + 命令 ×5 + 装配 + 后端降级链
backends/          MediaBackend 抽象 + ComfyUI / Ark / OpenAI / CustomHTTP 实现
workflow/          注册表（内置+用户+配置覆盖）+ 槽位映射（infer/validate/apply/inject_lora）
models.py          MediaRequest / MediaResult / WorkflowDef
```

## License

MIT
