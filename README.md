# Media Caption · 本地图片理解工具 / Local Image Understanding Tool

一个**完全本地运行**的图片信息提取工具：用视觉模型（BLIP）生成图片语义描述，用 Windows 自带 OCR 识别图中文字，并提取主色调、亮度、EXIF 等图像特征，把结果注入大模型上下文，让**纯文本大模型也能"看懂"图片**。

A fully **local, offline** image understanding tool: it uses a vision model (BLIP) to generate semantic captions, Windows built-in OCR to extract text, and pulls out color / brightness / EXIF features — then injects everything into an LLM context so a **text-only LLM can "see" images**.

本项目可作为 **DeepSeek Harness 插件** 使用（工具：`read_image_ocr` / `read_image_caption` / `read_file`），也可脱离插件独立用命令行调用。

It works both as a **DeepSeek Harness plugin** (tools: `read_image_ocr` / `read_image_caption` / `read_file`) and as a standalone CLI tool.

---

## ✨ 功能 / Features

| 能力 Capability | 说明 Description | 依赖 Dependencies |
| --- | --- | --- |
| 🖼️ **画面语义描述 Caption** | BLIP 生成图片内容描述（物体/场景/颜色/构图）；GPU/CPU 自动选择。BLIP describes the image (objects/scene/colors/composition); auto CUDA/CPU. | PyTorch + transformers |
| 🔤 **文字识别 OCR** | Windows 自带 OCR 引擎（WinRT），识别图中文字，支持中英文。Windows built-in OCR (WinRT), EN/CN support. | Windows 10/11 |
| 🎨 **图像特征 Features** | 尺寸、格式、主色调占比、平均亮度、EXIF（时间/相机/ISO/曝光）。Size, format, dominant colors, brightness, EXIF metadata. | .NET System.Drawing |
| 📄 **文件读取 File** | 文本文件内容提取，供 LLM 基于内容处理。Extract text file content for LLM. | 无 None |

三路信息（OCR 文字 + 语义描述 + 图像特征）会被打包成一段**专门提示词**，随用户命令一起注入模型。

All three signals (OCR text + semantic caption + image features) are packed into a **dedicated prompt** and injected into the model together with the user's command.

---

## 📁 目录结构 / Directory Layout

```
media-caption-release/
├── README.md               # 本文件 This file
├── requirements.txt        # Python 依赖 Dependencies
├── .gitattributes          # Git LFS 配置 Git LFS config
├── caption.py              # 命令行图片描述脚本 CLI caption script (BLIP)
├── example.py              # 独立使用示例 Standalone example
└── model/                  # 模型目录（标准 transformers 格式）Model dir (transformers format)
    ├── config.json
    ├── preprocessor_config.json
    ├── tokenizer_config.json
    ├── tokenizer.json
    ├── vocab.txt
    ├── special_tokens_map.json
    └── model.safetensors   # 约 Approx. 944MB
```

---

## 🔧 环境要求 / Requirements

- **Python 3.10+**（已在本机 3.13 验证 / verified on 3.13）
- 推理框架 / Inference stack（二选一 either）：
  - **CPU**：`pip install torch`（CPU 版即可，约 0.9s/张 approx. 0.9s/image）
  - **GPU**：推荐 `torch cu128`（支持 Blackwell / RTX 50 系；recommended for Blackwell / RTX 50-series），显存 ≥ 4GB
- Windows 10/11（仅 OCR 部分需要 / only needed for OCR）

### 安装 / Install

```bash
pip install -r requirements.txt
# GPU 用户建议单独安装匹配显卡架构的 torch / For GPU, install the build matching your GPU:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

> ⚠️ **Windows / Anaconda 注意 / Note**：若 `import torch` 报 `OMP Error #15`（libiomp5md.dll 重复初始化），
> 设置环境变量 `KMP_DUPLICATE_LIB_OK=TRUE` 即可（脚本已内置该逻辑 / the script sets it already）。

---

## 🚀 使用 / Usage

### 命令行 / CLI（独立使用 standalone）

```bash
python caption.py "C:\path\to\image.png"
```

输出一行 JSON / prints one JSON line：

```json
{"ok": true, "caption": "a red circle and green square on a blue background", "device": "cuda", "load_s": 3.7, "infer_s": 0.45}
```

首次运行会自动从 [HuggingFace](https://huggingface.co/Salesforce/blip-image-captioning-base) 下载模型（约 944MB，存到 `model/`）；已存在则直接加载。
On first run the model is downloaded from HuggingFace (~944MB, into `model/`); afterwards it loads locally.

### 作为 DeepSeek Harness 插件 / As a DSH plugin（推荐 recommended）

在会话中直接下指令，模型会自动调用工具。Just ask in the session — the model calls the tools for you.

```
看一下 D:\demo.png，里面有什么？
读一下 D:\demo.txt，然后总结
```

| 工具 Tool | 作用 Purpose |
| --- | --- |
| `read_image_ocr` | OCR 文字 + 图像特征（主色/亮度/EXIF）。OCR text + image features. |
| `read_image_caption` | BLIP 生成画面语义描述。BLIP semantic caption. |
| `read_file` | 读取文本文件内容。Read text file content. |
| `clear_media_context` | 清空已注入的上下文。Clear injected context. |

每次调用都会把「内容 + 专门提示词」注入 system prompt，与用户命令一起交给模型，后续多轮对话可持续引用。
Every call injects the content plus a dedicated prompt into the system prompt, so later turns can keep referencing it.

### Python 调用 / Python API

```python
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

processor = BlipProcessor.from_pretrained("./model")
model = BlipForConditionalGeneration.from_pretrained("./model").to("cuda" if torch.cuda.is_available() else "cpu")
image = Image.open("demo.png").convert("RGB")
inputs = processor(image, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=80)
print(processor.decode(out[0], skip_special_tokens=True))
```

---

## ❓ 常见问题 / FAQ

- **GPU 报 `no kernel image is available`**：显卡架构与 torch 编译目标不匹配（如 Blackwell 卡 + 老 cu126 torch）。
  安装 cu128 版 torch 即可；脚本检测到 GPU 不可用时会自动回退 CPU。
  *GPU reports `no kernel image is available`: the GPU architecture doesn't match the torch build
  (e.g. Blackwell + cu126 torch). Install the cu128 build; the script falls back to CPU automatically.*
- **OCR 识别错乱 / noisy OCR**：Windows OCR 对中文小字/艺术字效果一般；带文字的图优先用 `read_image_ocr`，画面理解用 `read_image_caption`。
  *Windows OCR struggles with small/artistic CJK text; use `read_image_ocr` for text-heavy images and `read_image_caption` for scenes.*
- **模型下载慢 / slow download**：可设置 `HF_ENDPOINT=https://hf-mirror.com` 使用国内镜像。
  *Set `HF_ENDPOINT=https://hf-mirror.com` for a China mirror.*
- **想换更强模型 / want a stronger model**：Florence-2、LLaVA 等同样适用，只需改 `caption.py` 的模型类与加载逻辑。
  *Florence-2, LLaVA, etc. work too — just swap the model class and loading logic in `caption.py`.*

---

## 📄 License / 许可

- 代码 / Code：MIT
- 模型 / Model：BLIP（[Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base)），遵循其原始许可 / under its original license.
