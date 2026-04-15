# ArkMSA CoI阶段完整实现

本代码库实现了论文《Enhancing Multimodal Sentiment Analysis via Learning from Large Language Model》中的CoI（Chain-of-Thought）阶段。

## 📋 目录

- [概述](#概述)
- [安装依赖](#安装依赖)
- [使用方法](#使用方法)
- [代码说明](#代码说明)
- [数据格式](#数据格式)
- [配置说明](#配置说明)
- [注意事项](#注意事项)

## 🎯 概述

CoI阶段的主要任务是：

1. **图像描述生成** (CoT Step 1): 使用多模态大语言模型（如MiniGPT4或GPT-4V）生成图像描述
2. **推理理由生成** (CoT Step 2): 基于文本、图像描述和标签，生成推理理由
3. **数据增强**: 将生成的描述和理由添加到原始数据中，构成可直接训练的数据集

### 论文信息

- **论文标题**: Enhancing Multimodal Sentiment Analysis via Learning from Large Language Model
- **作者**: Ning Pang, Wansen Wu, Yue Hu, Kai Xu, Quanjun Yin, Long Qin
- **GitHub**: https://github.com/ningpang/ArkMSA
- **发表**: IEEE ICME 2024

## 💻 安装依赖

```bash
# 安装必要的Python包
pip install openai>=1.0.0
pip install requests
pip install pillow
pip install numpy
pip install tqdm

# 可选：如果使用MiniGPT4，需要额外安装
# pip install minigpt4  # 需要按照官方文档配置
```

## 🚀 使用方法

### 方法1: 使用OpenAI GPT-4V API

```python
from coi_stage_example import CoIDataGenerator

# 初始化生成器
generator = CoIDataGenerator(
    api_key="your-openai-api-key",  # 或设置环境变量 OPENAI_API_KEY
    image_dir="./twitter15_images",
    output_dir="./output",
    delay=1.0  # API调用间隔
)

# 处理数据
results = generator.process_dataset(
    input_path="./twitter15_raw.json",
    output_filename="tw15_coi_results.json",
    use_openai=True,
    save_interval=10
)
```

### 方法2: 使用命令行

```bash
python coi_stage_example.py \
    --input ./twitter15_raw.json \
    --image_dir ./twitter15_images \
    --output ./output \
    --delay 1.0 \
    --save_interval 10
```

### 方法3: 使用模拟模式（测试用）

```bash
python coi_stage_example.py \
    --input ./twitter15_raw.json \
    --image_dir ./twitter15_images \
    --output ./output \
    --simulate  # 不调用API，使用模拟数据
```

### 方法4: 使用MiniGPT4（需要复杂配置）

```python
from coi_stage_framework import MiniGPT4Adapter, CoIProcessor

# 初始化MiniGPT4适配器
mllm_adapter = MiniGPT4Adapter(
    model_name="mini-gpt4",
    max_tokens=512
)

# 初始化处理器
processor = CoIProcessor(
    mllm_adapter=mllm_adapter,
    image_dir="./twitter15_images",
    output_dir="./output",
    delay=1.0
)

# 加载并处理数据
raw_data = processor.load_original_data("./twitter15_raw.json")
results = processor.process_batch(raw_data)

# 保存结果
processor.save_results(results, "tw15_coi_results.json")
```

## 📖 代码说明

### 核心文件

1. **coi_stage_framework.py**: 完整的CoI阶段框架
   - `MLLMInterface`: 多模态大语言模型接口抽象类
   - `MiniGPT4Adapter`: MiniGPT4适配器
   - `OpenAIAdapter`: OpenAI GPT-4V适配器
   - `CoIProcessor`: CoI阶段处理器

2. **coi_stage_example.py**: 实用示例代码
   - `CoIDataGenerator`: 数据生成器类
   - 支持OpenAI API和模拟模式
   - 包含完整的错误处理和日志记录

### 关键类说明

#### CoIDataGenerator

主要方法：

- `generate_description_cot_step1()`: 生成图像描述
- `generate_rationale_cot_step2()`: 生成推理理由
- `process_sample()`: 处理单个样本
- `process_dataset()`: 处理整个数据集
- `validate_output()`: 验证输出格式

#### MLLM适配器

支持多种MLLM：

- **MiniGPT4**: 论文中使用的开源模型
- **GPT-4V**: OpenAI的多模态模型
- **其他**: 可扩展支持其他多模态模型

## 📊 数据格式

### 输入格式（原始数据）

```json
[
  {
    "text": "RT @FundsOverBuns: $T$ went from pedophile to messing with cougars all within a week",
    "target": "Tyga",
    "gt_label": "0",
    "ImageID": "975807"
  }
]
```

### 输出格式（CoI处理后）

```json
[
  {
    "text": "RT @FundsOverBuns: $T$ went from pedophile to messing with cougars all within a week",
    "target": "Tyga",
    "gt_label": "0",
    "ImageID": "975807",
    "description": "The image shows a woman with black hair and a black dress, possibly at a red carpet event. She has a serious expression and is looking directly at the camera.",
    "reason": "Sentiment: Neutral. The description does not express a clear sentiment towards Tyga, but rather just describes his appearance at an event."
  }
]
```

### 字段说明

| 字段 | 说明 | 类型 |
|------|------|------|
| `text` | 原始推文文本 | string |
| `target` | 目标实体 | string |
| `gt_label` | 真实标签 (0: Neutral, 1: Positive, -1: Negative) | string |
| `ImageID` | 图像ID | string |
| `description` | 图像描述（由MLLM生成） | string |
| `reason` | 推理理由（由MLLM生成） | string |

## ⚙️ 配置说明

### CoT Prompt模板

#### Step 1: 图像描述生成

```
Here are some content that people post on Twitter, and these content are 
composed of text and image. Please note that the text and image may or may 
not be relevant, so make your own judgment. By considering the text that 
<text> and analyzing the image, please give a description about the image 
within <20> words.
```

#### Step 2: 推理理由生成

```
Based on the description: '<description>', and considering the text: '<text>', 
what is the sentiment polarity towards '<target>'? The correct sentiment is 
<label>. Summarize the sentiment polarity, and return only one of these words: 
[<Negative>, <Neutral>, <Positive>]. Make the answer format like: 
[Sentiment: Reason]
```

### 标签映射

| 标签值 | 含义 |
|--------|------|
| "0" | Neutral（中性） |
| "1" | Positive（积极） |
| "-1" | Negative（消极） |

## ⚠️ 注意事项

### 1. API调用限制

- OpenAI API有速率限制，建议设置适当的延迟（如1-2秒）
- 大量数据处理可能需要较长时间
- 建议使用检查点功能，定期保存中间结果

### 2. 图像文件

- 确保图像文件存在于指定目录
- 支持的格式：.jpg, .jpeg, .png, .gif, .bmp
- 图像文件命名应与ImageID一致

### 3. 成本控制

- OpenAI API按token计费
- 建议先使用模拟模式测试
- 批量处理时监控API使用情况

### 4. MiniGPT4配置

MiniGPT4需要复杂的配置：

1. 安装MiniGPT4及其依赖
2. 下载预训练模型权重
3. 配置CUDA环境
4. 可能需要大量GPU内存

建议使用OpenAI API，或参考MiniGPT4官方文档：
https://github.com/Vision-CAIR/MiniGPT-4

## 🔧 自定义扩展

### 添加新的MLLM适配器

```python
from coi_stage_framework import MLLMInterface

class MyCustomMLLM(MLLMInterface):
    def generate_description(self, text: str, image_path: str) -> str:
        # 实现图像描述生成
        pass
    
    def generate_rationale(self, text: str, description: str, target: str, label: str) -> str:
        # 实现推理理由生成
        pass

# 使用自定义适配器
mllm_adapter = MyCustomMLLM()
processor = CoIProcessor(mllm_adapter=mllm_adapter, ...)
```

### 修改Prompt模板

```python
# 在generate_description_cot_step1中修改prompt变量
prompt = f"""Your custom prompt here...
Text: {text}
Generate description within {max_words} words:"""

# 在generate_rationale_cot_step2中修改prompt变量
prompt = f"""Your custom rationale prompt here...
Description: {description}
Text: {text}
Target: {target}
Sentiment: {label_name}"""
```

## 📈 性能优化

### 批量处理

```python
# 使用更大的save_interval减少I/O操作
results = generator.process_dataset(
    input_path="./data.json",
    save_interval=50,  # 每50条保存一次
    batch_size=5  # 如果API支持批量处理
)
```

### 并行处理

```python
from concurrent.futures import ThreadPoolExecutor

# 使用多线程（注意API限制）
with ThreadPoolExecutor(max_workers=2) as executor:
    futures = []
    for sample in data:
        future = executor.submit(generator.process_sample, sample)
        futures.append(future)
    
    results = [f.result() for f in futures if f.result()]
```

## 🐛 常见问题

### Q1: API调用失败怎么办？

A: 
- 检查API密钥是否正确
- 确认账户余额充足
- 检查网络连接
- 增加延迟时间

### Q2: 图像文件找不到？

A:
- 确认图像目录路径正确
- 检查文件扩展名
- 确认文件名与ImageID匹配

### Q3: 生成的描述质量不佳？

A:
- 调整temperature参数
- 修改prompt模板
- 增加max_tokens
- 使用更高质量的模型

### Q4: 处理速度慢？

A:
- 减少delay时间（注意API限制）
- 使用批量处理
- 考虑使用多个API密钥

## 📚 相关资源

- **论文**: https://arxiv.org/abs/xxxx.xxxx
- **GitHub**: https://github.com/ningpang/ArkMSA
- **MiniGPT4**: https://github.com/Vision-CAIR/MiniGPT-4
- **OpenAI API**: https://platform.openai.com/

## 📄 许可证

本项目仅供学术研究使用。

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📧 联系方式

如有问题，请联系：
- 论文作者：pangning14@nudt.edu.cn
- 项目维护：ningpang

---

**注意**: 使用本代码时，请遵守相关API的使用条款和限制。
