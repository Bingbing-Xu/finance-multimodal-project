# ArkMSA CoI阶段使用指南

## 📚 目录

1. [快速开始](#快速开始)
2. [详细教程](#详细教程)
3. [高级用法](#高级用法)
4. [常见问题](#常见问题)
5. [性能优化](#性能优化)

---

## 🚀 快速开始

### 第1步：安装依赖

```bash
pip install -r requirements.txt
```

### 第2步：准备数据

创建以下目录结构：

```
project/
├── images/          # 存放图像文件
│   ├── 123456.jpg
│   ├── 789012.png
│   └── ...
├── data/            # 存放数据文件
│   └── twitter15_raw.json
├── output/          # 输出目录（自动生成）
└── coi_stage_example.py
```

### 第3步：运行代码

#### 方法A：使用快速开始脚本

```bash
python quick_start.py
```

#### 方法B：直接运行示例

```python
from coi_stage_example import CoIDataGenerator

# 创建生成器
generator = CoIDataGenerator(
    image_dir="./images",
    output_dir="./output",
    use_openai=False  # 使用模拟模式
)

# 处理数据
results = generator.process_file(
    input_path="./data/twitter15_raw.json",
    output_filename="coi_results.json"
)
```

### 第4步：验证结果

```python
# 验证输出格式
is_valid = generator.validate_result("./output/coi_results.json")
print(f"格式验证: {'通过' if is_valid else '失败'}")
```

---

## 📖 详细教程

### 1. 数据准备

#### 输入数据格式

```json
[
  {
    "text": "I love the new $T$! It's amazing!",
    "target": "iPhone",
    "gt_label": "1",
    "ImageID": "123456"
  }
]
```

#### 图像文件命名

- 图像文件应与`ImageID`对应
- 支持的格式：`.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`
- 例如：`ImageID`为"123456"，图像文件可以是"123456.jpg"

### 2. 使用OpenAI API

#### 设置API密钥

```bash
# 方法1: 环境变量
export OPENAI_API_KEY="sk-your-api-key"

# 方法2: 代码中设置
import os
os.environ["OPENAI_API_KEY"] = "sk-your-api-key"
```

#### 调用API

```python
generator = CoIDataGenerator(
    api_key="sk-your-api-key",  # 或从环境变量读取
    image_dir="./images",
    output_dir="./output",
    delay=1.0  # API调用间隔
)

results = generator.process_dataset(
    input_path="./twitter15_raw.json",
    output_filename="tw15_coi_results.json",
    use_openai=True,
    save_interval=10  # 每10条保存一次
)
```

### 3. 使用MiniGPT4

#### 安装MiniGPT4

```bash
# 克隆MiniGPT4仓库
git clone https://github.com/Vision-CAIR/MiniGPT-4.git
cd MiniGPT-4

# 安装依赖
pip install -r requirements.txt

# 下载预训练权重（需要按照官方文档操作）
```

#### 使用MiniGPT4适配器

```python
from coi_stage_framework import MiniGPT4Adapter, CoIProcessor

# 初始化适配器
mllm_adapter = MiniGPT4Adapter(
    model_name="mini-gpt4",
    max_tokens=512
)

# 创建处理器
processor = CoIProcessor(
    mllm_adapter=mllm_adapter,
    image_dir="./images",
    output_dir="./output",
    delay=2.0  # MiniGPT4可能需要更长时间
)

# 处理数据
raw_data = processor.load_original_data("./twitter15_raw.json")
results = processor.process_batch(raw_data)
processor.save_results(results, "tw15_coi_results.json")
```

### 4. 批量处理多个文件

```python
import os
from coi_stage_example import CoIDataGenerator

# 创建生成器
generator = CoIDataGenerator(
    image_dir="./images",
    output_dir="./output",
    delay=1.0
)

# 处理多个数据集
datasets = [
    ("./twitter15_raw.json", "tw15_coi_results.json"),
    ("./twitter17_raw.json", "tw17_coi_results.json"),
    ("./mvsa_single_raw.json", "mvsa_single_coi_results.json"),
]

for input_path, output_name in datasets:
    if os.path.exists(input_path):
        logger.info(f"处理: {input_path}")
        results = generator.process_dataset(
            input_path=input_path,
            output_filename=output_name,
            use_openai=True,
            save_interval=20
        )
        logger.info(f"完成: {len(results)} 条数据")
```

### 5. 从检查点恢复

```python
# 检查点文件包含部分处理结果
checkpoint_path = "./output/checkpoint_50.json"

# 加载已处理的数据
with open(checkpoint_path, 'r', encoding='utf-8') as f:
    processed_data = json.load(f)

# 继续处理剩余数据
start_idx = len(processed_data)
remaining_data = raw_data[start_idx:]

# 继续处理
results = generator.process_dataset(
    input_path=remaining_data,
    start_idx=start_idx,
    save_interval=10
)
```

---

## 🔧 高级用法

### 1. 自定义Prompt模板

```python
class MyCoIGenerator(CoIDataGenerator):
    def generate_description_cot_step1(self, text: str, image_path: str) -> str:
        # 自定义Step 1 prompt
        prompt = f"""Analyze this image in the context of the tweet: '{text}'
        Provide a detailed description focusing on sentiment-relevant elements."""
        
        # 调用API
        return self.call_openai_api(prompt, image_path)
    
    def generate_rationale_cot_step2(self, text: str, description: str, target: str, label: str) -> str:
        # 自定义Step 2 prompt
        prompt = f"""Given the tweet: '{text}'
        And image description: '{description}'
        Explain why the sentiment towards '{target}' is {label}."""
        
        return self.call_openai_api(prompt)
```

### 2. 添加自定义字段

```python
def add_custom_fields(sample: Dict[str, Any]) -> Dict[str, Any]:
    """添加自定义字段"""
    # 添加时间戳
    from datetime import datetime
    sample['timestamp'] = datetime.now().isoformat()
    
    # 添加数据质量评分
    sample['quality_score'] = calculate_quality_score(sample)
    
    # 添加来源信息
    sample['source'] = 'CoI_Stage_v1.0'
    
    return sample

# 在保存前添加自定义字段
for result in results:
    result = add_custom_fields(result)
```

### 3. 数据过滤和清洗

```python
def filter_low_quality_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """过滤低质量样本"""
    filtered = []
    
    for sample in samples:
        # 检查描述长度
        if len(sample['description']) < 10:
            logger.warning(f"跳过样本 {sample['ImageID']}: 描述太短")
            continue
        
        # 检查理由格式
        if not sample['reason'].startswith('Sentiment:'):
            logger.warning(f"跳过样本 {sample['ImageID']}: 理由格式不正确")
            continue
        
        filtered.append(sample)
    
    return filtered

# 过滤结果
results = filter_low_quality_samples(results)
```

### 4. 并行处理

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 创建线程锁
lock = threading.Lock()

def process_sample_safe(sample: Dict[str, Any]) -> Dict[str, Any]:
    """线程安全的样本处理"""
    result = generator.process_sample(sample, use_openai=True)
    
    # 线程安全地保存结果
    with lock:
        save_to_file(result, "partial_results.jsonl", append=True)
    
    return result

# 使用线程池
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = {
        executor.submit(process_sample_safe, sample): i 
        for i, sample in enumerate(data)
    }
    
    for future in as_completed(futures):
        idx = futures[future]
        try:
            result = future.result()
            logger.info(f"完成样本 {idx}")
        except Exception as e:
            logger.error(f"样本 {idx} 处理失败: {e}")
```

### 5. 集成到训练流程

```python
def prepare_training_data(coi_results_path: str, output_path: str):
    """准备训练数据"""
    with open(coi_results_path, 'r', encoding='utf-8') as f:
        coi_data = json.load(f)
    
    training_data = []
    
    for item in coi_data:
        # 构建训练样本
        training_sample = {
            'input_text': f"{item['text']} [SEP] {item['description']}",
            'target': item['target'],
            'label': item['gt_label'],
            'rationale': item['reason']
        }
        
        training_data.append(training_sample)
    
    # 保存训练数据
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"训练数据已保存: {output_path}")
```

---

## ❓ 常见问题

### Q1: API调用失败怎么办？

**A:**

1. **检查API密钥**
   ```python
   import os
   print(os.getenv("OPENAI_API_KEY"))  # 应该显示您的密钥
   ```

2. **检查账户余额**
   - 登录OpenAI平台查看API使用情况
   - 确保账户有足够余额

3. **检查网络连接**
   - 确保可以访问OpenAI API
   - 检查防火墙设置

4. **增加延迟**
   ```python
   generator = CoIDataGenerator(
       image_dir="./images",
       output_dir="./output",
       delay=2.0  # 增加到2秒
   )
   ```

5. **重试机制**
   ```python
   import time
   
   def call_api_with_retry(func, max_retries=3):
       for i in range(max_retries):
           try:
               return func()
           except Exception as e:
               if i < max_retries - 1:
                   time.sleep(2 ** i)  # 指数退避
               else:
                   raise e
   ```

### Q2: 图像文件找不到？

**A:**

1. **检查文件路径**
   ```python
   # 打印所有图像文件
   import os
   image_files = os.listdir("./images")
   print(f"找到 {len(image_files)} 个图像文件")
   print("前10个:", image_files[:10])
   ```

2. **检查文件命名**
   - 确保ImageID与文件名匹配
   - 检查文件扩展名

3. **使用调试模式**
   ```python
   generator = CoIDataGenerator(
       image_dir="./images",
       output_dir="./output"
   )
   
   # 测试单个图像
   path = generator.get_image_path("123456")
   print(f"图像路径: {path}")
   print(f"文件存在: {os.path.exists(path)}")
   ```

### Q3: 生成的描述质量不佳？

**A:**

1. **调整temperature参数**
   ```python
   response = self.client.chat.completions.create(
       model="gpt-4-vision-preview",
       messages=[...],
       temperature=0.8,  # 增加随机性
       max_tokens=200    # 增加长度
   )
   ```

2. **优化Prompt**
   - 提供更详细的上下文
   - 明确指定输出格式
   - 添加示例

3. **后处理**
   ```python
   def post_process_description(description: str) -> str:
       # 移除不必要的词语
       description = description.replace("In the image, ", "")
       description = description.replace("The image shows ", "")
       
       # 确保长度合适
       if len(description) > 100:
           description = description[:100] + "..."
       
       return description.strip()
   ```

### Q4: 处理速度慢？

**A:**

1. **减少延迟**
   ```python
   generator = CoIDataGenerator(
       image_dir="./images",
       output_dir="./output",
       delay=0.5  # 减少到0.5秒（注意API限制）
   )
   ```

2. **批量处理**
   ```python
   # 如果API支持批量处理
   batch_size = 5
   for i in range(0, len(data), batch_size):
       batch = data[i:i+batch_size]
       results = process_batch(batch)
   ```

3. **使用异步调用**
   ```python
   import asyncio
   import aiohttp
   
   async def process_async(sample):
       # 异步处理
       pass
   
   async def main():
       tasks = [process_async(sample) for sample in data]
       results = await asyncio.gather(*tasks)
   ```

### Q5: 输出格式不正确？

**A:**

1. **检查数据类型**
   ```python
   # 确保所有字段都是字符串
   sample = {
       "text": str(sample.get('text', '')),
       "target": str(sample.get('target', '')),
       "gt_label": str(sample.get('gt_label', '0')),
       "ImageID": str(sample.get('ImageID', '')),
       "description": str(description),
       "reason": str(reason)
   }
   ```

2. **验证输出**
   ```python
   def validate_output(data: List[Dict[str, Any]]) -> bool:
       required_fields = ['text', 'target', 'gt_label', 'ImageID', 'description', 'reason']
       
       for item in data:
           for field in required_fields:
               if field not in item or not isinstance(item[field], str):
                   logger.error(f"字段 {field} 无效")
                   return False
       
       return True
   ```

3. **使用JSON Schema**
   ```python
   from jsonschema import validate
   
   schema = {
       "type": "array",
       "items": {
           "type": "object",
           "properties": {
               "text": {"type": "string"},
               "target": {"type": "string"},
               "gt_label": {"type": "string"},
               "ImageID": {"type": "string"},
               "description": {"type": "string"},
               "reason": {"type": "string"}
           },
           "required": ["text", "target", "gt_label", "ImageID", "description", "reason"]
       }
   }
   
   validate(instance=data, schema=schema)
   ```

---

## 📈 性能优化

### 1. 缓存机制

```python
import hashlib
import pickle
from functools import lru_cache

class CachedCoIGenerator(CoIDataGenerator):
    def __init__(self, *args, cache_dir="./cache", **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_key(self, text: str, image_id: str) -> str:
        """生成缓存键"""
        content = f"{text}_{image_id}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_from_cache(self, cache_key: str):
        """从缓存获取"""
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def save_to_cache(self, cache_key: str, data):
        """保存到缓存"""
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
    
    def generate_description(self, text: str, image_path: str) -> str:
        """带缓存的描述生成"""
        image_id = os.path.basename(image_path).split('.')[0]
        cache_key = self.get_cache_key(text, image_id)
        
        # 尝试从缓存获取
        cached = self.get_from_cache(cache_key)
        if cached:
            return cached
        
        # 生成并缓存
        description = super().generate_description_cot_step1(text, image_path)
        self.save_to_cache(cache_key, description)
        
        return description
```

### 2. 批量API调用

```python
def batch_generate_descriptions(
    self, 
    samples: List[Dict[str, Any]], 
    batch_size: int = 5
) -> List[str]:
    """批量生成描述"""
    descriptions = []
    
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        
        # 构建批量prompt
        batch_prompt = []
        for sample in batch:
            prompt = f"Describe image related to: {sample['text']}"
            batch_prompt.append(prompt)
        
        # 批量调用API（如果支持）
        batch_descriptions = self.batch_api_call(batch_prompt)
        descriptions.extend(batch_descriptions)
    
    return descriptions
```

### 3. 进度监控

```python
from tqdm import tqdm
import time

class ProgressCoIGenerator(CoIDataGenerator):
    def process_dataset_with_progress(self, *args, **kwargs):
        """带进度条的处理"""
        input_path = kwargs.get('input_path', args[0])
        
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = []
        
        # 使用tqdm显示进度
        with tqdm(total=len(data), desc="Processing samples") as pbar:
            for i, sample in enumerate(data):
                result = self.process_sample(sample)
                if result:
                    results.append(result)
                
                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({'completed': len(results)})
                
                # 显示速度
                if i % 10 == 0:
                    pbar.set_description(f"Processing sample {i}")
        
        return results
```

### 4. 错误恢复

```python
class RobustCoIGenerator(CoIDataGenerator):
    def __init__(self, *args, max_retries=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_retries = max_retries
    
    def process_sample_with_retry(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """带重试的样本处理"""
        for attempt in range(self.max_retries):
            try:
                return self.process_sample(sample)
            except Exception as e:
                logger.warning(f"尝试 {attempt + 1} 失败: {e}")
                
                if attempt < self.max_retries - 1:
                    # 指数退避
                    time.sleep(2 ** attempt)
                else:
                    # 最后一次尝试也失败，返回None
                    logger.error(f"样本处理失败: {sample.get('ImageID', 'unknown')}")
                    return None
    
    def process_dataset_robust(self, *args, **kwargs):
        """鲁棒的数据集处理"""
        # 实现略...
        pass
```

---

## 📊 监控和日志

### 1. 详细日志

```python
import logging
from datetime import datetime

# 配置详细日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'coi_stage_{datetime.now():%Y%m%d_%H%M%S}.log'),
        logging.StreamHandler()
    ]
)

# 记录统计信息
class StatsLogger:
    def __init__(self):
        self.stats = {
            'total_samples': 0,
            'successful_samples': 0,
            'failed_samples': 0,
            'api_calls': 0,
            'cache_hits': 0,
            'total_time': 0
        }
    
    def log_stats(self):
        logger.info("=" * 60)
        logger.info("处理统计:")
        logger.info(f"  总样本数: {self.stats['total_samples']}")
        logger.info(f"  成功: {self.stats['successful_samples']}")
        logger.info(f"  失败: {self.stats['failed_samples']}")
        logger.info(f"  API调用: {self.stats['api_calls']}")
        logger.info(f"  缓存命中: {self.stats['cache_hits']}")
        logger.info(f"  总时间: {self.stats['total_time']:.2f}秒")
        logger.info("=" * 60)
```

### 2. 性能监控

```python
import time
from contextlib import contextmanager

@contextmanager
def timer(name: str):
    """上下文管理器用于计时"""
    start = time.time()
    yield
    elapsed = time.time() - start
    logger.info(f"{name} 耗时: {elapsed:.2f}秒")

# 使用示例
with timer("生成描述"):
    description = generator.generate_description(text, image_path)
```

---

## 🎯 最佳实践

### 1. 数据安全

- 不要在代码中硬编码API密钥
- 使用环境变量或配置文件
- 定期轮换API密钥
- 遵守数据隐私法规

### 2. 成本控制

- 使用模拟模式进行测试
- 监控API使用情况
- 设置预算告警
- 使用缓存避免重复调用

### 3. 质量保证

- 验证输出格式
- 抽样检查生成质量
- 记录处理日志
- 保存中间结果

### 4. 可重现性

- 固定随机种子
- 记录所有参数
- 保存原始数据
- 版本控制代码

---

## 📚 更多信息

- [README.md](README.md): 项目介绍
- [测试文档](test_coi_stage.py): 测试用例
- [论文](https://github.com/ningpang/ArkMSA): 原始论文
- [MiniGPT4](https://github.com/Vision-CAIR/MiniGPT-4): MiniGPT4文档
- [OpenAI API](https://platform.openai.com/): OpenAI API文档

---

**下一步**: 查看 [README.md](README.md) 了解项目概览，或运行 `python quick_start.py` 开始实践。
