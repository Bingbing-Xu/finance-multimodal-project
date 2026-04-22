"""
ArkMSA: CoI (Chain-of-Thought) Stage Implementation for Financial Sentiment Analysis
CoI阶段：金融股票舆情多模态分析（使用OpenAI兼容模式调用qwen3-vl-plus）
"""

import json
import os
import base64
from typing import List, Dict, Any, Optional
import time
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import logging

from openai import OpenAI

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CoISample:
    """CoI阶段生成的样本数据结构"""
    text: str                    # 原始文本（推文内容）
    target: str                  # 目标实体（股票代码/公司名称）
    gt_label: str               # 真实标签 (-1: Negative/利空, 0: Neutral/中性, 1: Positive/利好)
    ImageID: str               # 图像ID
    description: str           # 图像描述（由MLLM生成）
    reason: str               # 推理理由（由MLLM生成）

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CoISample':
        """从字典创建实例"""
        return cls(**data)


class MLLMInterface(ABC):
    """多模态大语言模型接口抽象类"""

    @abstractmethod
    def generate_description(self, text: str, image_path: str) -> str:
        """生成图像描述"""
        pass

    @abstractmethod
    def generate_rationale(self, text: str, description: str, target: str, label: str) -> str:
        """生成推理理由"""
        pass


class QwenVLAdapter(MLLMInterface):
    """阿里云千问多模态模型适配器（OpenAI兼容模式）- 金融股票特化版"""

    def __init__(
            self,
            api_key: str = None,
            model_name: str = "qwen3-vl-plus",
            max_retries: int = 3,
            retry_delay: float = 1.0,
            base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.api_call_count = 0

        if not self.api_key:
            raise ValueError("未提供API密钥。请设置DASHSCOPE_API_KEY环境变量")

        self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        logger.info(f"初始化金融分析适配器: {model_name}")

    def _encode_image_to_base64(self, image_path: str) -> str:
        """将图像编码为base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _call_api(self, messages: List[Dict]) -> str:
        """调用API"""
        for attempt in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=512,
                    temperature=0.1,
                    top_p=0.9
                )
                self.api_call_count += 1
                return completion.choices[0].message.content
            except Exception as e:
                logger.error(f"API调用异常 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                continue
        return ""

    def generate_description(self, text: str, image_path: str) -> str:
        """生成图像描述（CoT Step 1）- 最终优化版"""
        base64_image = self._encode_image_to_base64(image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    },
                    {
                        "type": "text",
                        "text": f"""You are a financial image analyst. Analyze the provided image, which is associated with the social media text: <{text}>.
    Your task is to generate a concise, evidence-grounded visual description for downstream sentiment analysis.

    Follow these steps:
    1. **Identify Image Type**: Determine whether the image is a candlestick chart, line chart, bar/volume chart, technical indicator panel, trading screenshot, portfolio/PnL screenshot, news screenshot, earnings table, multi-panel financial graphic, or financial meme.
    2. **Extract Key Visual Evidence**: Describe only what is clearly visible:
       - Price action: recent trend (up/down/sideways), breakout/breakdown, and apparent support/resistance.
       - Technical signals: clearly labeled moving average crossovers, volume spikes, RSI/MACD extremes or crossovers.
       - Legible text: ticker symbols, prices, percentage changes, headlines, dates, and visible gains/losses.
    3. **Synthesize**: Summarize the most decision-relevant visual financial signal(s) in the image.

    Requirements:
    - Focus only on the image itself. The accompanying social media text is context only; do not use it as evidence unless the same information is explicitly visible in the image.
    - Do not infer sentiment labels.
    - Do not invent unseen data, values, indicators, or patterns.
    - Use cautious language for unclear details, such as "appears" or "not clearly legible."
    - Prefer the single most salient financial signal over exhaustive detail.
    - Output only the description, with no extra text.
    - Length: 20-40 words, preferably 1 sentence; use 2 short sentences only if the image contains multiple distinct signals."""
                    }
                ]
            }
        ]

        description = self._call_api(messages)
        logger.debug(f"生成描述: {description[:100]}...")
        return description

    def generate_rationale(self, text: str, description: str, target: str, label: str) -> str:
        """生成推理理由（CoT Step 2）- 金融股票特化"""
        label_map = {"-1": "Bearish/Negative", "0": "Neutral", "1": "Bullish/Positive"}
        label_name = label_map.get(label, "Neutral")

        prompt = f"""You are an expert financial sentiment analyst. Analyze the multimodal sentiment towards **{target}** based on the following inputs.

**Input Data**
- Target Stock: {target}
- Text: {text}
- Image Description: {description}
- Correct Sentiment Label: {label_name}

**Part 1: Financial Sentiment Definitions for Social Media Posts**
- **Bullish/Positive**: A post should be labeled as Bullish/Positive if it expresses or implies that the target stock is likely to rise in the short term, or if it conveys information that would typically support a favorable market reaction. Signals may include optimistic interpretations of earnings or company developments, positive catalysts, buy or hold recommendations, favorable analyst views, or technical indicators suggesting upward momentum (e.g., breakout above resistance, support holding, strong accumulation). In social media contexts, informal expressions, slang, emojis, hashtags, or exaggerated language should be interpreted based on their overall trading implication rather than their surface tone alone.

- **Bearish/Negative**: A post should be labeled as Bearish/Negative if it expresses or implies that the target stock is likely to fall in the short term, or if it conveys information that would typically trigger an unfavorable market reaction. Signals may include pessimistic interpretations of earnings or guidance, negative company news, sell or short recommendations, dilution concerns, regulatory or operational risks, or technical indicators suggesting downward pressure (e.g., breakdown below support, lower lows, heavy selling). In social media contexts, negative sentiment should be labeled as Bearish only when it implies downside risk for the target stock.

- **Neutral**: A post should be labeled as Neutral if it does not convey a clear directional stance toward the target stock, if its implied market effect is weak or ambiguous, or if bullish and bearish signals are balanced. This includes factual statements without clear evaluation, reposted news without added opinion, questions, watchlist-style comments, or posts with conflicting signals that do not support a reliable bullish or bearish interpretation.

- **Annotation Principle**: Labels should be assigned according to the **net implied short-term trading stance** toward the target stock, rather than the writer's general emotion or the stock's actual future movement. Focus on context, discourse intent, and trading implication. If the target stock is unclear, the post discusses only the broader market without a clear stance on the target stock, or no reliable directional implication can be inferred, the label should be Neutral.

**Part 2: Chain-of-Thought Reasoning**
Follow this structure strictly to provide a detailed explanation of why the sentiment towards {target} is {label_name}:

1.  **Step 1: Text Analysis (Unimodal)**
    - Identify key phrases, financial terms, sentiment cues, emojis, slang, or hashtags in the text: "{text}".
    - Based *only* on the text, what is the implied short-term trading stance towards {target}? State the conclusion and the supporting evidence.

2.  **Step 2: Image Analysis (Unimodal)**
    - Analyze the visual signals based *only* on the image description: "{description}".
    - What specific financial elements (e.g., candlestick patterns, volume, trend lines, technical indicators, chart annotations) are present?
    - Based *only* on the image, what is the implied short-term trading stance towards {target}? State the conclusion and the supporting evidence.

3.  **Step 3: Cross-Modal Consistency & Conflict Resolution**
    - Compare the conclusions from Step 1 (text) and Step 2 (image).
    - Determine if the text and image are **Aligned** (both imply the same directional stance), **Complementary** (provide different but reinforcing evidence for the same stance), or **Conflicting** (e.g., bullish text with a bearish chart pattern).
    - **If Conflicting**: Identify the most plausible interpretation (e.g., sarcasm, outdated chart, text referring to a different aspect, or the image being a meme that contradicts the text). Explain why, after conflict resolution, the net implied trading stance leans towards {label_name}.

4.  **Step 4: Final Integrated Conclusion**
    - Synthesize the findings from the previous steps into a final, coherent reasoning.
    - Clearly state the final sentiment: **{label_name}**.
    - Explain how the integrated evidence from both modalities (or the resolution of a conflict) supports this final classification, keeping the annotation principle (net implied short-term trading stance) in mind.

**Output Format**:
Text Sentiment: [Implied stance and evidence]
Image Sentiment: [Implied stance and evidence]
Consistency Analysis: [Aligned/Complementary/Conflicting] - [Explanation]
Final Sentiment: {label_name}. [Final integrated reasoning]"""

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        return self._call_api(messages)


class CoIProcessor:
    """CoI阶段处理器 - 支持原生(-1,0,1)标签格式"""

    def __init__(
        self,
        mllm_adapter: MLLMInterface,
        image_dir: str,
        output_dir: str = "./output",
        delay: float = 1.0
    ):
        self.mllm_adapter = mllm_adapter
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.delay = delay
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"初始化CoI处理器，图像目录: {image_dir}")

    def load_original_data(self, data_path: str, format_type: str = "tsv", skip_header: bool = True) -> List[Dict[str, Any]]:
        """加载原始数据"""
        try:
            if format_type == "tsv":
                return self._load_tsv_data(data_path, skip_header)
            elif format_type == "json":
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"成功加载 {len(data)} 条JSON数据")
                return data
            else:
                raise ValueError(f"不支持的格式类型: {format_type}")
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            return []

    def _load_tsv_data(self, tsv_path: str, skip_header: bool = True) -> List[Dict[str, Any]]:
        """
        加载TSV格式数据
        支持两种标签格式：
        1. 原生(-1, 0, 1) - 直接使用
        2. Twitter格式(0, 1, 2) - 转换为(-1, 0, 1)
        """
        data = []

        with open(tsv_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split('\t')
                if len(parts) < 5:
                    continue

                # 跳过表头
                if line_num == 1 and skip_header:
                    try:
                        int(parts[1])  # 尝试解析第二列
                    except ValueError:
                        logger.info(f"跳过表头行: {line}")
                        continue

                try:
                    index = parts[0].strip()
                    label_str = parts[1].strip()
                    image_id = parts[2].strip()
                    text = parts[3].strip()
                    target = parts[4].strip()

                    # 清理图像ID
                    image_id_base = os.path.splitext(image_id)[0] if '.' in image_id else image_id

                    # 标签处理：支持(-1,0,1)原生格式和(0,1,2)Twitter格式
                    label_lower = label_str.lower()
                    if label_lower in ["negative", "bearish"]:
                        converted_label = "-1"
                    elif label_lower in ["neutral", "hold"]:
                        converted_label = "0"
                    elif label_lower in ["positive", "bullish"]:
                        converted_label = "1"
                    else:
                        try:
                            numeric_label = int(label_str)
                            # 自动检测格式：如果是-1,0,1则直接使用；如果是0,1,2则转换
                            if numeric_label in [-1, 0, 1]:
                                converted_label = str(numeric_label)
                            elif numeric_label == 0:  # Twitter格式: 0->-1
                                converted_label = "-1"
                            elif numeric_label == 1:  # Twitter格式: 1->0
                                converted_label = "0"
                            elif numeric_label == 2:  # Twitter格式: 2->1
                                converted_label = "1"
                            else:
                                converted_label = "0"  # 默认中性
                        except ValueError:
                            converted_label = "0"

                    data.append({
                        "text": text,
                        "target": target,
                        "gt_label": converted_label,
                        "ImageID": image_id_base,
                        "index": index
                    })

                except Exception as e:
                    logger.error(f"解析第{line_num}行失败: {e}")
                    continue

        logger.info(f"成功加载 {len(data)} 条数据，标签格式: (-1,0,1)")
        return data

    def get_image_path(self, image_id: str) -> str:
        """获取图像完整路径"""
        extensions = ['', '.jpg', '.jpeg', '.png', '.gif', '.bmp']
        for ext in extensions:
            image_path = os.path.join(self.image_dir, f"{image_id}{ext}" if ext else image_id)
            if os.path.exists(image_path):
                return image_path
        logger.warning(f"图像未找到: {image_id}")
        return ""

    def process_single_sample(self, sample: Dict[str, Any]) -> Optional[CoISample]:
        """处理单个样本"""
        try:
            text = sample.get('text', '')
            target = sample.get('target', '')
            gt_label = str(sample.get('gt_label', '0'))
            image_id = sample.get('ImageID', '')

            image_path = self.get_image_path(image_id)
            if not image_path:
                return None

            logger.info(f"处理样本 {image_id}: {target}")

            # Step 1: 生成图像描述
            description = self.mllm_adapter.generate_description(text, image_path)
            time.sleep(self.delay)

            # Step 2: 生成推理理由
            reason = self.mllm_adapter.generate_rationale(text, description, target, gt_label)
            time.sleep(self.delay)

            return CoISample(
                text=text, target=target, gt_label=gt_label,
                ImageID=image_id, description=description, reason=reason
            )

        except Exception as e:
            logger.error(f"处理样本失败: {e}")
            return None

    def process_batch(self, data: List[Dict[str, Any]], save_interval: int = 100) -> List[CoISample]:
        """批量处理数据"""
        results = []
        total = len(data)

        for i, sample in enumerate(data):
            logger.info(f"进度: {i+1}/{total} ({(i+1)/total*100:.1f}%)")

            coi_sample = self.process_single_sample(sample)
            if coi_sample:
                results.append(coi_sample)

            if (i + 1) % save_interval == 0:
                self.save_results(results, f"checkpoint_{i+1}.json")

        return results

    def save_results(self, results: List[CoISample], filename: str):
        """保存结果"""
        output_path = os.path.join(self.output_dir, filename)
        try:
            data = [sample.to_dict() for sample in results]
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"已保存 {len(results)} 条数据到 {filename}")
        except Exception as e:
            logger.error(f"保存失败: {e}")


def main():
    """主函数"""
    API_KEY = "sk-dd8f7e873129418b9524512102865ec0"

    #config = {
     #   "image_dir": "/home/remance/文档/xbb/FinancialDataset/images",
      #  "input_data": "/home/remance/文档/xbb/FinancialDataset/tsv/train.tsv",
       # "output_dir": "./coi_output_finance_plus",
     #   "model_name": "qwen3-vl-plus",
     #   "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
     #   "delay": 2.0,  # API调用间隔（秒）
     #   "save_interval": 1000
    #}

    config = {
        "image_dir": "/home/remance/文档/xbb/FinancialDataset/images",
        "input_data": "/home/remance/文档/xbb/FinancialDataset/tsv/train_safe.tsv",
        "output_dir": "./coi_output_finance_max",
        "model_name": "qwen-vl-max",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "delay": 2.0,  # API调用间隔（秒）
        "save_interval": 1000
    }


    # 初始化适配器
    adapter = QwenVLAdapter(
        api_key=API_KEY,
        model_name=config["model_name"],
        base_url=config["base_url"]
    )

    # 初始化处理器
    processor = CoIProcessor(
        mllm_adapter=adapter,
        image_dir=config["image_dir"],
        output_dir=config["output_dir"],
        delay=config["delay"]
    )

    # 加载数据（自动支持-1/0/1格式）
    raw_data = processor.load_original_data(
        config["input_data"],
        format_type="tsv",
        skip_header=True
    )

    if not raw_data:
        logger.error("未加载到数据")
        return

    logger.info(f"开始处理 {len(raw_data)} 条金融舆情数据...")
    results = processor.process_batch(raw_data, save_interval=config["save_interval"])

    if results:
        processor.save_results(results, "train.json")
        logger.info(f"完成！共处理 {len(results)} 条，API调用次数: {adapter.api_call_count}")


if __name__ == "__main__":
    main()