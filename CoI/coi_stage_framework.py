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
        """生成图像描述（CoT Step 1）- 金融股票特化"""
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
                        "text": f"""Here are financial social media posts about stock markets, composed of text and images. The text mentions: <{text}>. Note that text and image may be misaligned (e.g., bullish text with bearish chart, or sarcastic "to the moon" with crashing price), so analyze carefully. Focus on: stock tickers, candlestick patterns (green/red), price trends, trading volumes, technical indicators (MA/MACD/RSI), or financial memes (diamond hands, rocket emojis). Provide a concise description within <20> words emphasizing financial signals."""
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

        prompt = f"""Based on the text and image description provided, analyze the stock market sentiment towards <{target}>.

Text: {text}
Image Description: {description}
Correct Sentiment: {label_name}

Provide a detailed explanation of why the sentiment towards {target} is {label_name}. Consider:
1. Financial terminology (bullish/bearish, support/resistance, FOMO, panic selling, diamond hands, YOLO, short squeeze)
2. Multimodal alignment: Does the chart contradict the text? (e.g., "going to moon" text with red candlestick crash, or "buy the dip" with breakdown pattern)
3. Visual trading signals: Candlestick patterns (hammer, engulfing, doji), volume spikes, breakout/breakdown levels, green/red color psychology, all-time-high/low markers

Your response should include:
1. The sentiment classification
2. A detailed reasoning explaining how text and/or image support this classification, with attention to financial-specific signals (ticker mentions, price action, technical indicators, market psychology)
3. Mention of specific financial elements (chart patterns, trading keywords, numerical price data, WSB terminology) that influenced the decision

Format: Sentiment: {label_name}. [Your detailed reasoning here]"""

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

    config = {
        "image_dir": "D:/PythonProjects/finance_research/FinancialDataset/images",
        "input_data": "D:/PythonProjects/finance_research/FinancialDataset/tsv/train.tsv",
        "output_dir": "./coi_output_finance",
        "model_name": "qwen3-vl-plus",
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
        processor.save_results(results, "coi_final_results.json")
        logger.info(f"完成！共处理 {len(results)} 条，API调用次数: {adapter.api_call_count}")


if __name__ == "__main__":
    main()