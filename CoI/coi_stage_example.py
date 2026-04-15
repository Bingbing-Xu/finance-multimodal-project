"""
ArkMSA CoI阶段完整示例代码
演示如何使用OpenAI GPT-4V生成图像描述和推理理由
"""

import json
import os
import base64
from typing import List, Dict, Any, Optional
import time
import logging
from openai import OpenAI
import argparse

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CoIDataGenerator:
    """CoI数据生成器"""
    
    def __init__(
        self, 
        api_key: str = None,
        image_dir: str = "./images",
        output_dir: str = "./output",
        delay: float = 1.0,
        max_tokens: int = 150
    ):
        """
        初始化数据生成器
        
        Args:
            api_key: OpenAI API密钥
            image_dir: 图像目录
            output_dir: 输出目录
            delay: API调用延迟（秒）
            max_tokens: 最大生成token数
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = None
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.delay = delay
        self.max_tokens = max_tokens
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 标签映射
        self.label_map = {
            "0": "Neutral",
            "1": "Positive", 
            "-1": "Negative"
        }
        
        logger.info(f"初始化CoI数据生成器")
        logger.info(f"图像目录: {image_dir}")
        logger.info(f"输出目录: {output_dir}")
    
    def encode_image(self, image_path: str) -> str:
        """将图像编码为base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def get_image_path(self, image_id: str) -> str:
        """获取图像完整路径"""
        extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
        
        for ext in extensions:
            image_path = os.path.join(self.image_dir, f"{image_id}{ext}")
            if os.path.exists(image_path):
                return image_path
        
        image_path = os.path.join(self.image_dir, image_id)
        if os.path.exists(image_path):
            return image_path
        
        return ""
    
    def _get_client(self):
        """获取OpenAI客户端（惰性初始化）"""
        if self.client is None:
            if not self.api_key:
                raise ValueError("未设置OpenAI API密钥")
            self.client = OpenAI(api_key=self.api_key)
        return self.client

    def generate_description_cot_step1(
        self, 
        text: str, 
        image_path: str,
        max_words: int = 20
    ) -> str:
        """
        CoT Step 1: 生成图像描述
        
        Args:
            text: 原始文本
            image_path: 图像路径
            max_words: 最大词数
            
        Returns:
            图像描述
        """
        try:
            base64_image = self.encode_image(image_path)
            
            # TWITTER-15/17的CoT Step 1 Prompt
            prompt = f"""Here are some content that people post on Twitter, and these content are composed of text and image. Please note that the text and image may or may not be relevant, so make your own judgment. By considering the text that <{text}> and analyzing the image, please give a description about the image within <{max_words}> words."""
            
            client = self._get_client()
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=0.7
            )
            
            description = response.choices[0].message.content.strip()
            logger.info(f"生成描述: {description[:50]}...")
            return description
            
        except Exception as e:
            logger.error(f"生成描述失败: {e}")
            return f"The image shows visual content related to the tweet text."
    
    def generate_rationale_cot_step2(
        self, 
        text: str, 
        description: str, 
        target: str, 
        label: str
    ) -> str:
        """
        CoT Step 2: 生成推理理由
        
        Args:
            text: 原始文本
            description: 图像描述
            target: 目标实体
            label: 情感标签
            
        Returns:
            推理理由
        """
        try:
            label_name = self.label_map.get(label, "Neutral")
            
            # TWITTER-15/17的CoT Step 2 Prompt
            prompt = f"""Based on the description: '{description}', and considering the text: '{text}', what is the sentiment polarity towards '{target}'? The correct sentiment is {label_name}. Summarize the sentiment polarity, and return only one of these words: [<Negative>, <Neutral>, <Positive>]. Make the answer format like: [Sentiment: Reason]"""
            
            client = self._get_client()
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=0.7
            )
            
            rationale = response.choices[0].message.content.strip()
            
            # 确保格式正确
            if not rationale.startswith("Sentiment:"):
                rationale = f"Sentiment: {label_name}. {rationale}"
            
            logger.info(f"生成理由: {rationale[:50]}...")
            return rationale
            
        except Exception as e:
            logger.error(f"生成理由失败: {e}")
            return f"Sentiment: {self.label_map.get(label, 'Neutral')}. Based on the text and image description, the sentiment towards {target} is {self.label_map.get(label, 'Neutral').lower()}."
    
    def process_sample(
        self, 
        sample: Dict[str, Any],
        use_openai: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        处理单个样本
        
        Args:
            sample: 原始样本
            use_openai: 是否使用OpenAI API
            
        Returns:
            处理后的样本或None
        """
        try:
            text = sample.get('text', '')
            target = sample.get('target', '')
            gt_label = str(sample.get('gt_label', '0'))
            image_id = sample.get('ImageID', '')
            
            # 获取图像路径
            image_path = self.get_image_path(image_id)
            
            if not image_path:
                logger.warning(f"图像未找到: {image_id}")
                # 使用默认描述和理由
                description = "The image is not available."
                reason = f"Sentiment: {self.label_map.get(gt_label, 'Neutral')}. Unable to analyze image content."
            else:
                if use_openai:
                    # 使用OpenAI API
                    description = self.generate_description_cot_step1(text, image_path)
                    time.sleep(self.delay)
                    
                    reason = self.generate_rationale_cot_step2(text, description, target, gt_label)
                    time.sleep(self.delay)
                else:
                    # 使用模拟数据
                    description = f"This image shows content related to the tweet about {target}."
                    reason = f"Sentiment: {self.label_map.get(gt_label, 'Neutral')}. The text and image suggest a {self.label_map.get(gt_label, 'Neutral').lower()} sentiment towards {target}."
            
            # 构建输出样本
            output_sample = {
                "text": text,
                "target": target,
                "gt_label": gt_label,
                "ImageID": image_id,
                "description": description,
                "reason": reason
            }
            
            logger.info(f"成功处理样本: {image_id}")
            return output_sample
            
        except Exception as e:
            logger.error(f"处理样本失败: {e}")
            return None
    
    def process_dataset(
        self, 
        input_path: str,
        output_filename: str = None,
        start_idx: int = 0,
        save_interval: int = 10,
        use_openai: bool = True
    ) -> List[Dict[str, Any]]:
        """
        处理整个数据集
        
        Args:
            input_path: 输入数据路径
            output_filename: 输出文件名
            start_idx: 开始索引
            save_interval: 保存间隔
            use_openai: 是否使用OpenAI API
            
        Returns:
            处理后的数据列表
        """
        # 加载数据
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"加载数据: {len(data)} 条")
        
        results = []
        total = len(data)
        
        # 从指定位置开始处理
        for i in range(start_idx, total):
            logger.info(f"处理进度: {i+1}/{total} ({(i+1)/total*100:.1f}%)")
            
            sample = data[i]
            result = self.process_sample(sample, use_openai=use_openai)
            
            if result:
                results.append(result)
            
            # 定期保存
            if (i + 1) % save_interval == 0 and results:
                checkpoint_path = os.path.join(
                    self.output_dir, 
                    f"checkpoint_{i+1}.json"
                )
                self.save_results(results, checkpoint_path)
                logger.info(f"保存检查点: {checkpoint_path}")
        
        # 保存最终结果
        if output_filename and results:
            final_path = os.path.join(self.output_dir, output_filename)
            self.save_results(results, final_path)
            logger.info(f"处理完成！结果保存到: {final_path}")
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        """保存结果"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"结果已保存: {output_path}")
    
    def validate_output(self, output_path: str) -> bool:
        """验证输出格式"""
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not data:
                logger.error("输出文件为空")
                return False
            
            # 检查必填字段
            required_fields = ['text', 'target', 'gt_label', 'ImageID', 'description', 'reason']
            sample = data[0]
            
            for field in required_fields:
                if field not in sample:
                    logger.error(f"缺少必填字段: {field}")
                    return False
            
            logger.info(f"验证通过: {len(data)} 条数据")
            return True
            
        except Exception as e:
            logger.error(f"验证失败: {e}")
            return False
    
    def convert_to_training_format(self, input_path: str, output_path: str):
        """转换为训练格式"""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        training_data = []
        for item in data:
            training_sample = {
                "text": item["text"],
                "target": item["target"],
                "gt_label": item["gt_label"],
                "ImageID": item["ImageID"],
                "description": item["description"],
                "reason": item["reason"]
            }
            training_data.append(training_sample)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"训练格式已保存: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='ArkMSA CoI Stage Implementation')
    parser.add_argument('--input', type=str, required=True, help='输入数据路径')
    parser.add_argument('--image_dir', type=str, required=True, help='图像目录路径')
    parser.add_argument('--output', type=str, default='./output', help='输出目录')
    parser.add_argument('--api_key', type=str, help='OpenAI API密钥')
    parser.add_argument('--delay', type=float, default=1.0, help='API调用延迟')
    parser.add_argument('--save_interval', type=int, default=10, help='保存间隔')
    parser.add_argument('--simulate', action='store_true', help='使用模拟模式（不调用API）')
    
    args = parser.parse_args()
    
    # 初始化生成器
    generator = CoIDataGenerator(
        api_key=args.api_key,
        image_dir=args.image_dir,
        output_dir=args.output,
        delay=args.delay
    )
    
    # 处理数据
    results = generator.process_dataset(
        input_path=args.input,
        output_filename="coi_results.json",
        use_openai=not args.simulate,
        save_interval=args.save_interval
    )
    
    # 验证输出
    output_path = os.path.join(args.output, "coi_results.json")
    if generator.validate_output(output_path):
        logger.info(f"处理完成！共生成 {len(results)} 条数据")
    else:
        logger.error("输出验证失败")


if __name__ == "__main__":
    # 示例用法
    
    # 方法1: 命令行参数
    # python coi_stage_example.py --input ./twitter15_raw.json --image_dir ./images --output ./output
    
    # 方法2: 直接调用
    generator = CoIDataGenerator(
        image_dir="./images",
        output_dir="./output",
        delay=1.0
    )
    
    # 使用模拟模式处理数据（不调用API）
    results = generator.process_dataset(
        input_path="./twitter15_raw.json",
        output_filename="tw15_coi_results.json",
        use_openai=False  # 设置为True以使用OpenAI API
    )
    
    print(f"处理完成: {len(results)} 条数据")
