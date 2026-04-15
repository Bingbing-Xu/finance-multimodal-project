"""
ArkMSA CoI阶段快速开始脚本
5分钟快速上手教程
"""

import json
import os
from typing import List, Dict, Any
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QuickCoIGenerator:
    """快速CoI数据生成器"""
    
    def __init__(
        self,
        image_dir: str,
        output_dir: str = "./coi_output",
        use_openai: bool = False
    ):
        """
        初始化生成器
        
        Args:
            image_dir: 图像目录路径
            output_dir: 输出目录路径
            use_openai: 是否使用OpenAI API（需要API密钥）
        """
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.use_openai = use_openai
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 标签映射
        self.label_map = {
            "0": "Neutral",
            "1": "Positive",
            "-1": "Negative"
        }
        
        # 模拟描述模板
        self.description_templates = [
            "The image shows visual content related to the tweet about {target}.",
            "This image depicts a scene that may be relevant to the mentioned {target}.",
            "The picture displays content connected to the text discussing {target}.",
            "An image showing elements related to {target} in the tweet.",
            "Visual representation of content mentioned in the tweet about {target}."
        ]
        
        # 模拟理由模板
        self.rationale_templates = [
            "Sentiment: {label}. The text and image suggest a {label_lower} sentiment towards {target}.",
            "Sentiment: {label}. Based on the content analysis, the sentiment is {label_lower}.",
            "Sentiment: {label}. The tweet expresses a {label_lower} opinion about {target}.",
            "Sentiment: {label}. Considering both text and image, the sentiment is {label_lower}.",
            "Sentiment: {label}. The overall sentiment towards {target} is {label_lower}."
        ]
        
        logger.info(f"✓ 快速生成器已初始化")
        logger.info(f"  图像目录: {image_dir}")
        logger.info(f"  输出目录: {output_dir}")
        logger.info(f"  模式: {'OpenAI API' if use_openai else '模拟模式'}")
    
    def get_image_path(self, image_id: str) -> str:
        """获取图像路径"""
        extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
        
        for ext in extensions:
            path = os.path.join(self.image_dir, f"{image_id}{ext}")
            if os.path.exists(path):
                return path
        
        path = os.path.join(self.image_dir, image_id)
        if os.path.exists(path):
            return path
        
        return ""
    
    def generate_description(self, text: str, target: str, image_id: str) -> str:
        """生成图像描述"""
        import random
        
        if self.use_openai:
            # TODO: 集成OpenAI API
            # 这里应该调用GPT-4V API
            return f"Description of image {image_id} related to {target}."
        else:
            # 使用模拟数据
            template = random.choice(self.description_templates)
            return template.format(target=target)
    
    def generate_rationale(
        self, 
        text: str, 
        description: str, 
        target: str, 
        label: str
    ) -> str:
        """生成推理理由"""
        import random
        
        label_name = self.label_map.get(label, "Neutral")
        
        if self.use_openai:
            # TODO: 集成OpenAI API
            return f"Sentiment: {label_name}. Reason for sentiment."
        else:
            # 使用模拟数据
            template = random.choice(self.rationale_templates)
            return template.format(
                label=label_name,
                label_lower=label_name.lower(),
                target=target
            )
    
    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """处理单个样本"""
        text = sample.get('text', '')
        target = sample.get('target', '')
        gt_label = str(sample.get('gt_label', '0'))
        image_id = sample.get('ImageID', '')
        
        # 生成描述和理由
        description = self.generate_description(text, target, image_id)
        rationale = self.generate_rationale(text, description, target, gt_label)
        
        return {
            "text": text,
            "target": target,
            "gt_label": gt_label,
            "ImageID": image_id,
            "description": description,
            "reason": rationale
        }
    
    def process_file(
        self, 
        input_path: str,
        output_filename: str = "coi_results.json"
    ) -> List[Dict[str, Any]]:
        """处理整个文件"""
        logger.info(f"正在读取数据: {input_path}")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"✓ 读取 {len(data)} 条数据")
        
        # 处理数据
        results = []
        for i, sample in enumerate(data):
            logger.info(f"处理进度: {i+1}/{len(data)} ({(i+1)/len(data)*100:.1f}%)")
            
            result = self.process_sample(sample)
            results.append(result)
        
        # 保存结果
        output_path = os.path.join(self.output_dir, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ 结果已保存: {output_path}")
        logger.info(f"✓ 处理完成: {len(results)} 条数据")
        
        return results
    
    def validate_result(self, result_path: str) -> bool:
        """验证结果格式"""
        try:
            with open(result_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not data:
                logger.error("结果文件为空")
                return False
            
            # 检查必填字段
            required_fields = ['text', 'target', 'gt_label', 'ImageID', 'description', 'reason']
            sample = data[0]
            
            for field in required_fields:
                if field not in sample:
                    logger.error(f"缺少字段: {field}")
                    return False
            
            logger.info(f"✓ 验证通过: {len(data)} 条数据")
            return True
            
        except Exception as e:
            logger.error(f"验证失败: {e}")
            return False


def example_1_minute_tutorial():
    """1分钟快速教程"""
    print("\n" + "="*60)
    print("1分钟快速上手ArkMSA CoI阶段")
    print("="*60)
    
    # 创建示例输入数据
    sample_input = [
        {
            "text": "I love the new $T$! It's amazing! #iPhone",
            "target": "iPhone",
            "gt_label": "1",
            "ImageID": "sample_001"
        },
        {
            "text": "The weather is terrible today.",
            "target": "weather",
            "gt_label": "-1",
            "ImageID": "sample_002"
        },
        {
            "text": "Just had lunch.",
            "target": "lunch",
            "gt_label": "0",
            "ImageID": "sample_003"
        }
    ]
    
    # 保存示例输入
    os.makedirs("./example_data", exist_ok=True)
    with open("./example_data/sample_input.json", 'w', encoding='utf-8') as f:
        json.dump(sample_input, f, indent=2, ensure_ascii=False)
    
    print("\n步骤1: 创建输入数据")
    print("  输入文件: ./example_data/sample_input.json")
    print("  数据格式: Twitter15/17原始格式")
    
    # 初始化生成器
    generator = QuickCoIGenerator(
        image_dir="./example_images",  # 图像目录
        output_dir="./example_output", # 输出目录
        use_openai=False               # 使用模拟模式
    )
    
    print("\n步骤2: 初始化生成器")
    print("  使用模拟模式（无需API密钥）")
    
    # 处理数据
    results = generator.process_file(
        input_path="./example_data/sample_input.json",
        output_filename="coi_results.json"
    )
    
    print("\n步骤3: 处理数据")
    print("  生成图像描述（CoT Step 1）")
    print("  生成推理理由（CoT Step 2）")
    
    # 验证结果
    is_valid = generator.validate_result("./example_output/coi_results.json")
    
    print("\n步骤4: 验证结果")
    print(f"  格式验证: {'通过' if is_valid else '失败'}")
    
    # 显示示例结果
    print("\n步骤5: 查看结果")
    print("  输出文件: ./example_output/coi_results.json")
    print("\n示例输出:")
    
    sample_result = results[0] if results else None
    if sample_result:
        print(f"  Text: {sample_result['text']}")
        print(f"  Target: {sample_result['target']}")
        print(f"  Label: {sample_result['gt_label']}")
        print(f"  Description: {sample_result['description']}")
        print(f"  Reason: {sample_result['reason']}")
    
    print("\n" + "="*60)
    print("完成！现在您可以使用真实数据和API了")
    print("="*60)
    
    return results


def example_real_usage():
    """真实使用示例"""
    print("\n" + "="*60)
    print("真实使用示例")
    print("="*60)
    
    # 示例：使用OpenAI API
    print("\n示例1: 使用OpenAI GPT-4V API")
    print("-" * 40)
    print("""
from coi_stage_example import CoIDataGenerator

# 设置API密钥
os.environ["OPENAI_API_KEY"] = "your-api-key"

# 创建生成器
generator = CoIDataGenerator(
    image_dir="./twitter15_images",
    output_dir="./output",
    delay=1.0  # 避免API限制
)

# 处理数据
results = generator.process_dataset(
    input_path="./twitter15_raw.json",
    output_filename="tw15_coi_results.json",
    use_openai=True,
    save_interval=10
)
    """)
    
    # 示例：使用MiniGPT4
    print("\n示例2: 使用MiniGPT4")
    print("-" * 40)
    print("""
from coi_stage_framework import MiniGPT4Adapter, CoIProcessor

# 初始化MiniGPT4
mllm_adapter = MiniGPT4Adapter(
    model_name="mini-gpt4",
    max_tokens=512
)

# 创建处理器
processor = CoIProcessor(
    mllm_adapter=mllm_adapter,
    image_dir="./twitter15_images",
    output_dir="./output"
)

# 处理数据
raw_data = processor.load_original_data("./twitter15_raw.json")
results = processor.process_batch(raw_data)
processor.save_results(results, "tw15_coi_results.json")
    """)
    
    # 示例：批量处理
    print("\n示例3: 批量处理多个文件")
    print("-" * 40)
    print("""
import os
from coi_stage_example import CoIDataGenerator

generator = CoIDataGenerator(
    image_dir="./images",
    output_dir="./output",
    delay=1.0
)

# 处理多个数据集
datasets = [
    ("./twitter15_raw.json", "tw15_coi_results.json"),
    ("./twitter17_raw.json", "tw17_coi_results.json"),
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
    """)


def example_customization():
    """自定义扩展示例"""
    print("\n" + "="*60)
    print("自定义扩展示例")
    print("="*60)
    
    print("""
1. 添加新的MLLM适配器

from coi_stage_framework import MLLMInterface

class MyCustomMLLM(MLLMInterface):
    def generate_description(self, text: str, image_path: str) -> str:
        # 调用您的多模态模型API
        return "Generated description"
    
    def generate_rationale(self, text: str, description: str, target: str, label: str) -> str:
        # 调用您的模型生成理由
        return "Generated rationale"

# 使用自定义适配器
mllm_adapter = MyCustomMLLM()
processor = CoIProcessor(mllm_adapter=mllm_adapter, ...)
""")
    
    print("""
2. 修改Prompt模板

# 在代码中找到prompt变量并修改
cot_step1_prompt = "您的自定义prompt..."
cot_step2_prompt = "您的自定义prompt..."
""")
    
    print("""
3. 添加后处理步骤

class MyCoIGenerator(CoIDataGenerator):
    def post_process(self, result: Dict[str, Any]) -> Dict[str, Any]:
        # 添加自定义后处理
        result['custom_field'] = 'custom_value'
        return result
""")


def main():
    """主函数"""
    print("\n" + "🚀" * 30)
    print("ArkMSA CoI阶段快速开始")
    print("🚀" * 30)
    
    # 显示菜单
    print("\n请选择:")
    print("1. 1分钟快速教程")
    print("2. 真实使用示例")
    print("3. 自定义扩展示例")
    print("4. 运行测试")
    
    choice = input("\n输入选项 (1-4): ").strip()
    
    if choice == "1":
        example_1_minute_tutorial()
    elif choice == "2":
        example_real_usage()
    elif choice == "3":
        example_customization()
    elif choice == "4":
        import subprocess
        subprocess.run(["python", "test_coi_stage.py"])
    else:
        print("\n无效选项，运行1分钟教程...")
        example_1_minute_tutorial()
    
    print("\n" + "="*60)
    print("感谢使用！更多信息请查看 README.md")
    print("="*60)


if __name__ == "__main__":
    # 如果直接运行，执行快速教程
    if len(os.sys.argv) == 1:
        example_1_minute_tutorial()
    else:
        main()
