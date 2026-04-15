"""
ArkMSA CoI阶段测试脚本
用于验证代码正确性和数据格式
"""

import json
import os
import tempfile
import shutil
from typing import List, Dict, Any
import unittest
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入要测试的模块
from coi_stage_framework import CoISample, MiniGPT4Adapter, CoIProcessor
from coi_stage_example import CoIDataGenerator


class TestCoIStage(unittest.TestCase):
    """测试CoI阶段"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.test_dir = tempfile.mkdtemp()
        self.image_dir = os.path.join(self.test_dir, "images")
        self.output_dir = os.path.join(self.test_dir, "output")
        
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 创建测试图像
        self.create_test_image("test_image1.jpg")
        self.create_test_image("test_image2.png")
        
        # 创建测试数据
        self.test_data = [
            {
                "text": "I love the new iPhone! $T$ is amazing.",
                "target": "iPhone",
                "gt_label": "1",
                "ImageID": "test_image1"
            },
            {
                "text": "The weather is terrible today.",
                "target": "weather",
                "gt_label": "-1",
                "ImageID": "test_image2"
            },
            {
                "text": "Just had lunch.",
                "target": "lunch",
                "gt_label": "0",
                "ImageID": "non_existent"
            }
        ]
        
        # 保存测试数据
        self.test_data_path = os.path.join(self.test_dir, "test_data.json")
        with open(self.test_data_path, 'w', encoding='utf-8') as f:
            json.dump(self.test_data, f, indent=2, ensure_ascii=False)
    
    def tearDown(self):
        """测试后的清理工作"""
        shutil.rmtree(self.test_dir)
    
    def create_test_image(self, filename: str):
        """创建测试图像"""
        # 创建一个简单的测试图像
        try:
            from PIL import Image
            import numpy as np
            
            # 创建随机图像
            img_array = np.random.rand(100, 100, 3) * 255
            img_array = img_array.astype(np.uint8)
            
            img = Image.fromarray(img_array)
            img.save(os.path.join(self.image_dir, filename))
            
        except ImportError:
            # 如果PIL不可用，创建空文件
            with open(os.path.join(self.image_dir, filename), 'wb') as f:
                f.write(b'dummy image data')
    
    def test_coisample_dataclass(self):
        """测试CoISample数据类"""
        logger.info("测试CoISample数据类...")
        
        sample = CoISample(
            text="Test text",
            target="Test target",
            gt_label="1",
            ImageID="123456",
            description="Test description",
            reason="Test reason"
        )
        
        # 测试to_dict
        sample_dict = sample.to_dict()
        self.assertIsInstance(sample_dict, dict)
        self.assertEqual(sample_dict["text"], "Test text")
        self.assertEqual(sample_dict["target"], "Test target")
        self.assertEqual(sample_dict["gt_label"], "1")
        self.assertEqual(sample_dict["ImageID"], "123456")
        self.assertEqual(sample_dict["description"], "Test description")
        self.assertEqual(sample_dict["reason"], "Test reason")
        
        # 测试from_dict
        new_sample = CoISample.from_dict(sample_dict)
        self.assertEqual(new_sample.text, sample.text)
        self.assertEqual(new_sample.target, sample.target)
        self.assertEqual(new_sample.gt_label, sample.gt_label)
        
        logger.info("✓ CoISample测试通过")
    
    def test_minigpt4_adapter(self):
        """测试MiniGPT4适配器"""
        logger.info("测试MiniGPT4适配器...")
        
        adapter = MiniGPT4Adapter()
        
        # 测试描述生成
        description = adapter.generate_description(
            text="Test text",
            image_path=os.path.join(self.image_dir, "test_image1.jpg")
        )
        self.assertIsInstance(description, str)
        self.assertGreater(len(description), 0)
        
        # 测试理由生成
        rationale = adapter.generate_rationale(
            text="Test text",
            description="Test description",
            target="Test target",
            label="1"
        )
        self.assertIsInstance(rationale, str)
        self.assertGreater(len(rationale), 0)
        
        logger.info("✓ MiniGPT4适配器测试通过")
    
    def test_data_generator_simulation(self):
        """测试数据生成器（模拟模式）"""
        logger.info("测试数据生成器（模拟模式）...")
        
        from quick_start import QuickCoIGenerator
        generator = QuickCoIGenerator(
            image_dir=self.image_dir,
            output_dir=self.output_dir,
            use_openai=False
        )
        
        # 处理单个样本
        sample = self.test_data[0]
        result = generator.process_sample(sample)
        
        self.assertIsNotNone(result)
        self.assertEqual(result["text"], sample["text"])
        self.assertEqual(result["target"], sample["target"])
        self.assertEqual(result["gt_label"], sample["gt_label"])
        self.assertEqual(result["ImageID"], sample["ImageID"])
        self.assertIsInstance(result["description"], str)
        self.assertIsInstance(result["reason"], str)
        self.assertGreater(len(result["description"]), 0)
        self.assertGreater(len(result["reason"]), 0)
        
        logger.info("✓ 数据生成器测试通过")
    
    def test_dataset_processing(self):
        """测试数据集处理"""
        logger.info("测试数据集处理...")
        
        from quick_start import QuickCoIGenerator
        generator = QuickCoIGenerator(
            image_dir=self.image_dir,
            output_dir=self.output_dir,
            use_openai=False
        )
        
        # 处理数据集
        results = generator.process_file(
            input_path=self.test_data_path,
            output_filename="test_results.json"
        )
        
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        # 验证结果
        for result in results:
            self._validate_sample_format(result)
        
        # 验证输出文件
        output_path = os.path.join(self.output_dir, "test_results.json")
        self.assertTrue(os.path.exists(output_path))
        
        is_valid = generator.validate_result(output_path)
        self.assertTrue(is_valid)
        
        logger.info("✓ 数据集处理测试通过")
    
    def test_image_path_resolution(self):
        """测试图像路径解析"""
        logger.info("测试图像路径解析...")
        
        from quick_start import QuickCoIGenerator
        generator = QuickCoIGenerator(
            image_dir=self.image_dir,
            output_dir=self.output_dir,
            use_openai=False
        )
        
        # 测试不同扩展名
        path1 = generator.get_image_path("test_image1")
        path2 = generator.get_image_path("test_image2")
        
        self.assertTrue(os.path.exists(path1))
        self.assertTrue(os.path.exists(path2))
        
        # 测试不存在的图像
        path3 = generator.get_image_path("non_existent")
        self.assertEqual(path3, "")
        
        logger.info("✓ 图像路径解析测试通过")
    
    def test_output_format(self):
        """测试输出格式"""
        logger.info("测试输出格式...")
        
        from quick_start import QuickCoIGenerator
        generator = QuickCoIGenerator(
            image_dir=self.image_dir,
            output_dir=self.output_dir,
            use_openai=False
        )
        
        # 生成测试输出
        test_output = [
            {
                "text": "Test text",
                "target": "Test target",
                "gt_label": "1",
                "ImageID": "123456",
                "description": "Test description",
                "reason": "Test reason"
            }
        ]
        
        output_path = os.path.join(self.output_dir, "format_test.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(test_output, f, indent=2, ensure_ascii=False)
        
        # 验证格式
        is_valid = generator.validate_result(output_path)
        self.assertTrue(is_valid)
        
        # 读取并验证内容
        with open(output_path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        self.assertEqual(len(loaded_data), 1)
        self.assertEqual(loaded_data[0]["text"], "Test text")
        
        logger.info("✓ 输出格式测试通过")
    
    def test_error_handling(self):
        """测试错误处理"""
        logger.info("测试错误处理...")
        
        from quick_start import QuickCoIGenerator
        generator = QuickCoIGenerator(
            image_dir=self.image_dir,
            output_dir=self.output_dir,
            use_openai=False
        )
        
        # 测试无效样本
        invalid_sample = {"invalid": "data"}
        result = generator.process_sample(invalid_sample)
        
        # 应该返回None或有效结果
        if result is not None:
            self._validate_sample_format(result)
        
        logger.info("✓ 错误处理测试通过")
    
    def test_label_mapping(self):
        """测试标签映射"""
        logger.info("测试标签映射...")
        
        from quick_start import QuickCoIGenerator
        generator = QuickCoIGenerator(
            image_dir=self.image_dir,
            output_dir=self.output_dir,
            use_openai=False
        )
        
        # 测试标签映射
        self.assertEqual(generator.label_map["0"], "Neutral")
        self.assertEqual(generator.label_map["1"], "Positive")
        self.assertEqual(generator.label_map["-1"], "Negative")
        
        # 测试无效标签
        self.assertEqual(generator.label_map.get("999", "Neutral"), "Neutral")
        
        logger.info("✓ 标签映射测试通过")
    
    def _validate_sample_format(self, sample: Dict[str, Any]):
        """验证样本格式"""
        required_fields = ['text', 'target', 'gt_label', 'ImageID', 'description', 'reason']
        
        for field in required_fields:
            self.assertIn(field, sample)
            self.assertIsInstance(sample[field], str)
        
        # 验证标签值
        self.assertIn(sample['gt_label'], ['0', '1', '-1'])


def run_comprehensive_test():
    """运行综合测试"""
    logger.info("=" * 60)
    logger.info("开始ArkMSA CoI阶段综合测试")
    logger.info("=" * 60)
    
    # 运行单元测试
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCoIStage)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 统计结果
    logger.info("\n" + "=" * 60)
    logger.info(f"测试运行: {result.testsRun}")
    logger.info(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    logger.info(f"失败: {len(result.failures)}")
    logger.info(f"错误: {len(result.errors)}")
    
    if result.failures:
        logger.error("\n失败的测试:")
        for test, traceback in result.failures:
            logger.error(f"- {test}")
    
    if result.errors:
        logger.error("\n错误的测试:")
        for test, traceback in result.errors:
            logger.error(f"- {test}")
    
    logger.info("=" * 60)
    
    return result.wasSuccessful()


def create_sample_data():
    """创建示例数据"""
    logger.info("创建示例数据...")
    
    # 示例Twitter15/17格式的数据
    sample_data = [
        {
            "text": "I love the new $T$! It's amazing! #iPhone",
            "target": "iPhone",
            "gt_label": "1",
            "ImageID": "sample_001"
        },
        {
            "text": "The weather is terrible today. $T$ sucks!",
            "target": "weather",
            "gt_label": "-1",
            "ImageID": "sample_002"
        },
        {
            "text": "Just had lunch at $T$. It was okay.",
            "target": "restaurant",
            "gt_label": "0",
            "ImageID": "sample_003"
        },
        {
            "text": "RT @user: $T$ is the best phone ever!",
            "target": "Samsung",
            "gt_label": "1",
            "ImageID": "sample_004"
        },
        {
            "text": "I don't like the new $T$ design.",
            "target": "Tesla",
            "gt_label": "-1",
            "ImageID": "sample_005"
        }
    ]
    
    # 保存示例数据
    with open("sample_input.json", 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    logger.info("✓ 示例数据已保存到: sample_input.json")
    
    # 创建示例输出
    sample_output = [
        {
            "text": "I love the new $T$! It's amazing! #iPhone",
            "target": "iPhone",
            "gt_label": "1",
            "ImageID": "sample_001",
            "description": "The image shows a sleek smartphone with a modern design, displayed on a clean white surface with good lighting.",
            "reason": "Sentiment: Positive. The text expresses strong positive sentiment with words like 'love' and 'amazing', and the image shows an attractive product."
        },
        {
            "text": "The weather is terrible today. $T$ sucks!",
            "target": "weather",
            "gt_label": "-1",
            "ImageID": "sample_002",
            "description": "The image depicts a gloomy, overcast sky with dark clouds and rain visible through a window.",
            "reason": "Sentiment: Negative. The text uses negative words like 'terrible' and 'sucks', and the image shows unpleasant weather conditions."
        }
    ]
    
    with open("sample_output.json", 'w', encoding='utf-8') as f:
        json.dump(sample_output, f, indent=2, ensure_ascii=False)
    
    logger.info("✓ 示例输出已保存到: sample_output.json")


if __name__ == "__main__":
    import sys
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == "--create-sample":
            create_sample_data()
        else:
            logger.error(f"未知参数: {sys.argv[1]}")
            logger.info("用法: python test_coi_stage.py [--create-sample]")
    else:
        # 运行测试
        success = run_comprehensive_test()
        
        if success:
            logger.info("\n🎉 所有测试通过！")
        else:
            logger.error("\n❌ 部分测试失败")
            sys.exit(1)
      