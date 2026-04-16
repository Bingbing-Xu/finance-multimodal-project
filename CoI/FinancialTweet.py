"""
Financial Tweets Stocks 数据集预处理脚本
"""

import csv
import os
import json
import requests
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import time
import random
from dataclasses import dataclass
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class FinancialTweetSample:
    """数据样本结构"""
    index: str
    text: str
    target: str
    sentiment_label: str
    image_url: str
    tweet_id: str


class FinancialDatasetPreprocessor:
    """金融推文数据集预处理器"""

    def __init__(
            self,
            raw_csv_path: str,
            output_dir: str,
            train_ratio: float = 0.8,
            random_seed: int = 42,
            skip_downloaded: bool = True,
            delay: float = 0.5
    ):
        """
        初始化预处理器

        Args:
            raw_csv_path: 原始 CSV 文件路径
            output_dir: 输出目录
            train_ratio: 训练集比例（0-1之间）
            random_seed: 随机种子，确保可复现
            skip_downloaded: 是否跳过已下载的图片
            delay: 下载延迟（秒）
        """
        self.raw_csv_path = Path(raw_csv_path)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        self.skip_downloaded = skip_downloaded
        self.delay = delay

        # 创建输出目录结构
        self.tsv_dir = self.output_dir / "tsv"
        self.image_dir = self.output_dir / "images"
        self.tsv_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)

        # 输出文件路径
        self.train_tsv_path = self.tsv_dir / "train.tsv"
        self.test_tsv_path = self.tsv_dir / "test.tsv"

        logger.info(f"初始化预处理器:")
        logger.info(f"  - 输入CSV: {self.raw_csv_path}")
        logger.info(f"  - 输出目录: {self.output_dir}")
        logger.info(f"  - 训练集比例: {train_ratio:.0%}")
        logger.info(f"  - 随机种子: {random_seed}")

    def load_raw_data(self) -> List[Dict[str, Any]]:
        """
        加载原始 CSV 数据，仅保留有图片的样本

        Returns:
            过滤后的数据列表
        """
        data = []
        skipped_no_image = 0
        skipped_invalid = 0

        with open(self.raw_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for idx, row in enumerate(reader, 1):
                try:
                    # 提取字段
                    image_url = row.get('image_url', '').strip()
                    proxy_image_url = row.get('proxy_image_url', '').strip()
                    text = row.get('description', '').strip()

                    # 读取原始情感标签并转为小写
                    sentiment = row.get('sentiment', '').strip().lower()

                    financial_info = row.get('financial_info', '').strip()
                    tweet_url = row.get('url', '').strip()
                    timestamp = row.get('timestamp', '').strip()

                    # 必须使用主 image_url
                    if not image_url:
                        if proxy_image_url:
                            image_url = proxy_image_url
                        else:
                            skipped_no_image += 1
                            continue  # 跳过无图片的样本

                    # 跳过无效数据
                    if not text or sentiment not in ['bearish', 'neutral', 'bullish']:
                        logger.warning(f"跳过无效数据行 {idx}: 文本为空或标签无效 (sentiment='{sentiment}')")
                        skipped_invalid += 1
                        continue

                    # 解析股票代码
                    target = self._extract_stock_symbol(financial_info)
                    if not target:
                        # 从文本中提取股票代码（例如 $AAPL）
                        target = self._extract_symbol_from_text(text)
                        if not target:
                            target = "UNKNOWN"

                    # 生成唯一 tweet_id
                    tweet_id = self._generate_tweet_id(tweet_url, timestamp, idx)

                    data.append({
                        'index': str(len(data)),  # 临时索引，划分后重新编号
                        'text': text,
                        'target': target,
                        'sentiment': sentiment,  # 保存小写形式
                        'image_url': image_url,
                        'tweet_id': tweet_id
                    })

                except Exception as e:
                    logger.error(f"解析第{idx}行失败: {e}")
                    skipped_invalid += 1
                    continue

        logger.info(f"数据加载完成:")
        logger.info(f"  - 有效样本: {len(data)}")
        logger.info(f"  - 无图片跳过: {skipped_no_image}")
        logger.info(f"  - 无效数据跳过: {skipped_invalid}")

        # 统计标签分布
        self._log_label_distribution(data)

        return data

    def _extract_stock_symbol(self, financial_info: str) -> Optional[str]:
        """从 financial_info 字段提取股票代码"""
        if not financial_info:
            return None

        try:
            # 尝试解析JSON
            if financial_info.startswith('{') or financial_info.startswith('['):
                info = json.loads(financial_info)
                if isinstance(info, dict) and 'symbol' in info:
                    return info['symbol'].upper()
                elif isinstance(info, list) and len(info) > 0:
                    return info[0].get('symbol', '').upper()

            # 如果不是JSON，直接使用字符串
            symbol = ''.join(c for c in financial_info if c.isalnum() or c in ['$', '-'])
            return symbol.upper()[:10] if symbol else None

        except json.JSONDecodeError:
            symbol = ''.join(c for c in financial_info if c.isalnum() or c in ['$', '-'])
            return symbol.upper()[:10] if symbol else None

    def _extract_symbol_from_text(self, text: str) -> Optional[str]:
        """
        从文本中提取股票代码（例如 $AAPL, $TSLA）

        Args:
            text: 推文文本

        Returns:
            股票代码或 None
        """
        import re
        # 匹配 $AAPL, $TSLA 等格式
        match = re.search(r'\$([A-Z]{2,5})', text)
        if match:
            return match.group(1)
        return None

    def _generate_tweet_id(self, tweet_url: str, timestamp: str, idx: int) -> str:
        """生成唯一的 tweet_id"""
        if tweet_url:
            parts = tweet_url.strip('/').split('/')
            if len(parts) >= 2:
                return parts[-1]

        if timestamp:
            return f"tweet_{timestamp}"

        return f"tweet_{idx}"

    def download_image(self, image_url: str, image_id: str) -> bool:
        """下载图片"""
        # 检测文件扩展名
        ext = '.jpg'  # 默认扩展名
        url_filename = image_url.split('?')[0].split('/')[-1]
        if '.' in url_filename:
            url_ext = url_filename.split('.')[-1].lower()
            if url_ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp']:
                ext = f'.{url_ext}'

        # 保存路径
        image_path = self.image_dir / f"{image_id}{ext}"

        # 如果文件已存在且设置为跳过，则直接返回
        if self.skip_downloaded and image_path.exists():
            logger.debug(f"图片已存在，跳过: {image_path}")
            return True

        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(image_url, headers=headers, timeout=30)
            response.raise_for_status()

            # 检查内容类型
            content_type = response.headers.get('content-type', '')
            if 'image' not in content_type:
                logger.warning(f"URL返回的不是图片: {content_type}")
                return False

            with open(image_path, 'wb') as f:
                f.write(response.content)

            logger.info(f"成功下载: {image_id}{ext} ({len(response.content)} bytes)")
            time.sleep(self.delay)
            return True

        except Exception as e:
            logger.warning(f"下载失败 {image_url}: {e}")
            return False

    def split_dataset(self, data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        划分训练集和测试集，保持标签分布

        Args:
            data: 原始数据列表

        Returns:
            (train_data, test_data) 元组
        """
        logger.info(f"\n开始划分数据集（比例: {self.train_ratio:.0%} train, {1-self.train_ratio:.0%} test）...")

        # 按标签分组，确保每类样本都按比例划分
        label_groups = {'bearish': [], 'neutral': [], 'bullish': []}
        for sample in data:
            label_groups[sample['sentiment']].append(sample)

        train_data = []
        test_data = []

        # 对每组进行划分
        for label, samples in label_groups.items():
            if not samples:
                continue

            # 打乱顺序
            random.seed(self.random_seed)
            random.shuffle(samples)

            # 计算划分点
            split_idx = int(len(samples) * self.train_ratio)

            train_data.extend(samples[:split_idx])
            test_data.extend(samples[split_idx:])

            logger.info(
                f"  - {label}: {len(samples)} 条 → train: {len(samples[:split_idx])}, test: {len(samples[split_idx:])}")

        # 重新打乱，确保训练集和测试集内顺序随机
        random.seed(self.random_seed + 1)
        random.shuffle(train_data)
        random.shuffle(test_data)

        # 重新编号索引
        for idx, sample in enumerate(train_data):
            sample['index'] = str(idx)
        for idx, sample in enumerate(test_data):
            sample['index'] = str(idx)

        logger.info(f"划分完成:")
        logger.info(f"  - 训练集: {len(train_data)} 条")
        logger.info(f"  - 测试集: {len(test_data)} 条")
        logger.info(f"  - 总计: {len(train_data) + len(test_data)} 条")

        return train_data, test_data

    def _log_label_distribution(self, data: List[Dict[str, Any]]):
        """统计并记录标签分布"""
        from collections import Counter
        sentiments = [s['sentiment'] for s in data]
        dist = Counter(sentiments)

        logger.info(f"标签分布:")
        for label in ['bearish', 'neutral', 'bullish']:
            count = dist.get(label, 0)
            pct = count / len(data) * 100 if data else 0
            logger.info(f"  - {label}: {count} 条 ({pct:.1f}%)")

    def convert_label_to_framework_format(self, sentiment: str) -> str:
        """

        Args:
            sentiment: 小写情感标签 (bearish/neutral/bullish)

        Returns:
            转换后的标签 (-1/0/1)
        """
        # BUGFIX: 所有键都改为小写，与 load_raw_data() 保持一致
        label_mapping = {
            'bearish': '-1',
            'neutral': '0',
            'bullish': '1'
        }

        if sentiment not in label_mapping:
            logger.warning(f"未知情感标签 '{sentiment}'，默认设为中立")
            return '0'

        return label_mapping[sentiment]

    def process_and_save_subset(self, data: List[Dict[str, Any]], subset_name: str) -> int:
        """
        处理并保存一个子集（train或test）

        Args:
            data: 子集数据
            subset_name: 'train' 或 'test'

        Returns:
            成功下载的图片数量
        """
        logger.info(f"\n开始处理 {subset_name.upper()} 集...")

        tsv_rows = []
        successful_downloads = 0

        for idx, sample in enumerate(data):
            image_url = sample['image_url']
            tweet_id = sample['tweet_id']

            success = self.download_image(image_url, tweet_id)

            if not success:
                logger.warning(f"跳过样本 {idx + 1}: 图片下载失败")
                continue

            # 转换标签（输出 -1, 0, 1）
            framework_label = self.convert_label_to_framework_format(sample['sentiment'])

            tsv_row = {
                'index': sample['index'],
                'label': framework_label,
                'ImageID': tweet_id,
                'text': sample['text'].replace('\t', ' ').replace('\n', ' ').strip(),
                'target': sample['target']
            }

            tsv_rows.append(tsv_row)
            successful_downloads += 1

            if (successful_downloads) % 50 == 0:
                logger.info(f"  - 已处理: {successful_downloads}/{len(data)}")

        # 保存 TSV 文件
        tsv_path = self.tsv_dir / f"{subset_name}.tsv"
        self.save_tsv(tsv_rows, tsv_path)

        logger.info(f"{subset_name.upper()} 集处理完成: {successful_downloads}/{len(data)} 条")

        # 记录该子集的标签分布
        self._log_subset_label_distribution(tsv_rows, subset_name)

        return successful_downloads

    def _log_subset_label_distribution(self, rows: List[Dict[str, str]], subset_name: str):
        """记录子集的标签分布"""
        from collections import Counter
        labels = [row['label'] for row in rows]
        dist = Counter(labels)

        logger.info(f"{subset_name.upper()} 集标签分布:")
        label_map = {'-1': 'Bearish', '0': 'Neutral', '1': 'Bullish'}
        for label in ['-1', '0', '1']:
            count = dist.get(label, 0)
            pct = count / len(rows) * 100 if rows else 0
            logger.info(f"  - {label_map[label]}: {count} 条 ({pct:.1f}%)")

    def save_tsv(self, rows: List[Dict[str, str]], tsv_path: Path):
        """保存 TSV 文件（无表头）"""
        with open(tsv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter='\t')

            for row in rows:
                writer.writerow([
                    row['index'],
                    row['label'],
                    row['ImageID'],
                    row['text'],
                    row['target']
                ])

        logger.info(f"TSV 文件已保存: {tsv_path} ({len(rows)} 行)")

    def preprocess_and_save(self):
        """执行完整的预处理流程"""
        logger.info("=" * 60)
        logger.info("开始预处理 Financial Tweets Stocks 数据集")
        logger.info("=" * 60)

        # 1. 加载原始数据（过滤无图片样本）
        logger.info("步骤 1: 加载原始数据...")
        raw_data = self.load_raw_data()

        if not raw_data:
            logger.error("未找到任何带图片的有效数据")
            return

        # 2. 划分数据集
        logger.info("步骤 2: 划分数据集...")
        train_data, test_data = self.split_dataset(raw_data)

        # 3. 处理并保存 Train 集
        logger.info("步骤 3: 处理训练集...")
        train_count = self.process_and_save_subset(train_data, 'train')

        # 4. 处理并保存 Test 集
        logger.info("步骤 4: 处理测试集...")
        test_count = self.process_and_save_subset(test_data, 'test')

        # 5. 生成总体摘要
        self.generate_summary(train_count, test_count)

        logger.info("=" * 60)
        logger.info("预处理完成！")
        logger.info(f"  - 训练集: {train_count} 条")
        logger.info(f"  - 测试集: {test_count} 条")
        logger.info(f"  - 总计: {train_count + test_count} 条")
        logger.info("=" * 60)

        # 6. 显示配置模板
        self.show_config_template()

    def generate_summary(self, train_count: int, test_count: int):
        """生成数据集摘要"""
        # 读取TSV文件统计标签分布
        def count_labels(file_path: Path) -> Dict[str, int]:
            counts = {"-1": 0, "0": 0, "1": 0}
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    if len(row) >= 2:
                        label = row[1]
                        if label in counts:
                            counts[label] += 1
            return counts

        train_labels = count_labels(self.train_tsv_path)
        test_labels = count_labels(self.test_tsv_path)

        summary = {
            "train_samples": train_count,
            "test_samples": test_count,
            "total_samples": train_count + test_count,
            "train_ratio": self.train_ratio,
            "label_distribution": {
                "train": train_labels,
                "test": test_labels
            },
            "output_files": {
                "train_tsv": str(self.train_tsv_path),
                "test_tsv": str(self.test_tsv_path),
                "image_dir": str(self.image_dir)
            }
        }

        # 保存摘要
        summary_path = self.output_dir / "dataset_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info("\n===== 数据集摘要 =====")
        logger.info(f"训练集: {summary['train_samples']} 条")
        logger.info(f"测试集: {summary['test_samples']} 条")
        logger.info(f"标签分布 (Train): {summary['label_distribution']['train']}")
        logger.info(f"标签分布 (Test): {summary['label_distribution']['test']}")

    def show_config_template(self):
        """显示配置模板"""
        template = f"""
# ============= ArkMSA 框架配置模板 =============
# 请将此配置复制到 coi_stage_framework.py 中

# 训练集配置
train_config = {{
    "image_dir": "{self.image_dir}",           # 图片目录（train/test共用）
    "input_data": "{self.train_tsv_path}",      # 训练集 TSV
    "data_format": "tsv",
    "skip_header": False,
    "output_dir": "./coi_output_train",
    "model_name": "qwen-vl-max",
    "batch_size": 1,
    "delay": 3.0,
    "save_interval": 1000
}}

# 测试集配置
test_config = {{
    "image_dir": "{self.image_dir}",           # 图片目录（train/test共用）
    "input_data": "{self.test_tsv_path}",       # 测试集 TSV
    "data_format": "tsv",
    "skip_header": False,
    "output_dir": "./coi_output_test",
    "model_name": "qwen-vl-max",
    "batch_size": 1,
    "delay": 3.0,
    "save_interval": 1000
}}

# 先处理训练集：
# processor = CoIProcessor(mllm_adapter=qwen_adapter, **train_config)
# 再处理测试集：
# processor = CoIProcessor(mllm_adapter=qwen_adapter, **test_config)
# =============================================
"""
        print(template)


def main():
    """主函数"""

    # ==================== 配置参数 ====================
    config = {
        # 输入文件（从 HuggingFace 下载的 CSV）
        "raw_csv_path": "D:/PythonProjects/financial_tweet_socket/financial-tweets-stocks/stock.csv",

        # 输出目录
        "output_dir": "D:/PythonProjects/finance_research/FinancialDataset",

        # 训练集比例（0.8 = 80% train, 20% test）
        "train_ratio": 0.8,

        # 随机种子（确保可复现）
        "random_seed": 42,

        # 其他参数
        "skip_downloaded": True,  # 断点续传
        "download_delay": 0.5     # 下载延迟（秒）
    }
    # =================================================

    # 检查输入文件
    if not os.path.exists(config["raw_csv_path"]):
        logger.error(f"错误: 输入文件不存在: {config['raw_csv_path']}")
        logger.error("请先从 HuggingFace 下载数据集:")
        logger.error("  huggingface-cli download StephanAkkerman/financial-tweets-stocks")
        return

    # 初始化预处理器
    preprocessor = FinancialDatasetPreprocessor(
        raw_csv_path=config["raw_csv_path"],
        output_dir=config["output_dir"],
        train_ratio=config["train_ratio"],
        random_seed=config["random_seed"],
        skip_downloaded=config["skip_downloaded"],
        delay=config["download_delay"]
    )

    # 执行预处理
    preprocessor.preprocess_and_save()


if __name__ == "__main__":
    main()