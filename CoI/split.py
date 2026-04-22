"""
将现有 train/test 数据重新划分为 train/dev/test = 6:2:2，保持标签分布均衡
"""

import json
import random
from collections import Counter
from sklearn.model_selection import train_test_split


def load_json(file_path):
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, file_path):
    """保存JSON文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def stratified_split(data, labels, train_ratio=0.6, dev_ratio=0.2, test_ratio=0.2, random_seed=42):
    """
    分层划分数据，确保各类别分布均衡

    Args:
        data: 样本列表
        labels: 对应的标签列表
        train_ratio, dev_ratio, test_ratio: 三部分比例
        random_seed: 随机种子
    """
    # 先分出 train + temp (dev + test)
    train_data, temp_data = train_test_split(
        data,
        test_size=(dev_ratio + test_ratio),
        random_state=random_seed,
        stratify=labels
    )

    # 从 temp 中分出 dev 和 test
    temp_labels = [item['gt_label'] for item in temp_data]
    dev_data, test_data = train_test_split(
        temp_data,
        test_size=test_ratio / (dev_ratio + test_ratio),  # test 占 temp 的比例
        random_state=random_seed,
        stratify=temp_labels
    )

    return train_data, dev_data, test_data


def log_distribution(data, name):
    """打印标签分布"""
    labels = [item['gt_label'] for item in data]
    dist = Counter(labels)
    total = len(data)
    print(f"\n{name} (n={total}):")
    for label in ['-1', '0', '1']:
        count = dist.get(label, 0)
        pct = count / total * 100 if total > 0 else 0
        print(f"  {label}: {count} ({pct:.1f}%)")


def main():
    # ==================== 配置 ====================
    original_train_path = "/home/remance/文档/xbb/project/CoI/coi_output_finance_max/train.json"  # 修改为实际路径
    original_test_path = "/home/remance/文档/xbb/project/CoI/coi_output_finance_max/test.json"  # 修改为实际路径

    output_dir = "/home/remance/文档/xbb/project/CoI/coi_output_finance"  # 输出目录
    random_seed = 42
    # =============================================

    # 1. 加载原始数据
    print("=" * 60)
    print("加载原始数据...")
    original_train = load_json(original_train_path)
    original_test = load_json(original_test_path)

    print(f"原 Train: {len(original_train)} 条")
    print(f"原 Test: {len(original_test)} 条")
    print(f"总计: {len(original_train) + len(original_test)} 条")

    # 2. 合并所有数据
    all_data = original_train + original_test
    all_labels = [item['gt_label'] for item in all_data]

    print("\n原始总体分布:")
    log_distribution(all_data, "All Data")

    # 3. 重新分层划分
    print("\n" + "=" * 60)
    print("重新分层划分 (6:2:2)...")

    train_data, dev_data, test_data = stratified_split(
        all_data,
        all_labels,
        train_ratio=0.6,
        dev_ratio=0.2,
        test_ratio=0.2,
        random_seed=random_seed
    )

    # 4. 打乱每个集合内部顺序（可选）
    random.seed(random_seed)
    random.shuffle(train_data)
    random.shuffle(dev_data)
    random.shuffle(test_data)

    # 5. 输出分布统计
    print("\n" + "=" * 60)
    print("划分后分布:")
    log_distribution(train_data, "Train")
    log_distribution(dev_data, "Dev")
    log_distribution(test_data, "Test")

    # 6. 验证总数
    total_new = len(train_data) + len(dev_data) + len(test_data)
    print(f"\n验证总数: {total_new} (应与原始 {len(all_data)} 一致)")

    # 7. 保存文件
    print("\n" + "=" * 60)
    print("保存文件...")
    save_json(train_data, f"{output_dir}/train.json")
    save_json(dev_data, f"{output_dir}/dev.json")
    save_json(test_data, f"{output_dir}/test.json")
    print(f"已保存到 {output_dir}")

    # 8. 显示比例
    print("\n" + "=" * 60)
    print("实际比例:")
    print(f"Train: {len(train_data) / total_new * 100:.1f}%")
    print(f"Dev: {len(dev_data) / total_new * 100:.1f}%")
    print(f"Test: {len(test_data) / total_new * 100:.1f}%")


if __name__ == "__main__":
    main()