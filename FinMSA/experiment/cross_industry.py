import argparse
import random
import numpy as np
import torch
import json
from sklearn.model_selection import train_test_split
import sys
import os
# 获取当前文件所在目录的上级目录（即项目根目录）
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 导入项目中的模块
from config import Config, _MODEL_CLASSES
from data_utils.text_dataset import StockKnow
from data_utils.data_loader import get_plus_data_loader
from models.BERT_mlm import BERT_mlm
from framework import MLM_plus_framework


# ========== 行业映射配置 ==========
def build_industry_map():
    """
    手工构建股票代码到行业的映射字典。
    根据 dev.json 中实际出现的股票代码进行分类。
    """
    map_dict = {
        # ========== 科技 ==========
        'NVDA': 'tech', 'AMD': 'tech', 'AAPL': 'tech', 'MSFT': 'tech',
        'GOOG': 'tech', 'META': 'tech', 'PLTR': 'tech', 'ORCL': 'tech',
        'IBM': 'tech', 'CRM': 'tech', 'SNOW': 'tech', 'NET': 'tech',
        'AI': 'tech', 'PATH': 'tech', 'ARM': 'tech', 'SMCI': 'tech',
        'AVGO': 'tech', 'ROKU': 'tech', 'ZM': 'tech', 'CRWD': 'tech',
        'ADBE': 'tech', 'NOW': 'tech', 'INTU': 'tech', 'QCOM': 'tech',
        'MU': 'tech', 'LRCX': 'tech', 'AMAT': 'tech', 'CDNS': 'tech',
        'SNPS': 'tech', 'PANW': 'tech', 'FTNT': 'tech', 'TEAM': 'tech',
        'DDOG': 'tech', 'MDB': 'tech', 'ZS': 'tech', 'OKTA': 'tech',
        'WDAY': 'tech', 'VMW': 'tech', 'EBAY': 'tech', 'PYPL': 'tech',
        'SQ': 'tech', 'SHOP': 'tech', 'ADSK': 'tech', 'SNAP': 'tech',
        'PINS': 'tech', 'TWLO': 'tech', 'UBER': 'tech', 'LYFT': 'tech',
        'DOCU': 'tech', 'Z': 'tech', 'ZG': 'tech', 'RNG': 'tech',
        'FSLY': 'tech',

        # ========== 金融 ==========
        'JPM': 'finance', 'BAC': 'finance', 'WFC': 'finance', 'C': 'finance',
        'GS': 'finance', 'MS': 'finance', 'XLF': 'finance', 'KRE': 'finance',
        'V': 'finance', 'MA': 'finance', 'AXP': 'finance', 'COF': 'finance',
        'DFS': 'finance', 'SCHW': 'finance', 'TROW': 'finance', 'BLK': 'finance',
        'BK': 'finance', 'STT': 'finance', 'NTRS': 'finance', 'FITB': 'finance',
        'KEY': 'finance', 'HBAN': 'finance', 'RF': 'finance', 'CFG': 'finance',
        'ZION': 'finance', 'MTB': 'finance', 'PNC': 'finance', 'USB': 'finance',
        'TFC': 'finance', 'CMA': 'finance', 'ALL': 'finance', 'PRU': 'finance',
        'MET': 'finance', 'AIG': 'finance', 'AFL': 'finance', 'TRV': 'finance',
        'CB': 'finance', 'PGR': 'finance', 'CINF': 'finance', 'L': 'finance',
        'MMC': 'finance', 'AON': 'finance', 'AJG': 'finance', 'BRK.B': 'finance',
        'ICE': 'finance', 'CME': 'finance', 'MCO': 'finance', 'SPGI': 'finance',
        'FDS': 'finance', 'MSCI': 'finance',

        # ========== 消费 ==========
        'AMZN': 'consumer', 'TGT': 'consumer', 'COST': 'consumer', 'MCD': 'consumer',
        'SBUX': 'consumer', 'NKE': 'consumer', 'LULU': 'consumer', 'TJX': 'consumer',
        'ROST': 'consumer', 'HD': 'consumer', 'LOW': 'consumer', 'WMT': 'consumer',
        'DG': 'consumer', 'DLTR': 'consumer', 'CVS': 'consumer', 'WBA': 'consumer',
        'KR': 'consumer', 'GIS': 'consumer', 'K': 'consumer', 'MDLZ': 'consumer',
        'PEP': 'consumer', 'KO': 'consumer', 'MNST': 'consumer', 'TAP': 'consumer',
        'STZ': 'consumer', 'MO': 'consumer', 'PM': 'consumer', 'PG': 'consumer',
        'EL': 'consumer', 'CLX': 'consumer', 'CHD': 'consumer', 'KMB': 'consumer',
        'COTY': 'consumer',

        # ========== 汽车 ==========
        'TSLA': 'auto', 'RIVN': 'auto', 'NIO': 'auto', 'F': 'auto',
        'GM': 'auto', 'LCID': 'auto', 'XPEV': 'auto', 'LI': 'auto',
        'STLA': 'auto', 'HMC': 'auto', 'TM': 'auto', 'VWAGY': 'auto',
        'BMWYY': 'auto', 'DDAIF': 'auto',

        # ========== 能源/大宗 ==========
        'XOM': 'energy', 'CVX': 'energy', 'UNG': 'energy', 'COP': 'energy',
        'OXY': 'energy', 'EOG': 'energy', 'PSX': 'energy', 'VLO': 'energy',
        'MPC': 'energy', 'KMI': 'energy', 'WMB': 'energy', 'OKE': 'energy',
        'ET': 'energy', 'EPD': 'energy', 'LNG': 'energy', 'DUK': 'energy',
        'SO': 'energy', 'NEE': 'energy', 'AEP': 'energy', 'XLE': 'energy',
        'USO': 'energy', 'UCO': 'energy', 'PPL': 'energy', 'PXD': 'energy',
        'DVN': 'energy', 'HES': 'energy', 'MRO': 'energy', 'APA': 'energy',
        'CL': 'energy',  # 原油期货，归为能源

        # ========== 医药/医疗 ==========
        'PFE': 'health', 'UNH': 'health', 'JNJ': 'health', 'MRK': 'health',
        'LLY': 'health', 'ABBV': 'health', 'AMGN': 'health', 'GILD': 'health',
        'BIIB': 'health', 'REGN': 'health', 'VRTX': 'health', 'MRNA': 'health',
        'BNTX': 'health', 'NVO': 'health', 'SNY': 'health', 'AZN': 'health',
        'BMY': 'health',

        # ========== 工业 ==========
        'CAT': 'industrial', 'BA': 'industrial', 'GE': 'industrial', 'MMM': 'industrial',
        'HON': 'industrial', 'LMT': 'industrial', 'NOC': 'industrial', 'GD': 'industrial',
        'RTX': 'industrial', 'UPS': 'industrial', 'FDX': 'industrial', 'UNP': 'industrial',
        'NSC': 'industrial', 'CSX': 'industrial', 'CNI': 'industrial', 'KSU': 'industrial',
        'WM': 'industrial', 'RSG': 'industrial', 'DE': 'industrial', 'CMI': 'industrial',
        'PCAR': 'industrial', 'ITW': 'industrial', 'ETN': 'industrial', 'PH': 'industrial',
        'ROP': 'industrial', 'DOV': 'industrial', 'IR': 'industrial', 'FAST': 'industrial',
        'GWW': 'industrial', 'EMR': 'industrial', 'ROK': 'industrial', 'AME': 'industrial',
        'APD': 'industrial', 'LIN': 'industrial', 'SHW': 'industrial', 'PPG': 'industrial',
        'ECL': 'industrial', 'IFF': 'industrial', 'DD': 'industrial', 'FMC': 'industrial',
        'CF': 'industrial', 'MOS': 'industrial', 'NUE': 'industrial', 'STLD': 'industrial',
        'X': 'industrial', 'AA': 'industrial', 'ALB': 'industrial', 'FCX': 'industrial',
        'SCCO': 'industrial', 'TECK': 'industrial', 'BHP': 'industrial', 'RIO': 'industrial',
        'VALE': 'industrial', 'IP': 'industrial',

        # ========== 其他（ETF/指数/贵金属/加密货币） ==========
        'SPY': 'other', 'BTC': 'other', 'ETH': 'other', 'DXY': 'other', 'VIX': 'other',
        'TLT': 'other', 'IEF': 'other', 'JNK': 'other', 'LQD': 'other', 'HYG': 'other',
        'EEM': 'other', 'EFA': 'other', 'IWM': 'other', 'QQQ': 'other', 'DIA': 'other',
        'VOO': 'other', 'IVV': 'other', 'RSP': 'other', 'MDY': 'other', 'VB': 'other',
        'VO': 'other', 'VUG': 'other', 'VTV': 'other', 'IWB': 'other', 'IWD': 'other',
        'IWF': 'other', 'IWO': 'other', 'IWN': 'other', 'IJS': 'other', 'IJT': 'other',
        'IJK': 'other', 'IJJ': 'other', 'IWP': 'other', 'IWS': 'other', 'IWV': 'other',
        'IWY': 'other', 'IWZ': 'other', 'IWX': 'other',
        'NDX': 'other', 'RTY': 'other',
        'GOLD': 'other', 'SLV': 'other', 'GLD': 'other', 'IAU': 'other',
    }
    return map_dict


def get_industry_from_target(target_str, industry_map):
    """
    从 target 字符串（如 "TICKER$NVDA"）中提取股票代码并返回行业
    """
    code = target_str.replace('TICKER$', '').strip().split()[0].upper()
    return industry_map.get(code, None)


def load_all_samples(root_dir):
    """
    加载所有 json 文件（train/dev/test）并合并为一个列表
    """
    all_samples = []
    for split in ['train', 'dev', 'test']:
        file_path = os.path.join(root_dir, f'{split}.json')
        if not os.path.exists(file_path):
            continue
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_samples.extend(data)
    return all_samples


def filter_by_industries(samples, industries, industry_map):
    """
    筛选出属于指定行业的样本
    """
    filtered = []
    for sample in samples:
        target = sample.get('target', '')
        industry = get_industry_from_target(target, industry_map)
        if industry in industries:
            filtered.append(sample)
    return filtered


def prepare_data_for_processor(samples, use_caption=True, use_reason=True):
    """
    将样本列表转换为 StockKnow 中 bert_mlmp_process 函数期望的格式
    参数:
        use_caption: 是否包含图像描述 (description)
        use_reason: 是否包含推理理由 (reason)
    返回: ids, texts, descrips, reasons, images, labels, targets
    """
    ids = []
    texts = []
    descrips = []
    reasons = []
    images = []
    labels = []
    targets = []
    for sample in samples:
        target = sample['target'].replace('TICKER$', '')
        targets.append(target)
        ids.append(str(sample['ImageID']))
        images.append(str(sample['ImageID']))
        labels.append(int(sample['gt_label']) + 1)  # 转换为 0/1/2

        text = sample['text'].replace('$T$', 'Jake Paul')
        texts.append(text)

        # 根据参数决定是否使用描述
        if use_caption:
            descrips.append(sample['description'])
        else:
            descrips.append('')   # 空字符串，模型仍会处理但无实际内容

        # 根据参数决定是否使用理由
        if use_reason:
            reasons.append(sample['reason'])
        else:
            reasons.append('')

    return ids, texts, descrips, reasons, images, labels, targets


def run_cross_industry_experiment(args):
    # 1. 读取配置
    config = Config(args.config)
    # 根据 encoder_type 设置正确的模型路径
    paths = {'bert': config.bert_path,
             'roberta': config.roberta_path,
             'finbert': config.finbert_path}
    config.bert_path = paths[config.encoder_type]
    config.few_shot = None  # 跨行业实验不使用 few-shot

    # 2. 设置随机种子
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True

    # 3. 构建行业映射
    industry_map = build_industry_map()

    # 4. 加载所有原始样本
    print("Loading all samples...")
    all_samples = load_all_samples(config.root_dir)
    print(f"Total samples: {len(all_samples)}")

    # 5. 按行业筛选训练集和测试集样本
    train_industries = args.train_industries.split(',') if args.train_industries else ['tech', 'consumer']
    test_industries = args.test_industries.split(',') if args.test_industries else ['finance', 'auto']
    print(f"Training industries: {train_industries}")
    print(f"Testing industries: {test_industries}")

    train_samples = filter_by_industries(all_samples, train_industries, industry_map)
    test_samples = filter_by_industries(all_samples, test_industries, industry_map)
    print(f"Train samples: {len(train_samples)}, Test samples: {len(test_samples)}")

    # 6. 从训练集中划分验证集（10%）
    if len(train_samples) > 0:
        train_samples, val_samples = train_test_split(train_samples, test_size=0.1, random_state=config.seed)
    else:
        val_samples = []
    print(f"After split: Train = {len(train_samples)}, Val = {len(val_samples)}, Test = {len(test_samples)}")

    # 7. 初始化 StockKnow 处理器
    tokenizer_class = _MODEL_CLASSES[config.encoder_type]['tokenizer']
    processor = StockKnow(config, config.root_dir, tokenizer_class)
    processor.tokenizer = tokenizer_class.from_pretrained(config.bert_path, additional_special_tokens=["[P00]"])

    # 8. 根据 variant 确定输入配置
    if args.variant == 'text_only':
        use_caption = False
        use_reason = False
    elif args.variant == 'text_caption':
        use_caption = True
        use_reason = False
    else:
        # 默认 text_caption_reason
        use_caption = True
        use_reason = True

    # 9. 预处理数据（注意传入 use_caption, use_reason）
    print("Preprocessing training data...")
    train_raw = prepare_data_for_processor(train_samples, use_caption, use_reason)
    train_data = processor.bert_mlmp_process(train_raw)

    print("Preprocessing validation data...")
    val_raw = prepare_data_for_processor(val_samples, use_caption, use_reason)
    val_data = processor.bert_mlmp_process(val_raw)

    print("Preprocessing test data...")
    test_raw = prepare_data_for_processor(test_samples, use_caption, use_reason)
    test_data = processor.bert_mlmp_process(test_raw)

    # 10. 创建 DataLoader
    train_loader = get_plus_data_loader(config, train_data, shuffle=True)
    val_loader = get_plus_data_loader(config, val_data)
    test_loader = get_plus_data_loader(config, test_data)

    # 11. 初始化模型
    id2name = ["Negative Adverse Unfavorable Pessimistic Hostile Critical Dismal Gloomy Detrimental Defeatist Damaging",
               "Neutral Impartial Unbiased Objective Uninvolved Indifferent Balanced Nonpartisan Disinterested Equitable Fair-minded",
               "Positive Optimistic Favorable Encouraging Upbeat Good Constructive Affirmative Bright Promising Supportive"]
    encoder_class = _MODEL_CLASSES[config.encoder_type]['encoder']
    tokenizer = processor.tokenizer

    model = BERT_mlm(config, rels_num=3,
                     device=config.device,
                     id2name=id2name,
                     encoder=encoder_class,
                     tokenizer=tokenizer,
                     init_by_cls=None)
    model.to(config.device)

    # 12. 训练框架
    framework = MLM_plus_framework(config)

    # 13. 训练
    print("\n=== Starting cross-industry training ===")
    best_val_f1, best_epoch, test_acc, test_macf1, test_wf1 = framework.train(
        config, model, train_loader, val_loader, test_loader
    )

    # 14. 输出最终结果
    print("\n=== Cross-industry experiment results ===")
    print(f"Variant: {args.variant}")
    print(f"Training industries: {train_industries}")
    print(f"Testing industries: {test_industries}")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Test Macro F1: {test_macf1 * 100:.2f}%")
    print(f"Test Weighted F1: {test_wf1 * 100:.2f}%")

    # 保存结果
    with open('cross_industry_results.txt', 'a') as f:
        f.write(f"Variant: {args.variant}\n")
        f.write(f"Train: {train_industries}, Test: {test_industries}\n")
        f.write(f"Acc: {test_acc:.4f}, Macro-F1: {test_macf1:.4f}, Weighted-F1: {test_wf1:.4f}\n")
        f.write("-" * 50 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cross-industry evaluation for stock sentiment analysis")
    parser.add_argument('--config', default='config_dir/StockKnow.ini', help='Path to config file')
    parser.add_argument('--train_industries', default=None,
                        help='Comma-separated list of industries for training, e.g. "tech,consumer"')
    parser.add_argument('--test_industries', default=None,
                        help='Comma-separated list of industries for testing, e.g. "finance,auto"')
    parser.add_argument('--variant', default='text_caption',
                        choices=['text_only', 'text_caption'],
                        help='Input variant: text_only or text_caption')
    args = parser.parse_args()
    run_cross_industry_experiment(args)