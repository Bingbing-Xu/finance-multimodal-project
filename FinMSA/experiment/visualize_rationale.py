import json
import os
import sys
import re
import torch

# ========== 修复 OpenMP 冲突 ==========
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import warnings
warnings.filterwarnings('ignore')

# 获取当前文件所在目录的上级目录（即项目根目录）
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免GUI问题
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from argparse import ArgumentParser
from config import Config, _MODEL_CLASSES
from data_utils.text_dataset import StockKnow
from data_utils.data_loader import get_plus_data_loader
from models.BERT_mlm import BERT_mlm
from data_utils.dataset_utils import normalize_word

# ========== 增强停用词列表 ==========
STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'for', 'from', 'has', 'have', 'he',
    'in', 'is', 'it', 'its', 'of', 'on', 'or', 'she', 'that', 'the', 'this', 'to', 'was', 'were',
    'will', 'with', 'i', 'me', 'my', 'we', 'us', 'our', 'you', 'your', 'they', 'them', 'their',
    'not', 'do', 'does', 'did', 'being', 'been', 'are', 'am', 'can', 'could', 'would', 'should',
    'may', 'might', 'must', 'shall', 'has', 'have', 'had', 'go', 'goes', 'going', 'went', 'gone',
    'see', 'sees', 'saw', 'seen', 'look', 'looks', 'looked', 'looking', 'like', 'likes', 'liked',
    'just', 'now', 'here', 'there', 'then', 'when', 'where', 'why', 'how', 'what', 'which',
    'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 'via', 'over', 'under', 'between', 'into',
    'adi','du','sox','tan','nk','dil','ev','pt','cd','hum','goo','gp','amc','aka','nfl','cl','mara',
    'fm'
}

def is_valid_token(token):
    """判断 token 是否为有意义的词汇（过滤停用词、子词、数字、纯符号、过短）"""
    if not token:
        return False
    # 过滤特殊 token
    if token in ['[CLS]', '[SEP]', '[PAD]']:
        return False
    # 过滤子词（WordPiece 生成的 ##xxx）
    if token.startswith('##'):
        return False
    # 过滤纯数字
    if token.isdigit():
        return False
    # 过滤不含任何字母的 token（纯符号或数字符号混合）
    if not any(c.isalpha() for c in token):
        return False
    # 过滤停用词（忽略大小写）
    if token.lower() in STOPWORDS:
        return False
    # 过滤过短的（长度小于2，如 'a' 已被停用词过滤，但仍有单字符字母可能）
    if len(token) < 2:
        return False
    return True

def filter_importance(importance):
    """对重要性字典应用过滤，返回过滤后的字典"""
    filtered = {k: v for k, v in importance.items() if is_valid_token(k)}
    return filtered

# ========== 模型加载 ==========
def load_model(config_path, checkpoint_path, device):
    print(f"Loading config from: {config_path}")
    config = Config(config_path)
    paths = {
        'bert': config.bert_path,
        'roberta': config.roberta_path,
        'finbert': config.finbert_path
    }
    config.bert_path = paths[config.encoder_type]
    config.device = device
    config.few_shot = None

    encoder = _MODEL_CLASSES[config.encoder_type]
    print(f"Loading tokenizer from: {config.bert_path}")
    tokenizer = encoder['tokenizer'].from_pretrained(config.bert_path)
    processor = StockKnow(config, config.root_dir, tokenizer)

    # 加载测试数据
    test_json_path = os.path.join(config.root_dir, 'test.json')
    print(f"Loading test data from: {test_json_path}")

    if not os.path.exists(test_json_path):
        raise FileNotFoundError(f"Test file not found: {test_json_path}")

    with open(test_json_path, 'r', encoding='utf-8') as f:
        test_json = json.load(f)

    def extract_raw(data):
        ids, texts, descrips, reasons, imgs, labels, targets = [], [], [], [], [], [], []
        for item in data:
            target = item['target']
            targets.append(target)
            ids.append(str(item['ImageID']))
            imgs.append(str(item['ImageID']))
            labels.append(int(item['gt_label']) + 1)
            text = item['text'].replace('$T$', target)
            texts.append(normalize_word(text))
            descrips.append(normalize_word(item['description']))
            reasons.append(normalize_word(item['reason']))
        return ids, texts, descrips, reasons, imgs, labels, targets

    test_raw = extract_raw(test_json)
    print(f"Loaded {len(test_raw[0])} test samples")

    test_data = processor.bert_mlmp_process(test_raw)
    test_loader = get_plus_data_loader(config, test_data, batch_size=config.batch_size, shuffle=False)

    # 模型结构
    id2name = ["Negative Adverse Unfavorable Pessimistic Hostile Critical Dismal Gloomy Detrimental Defeatist Damaging",
               "Neutral Impartial Unbiased Objective Uninvolved Indifferent Balanced Nonpartisan Disinterested Equitable Fair-minded",
               "Positive Optimistic Favorable Encouraging Upbeat Good Constructive Affirmative Bright Promising Supportive"]

    print(f"Initializing model: {config.encoder_type}")
    model = BERT_mlm(
        config,
        rels_num=3,
        device=device,
        id2name=id2name,
        encoder=encoder['encoder'],
        tokenizer=tokenizer,
        init_by_cls=None
    ).to(device)

    print(f"Loading checkpoint from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, tokenizer, test_loader, config

# ========== 注意力重要性计算（带过滤） ==========
def compute_token_importance(model, test_loader, tokenizer, device):
    token_attentions = Counter()
    token_counts = Counter()
    model.eval()

    total_batches = len(test_loader)
    print(f"Total batches to process: {total_batches}")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            labels, input_ids, token_type_ids, attention_mask, reason_input_ids, reason_token_type_ids, reason_attention_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)

            if hasattr(model, 'encoder'):
                encoder = model.encoder
            else:
                encoder = model.model

            outputs = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_attentions=True
            )

            if outputs.attentions is None:
                print(f"Batch {batch_idx}: No attentions returned.")
                continue

            att = outputs.attentions[-1].mean(dim=1)
            cls_att = att[:, 0, :]

            for i in range(input_ids.size(0)):
                ids = input_ids[i]
                mask = attention_mask[i]
                att_weights = cls_att[i]

                if batch_idx == 0 and i == 0:
                    # 使用 skip_special_tokens=False 显示原始 token 便于调试
                    tokens_raw = [tokenizer.decode([ids[j].item()], skip_special_tokens=False) for j in range(min(10, len(ids)))]
                    print(f"Sample tokens (raw): {tokens_raw}")
                    print(f"Mask: {mask[:10].tolist()}")

                for j in range(ids.size(0)):
                    if mask[j] == 0:
                        break
                    # 使用 skip_special_tokens=True 自动合并子词
                    token = tokenizer.decode([ids[j].item()], skip_special_tokens=True)
                    # 在收集阶段就过滤
                    if is_valid_token(token):
                        token_attentions[token] += att_weights[j].item()
                        token_counts[token] += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{total_batches} batches")

    if not token_attentions:
        print("Error: No tokens collected.")
        return {}

    avg_importance = {t: token_attentions[t] / token_counts[t] for t in token_attentions}
    print(f"Collected {len(avg_importance)} unique tokens (after filtering)")
    return avg_importance

# ========== 可视化函数（使用过滤后的重要性） ==========
def plot_bar_importance(importance, title, top_k=20, save_path=None):
    if not importance:
        print("No importance data to plot.")
        return False

    # 再次过滤，确保干净
    filtered = filter_importance(importance)
    if not filtered:
        print("No tokens after filtering.")
        return False

    items = sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:top_k]
    tokens, scores = zip(*items)

    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(tokens)), scores, color='skyblue')
    plt.yticks(range(len(tokens)), tokens)
    plt.xlabel('Average Attention Weight', fontsize=12)
    plt.title(title, fontsize=14)
    plt.gca().invert_yaxis()

    for i, (bar, score) in enumerate(zip(bars, scores)):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                 f'{score:.4f}', va='center', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Bar chart saved to: {save_path}")

    plt.close()
    return True

def plot_wordcloud(importance, title, save_path=None):
    if not importance:
        print("No importance data to plot.")
        return False

    # 应用过滤
    filtered = filter_importance(importance)
    if not filtered:
        print("No tokens after filtering.")
        return False

    try:
        # 可选：进一步过滤低权重 token
        filtered = {k: v for k, v in filtered.items() if v > 0.001}
        if len(filtered) < 5:
            print(f"Warning: Only {len(filtered)} tokens with importance > 0.001")
            # 如果不满足，就使用全部过滤后的（不再降低阈值）

        wc = WordCloud(
            width=1200,
            height=600,
            background_color='white',
            max_words=100,
            relative_scaling=0.5,
            colormap='viridis'
        ).generate_from_frequencies(filtered)

        plt.figure(figsize=(12, 6))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=14)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Wordcloud saved to: {save_path}")

        plt.close()
        return True

    except Exception as e:
        print(f"Wordcloud generation failed: {e}")
        return False

# ========== 主程序 ==========
def main():
    parser = ArgumentParser()
    parser.add_argument('--config', default='config_dir/StockKnow.ini')
    parser.add_argument('--checkpoint', required=True, help='模型权重路径')
    parser.add_argument('--output_dir', default='./attention_results')
    parser.add_argument('--top_k', type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(args.output_dir)}")

    if not os.access(args.output_dir, os.W_OK):
        raise PermissionError(f"No write permission to: {args.output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from {args.checkpoint}...")
    try:
        model, tokenizer, test_loader, _ = load_model(args.config, args.checkpoint, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return

    print("Computing token importance...")
    try:
        importance = compute_token_importance(model, test_loader, tokenizer, device)
    except Exception as e:
        print(f"Error computing importance: {e}")
        import traceback
        traceback.print_exc()
        return

    if importance:
        print(f"\nGenerating visualizations...")

        bar_path = os.path.join(args.output_dir, 'bar.png')
        success1 = plot_bar_importance(
            importance,
            f"Top {args.top_k} Keywords by Attention",
            top_k=args.top_k,
            save_path=bar_path
        )

        wc_path = os.path.join(args.output_dir, 'wordcloud.png')
        success2 = plot_wordcloud(
            importance,
            "Attention Word Cloud",
            save_path=wc_path
        )

        json_path = os.path.join(args.output_dir, 'importance.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            filtered_imp = filter_importance(importance)
            json.dump(filtered_imp, f, ensure_ascii=False, indent=2)
        print(f"Filtered raw data saved to: {json_path}")

        if success1 or success2:
            print(f"\n✅ Results saved to: {args.output_dir}")
        else:
            print("\n❌ No visualizations were generated")
    else:
        print("No importance data generated. Exiting.")

if __name__ == '__main__':
    main()