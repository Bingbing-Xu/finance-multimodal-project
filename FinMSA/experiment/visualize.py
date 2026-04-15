import json
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ========== 字体设置 ==========
plt.rcParams['font.sans-serif'] = ['SimHei', 'Times New Roman']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# ========== 典型情感词库（仅强烈情感词） ==========
POSITIVE_TYPICAL = {
    'surged', 'rose', 'success', 'upgraded', 'shining', 'improving', 'amazing',
    'rally', 'breakout', 'soar', 'strength', 'win', 'gain', 'bullish','nicely'
}

NEGATIVE_TYPICAL = {
    'plunged', 'dropping', 'awful', 'halted', 'struggles', 'bearish', 'crash',
    'panic', 'decline', 'drop', 'loss', 'weak','doubted'
}

# 颜色
COLOR_POSITIVE = '#4CAF50'   # 翠绿
COLOR_NEGATIVE = '#FF6B6B'   # 珊瑚红
COLOR_NEUTRAL  = '#9E9E9E'   # 暖灰

def get_polarity(token):
    token_lower = token.lower()
    if token_lower in POSITIVE_TYPICAL:
        return 'positive'
    elif token_lower in NEGATIVE_TYPICAL:
        return 'negative'
    else:
        return 'neutral'

def load_importance(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_single_model(importance, model_name, top_k=10, save_path=None):
    items = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_k]
    tokens = [t for t, _ in items]
    weights = [w for _, w in items]

    colors = [COLOR_NEUTRAL] * len(tokens)   # 默认全灰
    for i, token in enumerate(tokens):
        pol = get_polarity(token)
        if pol == 'positive':
            colors[i] = COLOR_POSITIVE
        elif pol == 'negative':
            colors[i] = COLOR_NEGATIVE

    plt.figure(figsize=(10, 6))
    bars = plt.barh(tokens, weights, color=colors, edgecolor='none')
    plt.xlabel('平均注意力权重')
    plt.title(f'关键词注意力排名 (Top {top_k})\n{model_name}')
    plt.gca().invert_yaxis()

    # 数值标签
    for bar, w in zip(bars, weights):
        plt.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                 f'{w:.4f}', va='center', fontsize=9)

    # 图例（右下角）
    legend_elements = [
        Patch(facecolor=COLOR_POSITIVE, edgecolor='none', label='正面'),
        Patch(facecolor=COLOR_NEGATIVE, edgecolor='none', label='负面'),
        Patch(facecolor=COLOR_NEUTRAL,  edgecolor='none', label='中性')
    ]
    plt.legend(handles=legend_elements, loc='lower right', title='情感极性')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存 {model_name} 图表到: {save_path}")
    plt.show()

def main():
    base_json = r"D:\GraduateProject\ArkMSA\experiment3_results\base_attention\importance.json"
    full_json = r"D:\GraduateProject\ArkMSA\experiment3_results\full_attention\importance.json"
    output_dir = r"D:\GraduateProject\ArkMSA\experiment3_results\comparison"
    os.makedirs(output_dir, exist_ok=True)

    print("正在加载基础模型数据...")
    imp_base = load_importance(base_json)

    print("正在加载增强模型数据...")
    imp_full = load_importance(full_json)

    plot_single_model(imp_base, "仅文本+描述模型", top_k=10,
                      save_path=os.path.join(output_dir, "base_top10.png"))
    plot_single_model(imp_full, "文本+描述+理由模型", top_k=10,
                      save_path=os.path.join(output_dir, "full_top10.png"))

if __name__ == '__main__':
    main()