"""
安全修复 TSV 中可能被截断的股票代码
- 以原字段中的截断代码为基准
- 仅当文本中存在以该代码为前缀的更长 $CODE 或 LLM 返回匹配代码时才替换
- 否则保持原样
"""

import pandas as pd
import requests
import time
import re
import hashlib
from tqdm import tqdm
from collections import Counter

# ==================== 配置区域 ====================
INPUT_FILE = "/home/remance/文档/xbb/FinancialDataset/tsv/train.tsv"
OUTPUT_FILE = "/home/remance/文档/xbb/FinancialDataset/tsv/train_safe.tsv"

# API 配置（若不使用 LLM，可将 USE_LLM 设为 False）
USE_LLM = False
API_KEY = "sk-dd8f7e873129418b9524512102865ec0"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
MODEL_NAME = "qwen-max"
REQUEST_INTERVAL = 0.5
MAX_RETRIES = 3
ENABLE_CACHE = True
# =================================================

_cache = {}


def call_llm(prompt: str) -> str:
    if not USE_LLM:
        return ""
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "你是一个金融文本分析助手，专门从推文中识别主要讨论的股票代码。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0, "max_tokens": 20, "top_p": 0.95
    }
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(BASE_URL, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            result = resp.json()
            return result['choices'][0]['message']['content'].strip()
        except Exception as e:
            time.sleep(2 ** attempt)
    return ""


def build_llm_prompt(text: str) -> str:
    return f"""识别推文中主要讨论的股票代码（仅返回大写字母代码，不带$）。若无明确股票，返回UNKNOWN。
推文："{text}"
代码："""


def llm_extract_stock(text: str) -> str:
    if pd.isna(text) or not str(text).strip():
        return "UNKNOWN"
    text = str(text).strip()
    if ENABLE_CACHE:
        key = hashlib.md5(text.encode()).hexdigest()
        if key in _cache:
            return _cache[key]
    prompt = build_llm_prompt(text)
    result = call_llm(prompt)
    if result:
        result = result.upper()
        match = re.search(r'[A-Z]{1,5}', result)
        stock = match.group() if match else "UNKNOWN"
    else:
        stock = "UNKNOWN"
    if ENABLE_CACHE:
        _cache[key] = stock
    return stock


def extract_codes_from_text(text):
    """从文本中提取所有 $CODE（大写2-5字母）"""
    if pd.isna(text):
        return []
    return re.findall(r'\$([A-Z]{2,5})', str(text))


def find_completion(trunc_code, text):
    """
    查找是否存在以 trunc_code 为前缀的更长完整代码
    返回：(完整代码或None, 来源说明)
    """
    if not trunc_code or trunc_code == 'UNKNOWN':
        return None, 'invalid'

    # 1. 优先从文本中的 $CODE 查找
    codes = extract_codes_from_text(text)
    candidates = [c for c in codes if c.startswith(trunc_code) and len(c) > len(trunc_code)]
    if candidates:
        # 选择出现频率最高且最长的
        freq = Counter(candidates)
        max_count = max(freq.values())
        best = max([c for c, cnt in freq.items() if cnt == max_count], key=len)
        return best, 'text_match'

    # 2. 若启用LLM且文本中没有，尝试用LLM识别隐含代码
    if USE_LLM:
        llm_code = llm_extract_stock(text)
        if llm_code != 'UNKNOWN' and llm_code.startswith(trunc_code) and len(llm_code) > len(trunc_code):
            return llm_code, 'llm_match'

    return None, 'no_match'


def fix_ticker_safe(old_ticker, text):
    """
    安全修复 ticker 字段
    """
    if pd.isna(old_ticker) or not old_ticker.startswith('TICKER$'):
        return old_ticker

    trunc_code = old_ticker.replace('TICKER$', '')
    if trunc_code == 'UNKNOWN':
        return old_ticker

    completion, source = find_completion(trunc_code, text)
    if completion:
        return f'TICKER${completion}'
    else:
        return old_ticker


def main():
    print(f"读取 {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE, sep='\t', header=None,
                     names=['index', 'label', 'tweet_id', 'text', 'ticker'])

    print(f"处理 {len(df)} 条记录（USE_LLM={USE_LLM}）...")
    tqdm.pandas(desc="修复进度")
    df['ticker_fixed'] = df.progress_apply(
        lambda row: fix_ticker_safe(row['ticker'], row['text']), axis=1
    )

    changed = (df['ticker'] != df['ticker_fixed']).sum()
    print(f"修复了 {changed} 条记录的股票代码字段 ({changed / len(df) * 100:.1f}%)")

    df_output = df[['index', 'label', 'tweet_id', 'text', 'ticker_fixed']]
    df_output.to_csv(OUTPUT_FILE, sep='\t', index=False, header=False)
    print(f"已保存至 {OUTPUT_FILE}")


if __name__ == "__main__":
    main()