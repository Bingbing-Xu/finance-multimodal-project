import os
import torch
from config import Config, _MODEL_CLASSES
from data_utils.text_dataset import StockKnow
from data_utils.data_loader import get_plus_data_loader, get_data_loader
from models.BERT_cls import BERT_cls
from models.BERT_mlm import BERT_mlm
from framework import MLM_plus_framework  # 根据您的 model_type 选择对应框架类
from framework import CLS_framework
from framework import MLM_framework

def evaluate_test(checkpoint_path, config_path):
    """
    使用 framework.test 方法评估模型在测试集上的表现
    """
    # 1. 加载配置
    config = Config(config_path)

    # 根据 encoder_type 设置正确的预训练模型路径
    paths = {
        'bert': config.bert_path,
        'roberta': config.roberta_path,
        'finbert': config.finbert_path
    }
    config.bert_path = paths[config.encoder_type]
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.few_shot = None  # 确保不使用 few-shot

    # 2. 获取 tokenizer 和 processor
    encoder = _MODEL_CLASSES[config.encoder_type]
    tokenizer = encoder['tokenizer'].from_pretrained(config.bert_path)
    processor = StockKnow(config, config.root_dir, tokenizer)

    # 3. 加载测试数据
    _, _, test = processor.load_dataset()
    test_loader = get_plus_data_loader(config, test, batch_size=config.batch_size, shuffle=False)
    #cls
    #test_loader = get_data_loader(config, test, batch_size=config.batch_size, shuffle=False)
    #MLM
    #test_loader = get_data_loader(config, test, batch_size=config.batch_size, shuffle=False)

    # 4. 构建模型
    id2name = [
        "Negative Adverse Unfavorable Pessimistic Hostile Critical Dismal Gloomy Detrimental Defeatist Damaging",
        "Neutral Impartial Unbiased Objective Uninvolved Indifferent Balanced Nonpartisan Disinterested Equitable Fair-minded",
        "Positive Optimistic Favorable Encouraging Upbeat Good Constructive Affirmative Bright Promising Supportive"
    ]
    model = BERT_mlm(
        config,
        rels_num=3,
        device=config.device,
        id2name=id2name,
        encoder=encoder['encoder'],
        tokenizer=tokenizer,
        init_by_cls=None
    ).to(config.device)
    #cls
    #model = BERT_cls(config, num_label=3, model=encoder['model']).to(config.device)

    # 5. 加载最佳模型权重
    state_dict = torch.load(checkpoint_path, map_location=config.device)
    model.load_state_dict(state_dict)


    # 6. 创建框架对象（根据您的 model_type 选择，这里假设使用 mlmp）
    #    如果训练时使用了其他框架，请相应调整
    framework = MLM_plus_framework(config)

    #framework = MLM_framework(config)

    #cls
    #framework = CLS_framework(config)
    # 7. 使用框架的 test 方法进行评估（已手动加载权重，ckpt 参数传 None）
    test_acc, test_macf1, test_wf1 = framework.test(config, model, test_loader, ckpt=None)



    # 8. 打印最终结果
    print(f"\n{'=' * 50}")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Test Macro-F1: {test_macf1 * 100:.2f}%")
    print(f"Test Weighted-F1: {test_wf1 * 100:.2f}%")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    # 注意：请根据实际情况修改以下路径
    checkpoint = r"D:\GraduateProject\ArkMSA\save_results\Stock_Twitter_bert_mlmp_ARK_best.pt"
    config_file = r"D:\GraduateProject\ArkMSA\config_dir\StockKnow.ini"
    evaluate_test(checkpoint, config_file)