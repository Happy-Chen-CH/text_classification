# 导入工具包
import torch
import numpy as np
from train_eval import test
from importlib import import_module
import argparse
from utils import build_iterator
from utils import build_dataset

# 命令行参数解析
parser = argparse.ArgumentParser(description="Chinese Text Classification")
parser.add_argument("--model", type=str, default="bert", help="choose a model: bert")
args = parser.parse_args()

if __name__ == "__main__":
    if args.model == "bert":
        # 指定模型类型为bert
        model_name = "bert"
        x = import_module("models." + model_name)
        config = x.Config()
        # 设置随机种子
        np.random.seed(1)
        torch.manual_seed(1)
        # MPS 不支持 cuda.manual_seed_all 和 cudnn 设置
        if config.device.type != 'mps':
            torch.cuda.manual_seed_all(1)
            torch.backends.cudnn.deterministic = True
        # 数据迭代器的预处理和生成
        print("Loading data for Bert Model...")
        train_data, dev_data, test_data = build_dataset(config)
        train_iter = build_iterator(train_data, config)
        dev_iter = build_iterator(dev_data, config)
        test_iter = build_iterator(test_data, config)
        # 实例化模型并加载参数
        model = x.Model(config)
        print(model)

        model.load_state_dict(torch.load(config.save_path,map_location='cpu'),strict=False)
        # 量化BERT模型
        # 在 Apple Silicon Mac 上，FBGEMM 不可用，需要尝试 QNNPACK
        try:
            # 检查可用的量化引擎并设置
            if "qnnpack" in torch.backends.quantized.supported_engines:
                torch.backends.quantized.engine = "qnnpack"
                quantized_model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
                print("量化成功! 使用引擎: qnnpack")
            elif "fbgemm" in torch.backends.quantized.supported_engines:
                torch.backends.quantized.engine = "fbgemm"
                quantized_model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
                print("量化成功! 使用引擎: fbgemm")
            else:
                raise RuntimeError("没有可用的量化引擎")
        except Exception as e:
            print(f"量化失败: {e}")
            print("跳过量化，使用原始模型进行测试...")
            quantized_model = model
        print(quantized_model)
        #测试模型性能
        test(config, quantized_model, test_iter)
        #保存量化后的模型
        torch.save(quantized_model, config.save_path2)
