from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import argparse
"""
使用该脚本，将lora的权重合并大base model中
"""


def merge_lora_to_base_model(model_name_or_path, adapter_name_or_path, save_path):
# def merge_lora_to_base_model():
    # model_name_or_path = '/hujinwu/LLM_Assemble/pretrain_model/models--Qwen--Qwen1.5-32B-Chat/snapshots/0b1785d88bbe93aa90a8a19da8af78eccbf010a6'
    # adapter_name_or_path = '/hujinwu/code/robotchat/Firefly/output/wusuo/firefly-qwen1.5-32b-sft-qlora'
    # save_path = '/hujinwu/code/robotchat/Firefly/script/wusuo/firefly-qwen1.5-32b-sft-qlora'

    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_name_or_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if config.model_type == 'llama' else True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        # device_map='auto',
        device_map={'': 'cpu'}
    )
    model = PeftModel.from_pretrained(model, adapter_name_or_path, device_map={'': 'cpu'})
    model = model.merge_and_unload()

    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name_or_path", type=str, default='1')
    parser.add_argument("--adapter_name_or_path", type=str, default= '2')
    parser.add_argument("--save_path", type=str, default= '3')
    
    args = parser.parse_args()
    
    merge_lora_to_base_model(model_name_or_path=args.model_name_or_path,
                             adapter_name_or_path=args.adapter_name_or_path,
                             save_path=args.save_path)
    # merge_lora_to_base_model()



# deepspeed --include=localhost:4,5,6,7 train.py --train_args_file train_args/sft/qlora/qwen1.5-72b-sft-qlora.json