from gpt import GPT, GPTRewardModel, HFGPTRewardModel
from configs import get_configs
from llama import LLaMA, ModelArgs
from dataset import AnthropicHHRLHFDataset, DahoasRMStaticDataset, DahoasSFTStaticDataset, EYLSFTStaticDataset
import torch
import tiktoken
import click
from tokenizer import LLaMATokenizer
from trainers import RewardModelTrainer, SFTTrainer
import json
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"


device = 'cuda'

def chat(device='cuda'):
    # Init model
    cfg = get_configs("gpt2-medium/dropout")
    # cfg.hf_model = "/home/ubuntu/.cache/huggingface/hub/models--gpt2-medium/snapshots/425b0cc90498ac177aa51ba07be26fc2fea6af9d"
    # model = GPT.from_pretrained(cfg)
    # model = GPT.from_checkpoint(cfg, ckpt_path="/home/ubuntu/vuthede/minChatGPT/src/runs/sft_devu_gpt_sft_202304080803/sft_devu_gpt_sft_202304080803_step180000.pt")
    # ckpt = "/home/ubuntu/vuthede/minChatGPT/src/runs/sft_devu_gpt_sft_202304080803/sft_devu_gpt_sft_202304080803_step180000.pt"
    ckpt = "/home/ubuntu/vuthede/minChatGPT/src/runs/ppo_devu_gpt_ppo_202304100419/ppo_devu_gpt_ppo_202304100419_actor_step3000.pt"


    model = GPT.from_checkpoint(cfg, ckpt_path=ckpt)
    
    model.eval()
    model.to(device)

    # Init tokenizer
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    
    # Params
    max_new_tokens = 100
    temperature = 0.9
    top_k = 2

    prompt = ""
    while True:
        human_ask = input("Human:")
        if human_ask=="reset":
            prompt = ""
            continue
        human_ask = f"\nHuman: {str(human_ask)}"
        prompt += human_ask
        prompt = prompt[-1024:] 
        # print(f"Yo it is prompt so far############\n :{prompt}\n@@@@@@@@@@@@@@@@@@@@2")
        
        indices = encode(prompt)
        x = (torch.tensor(indices, dtype=torch.long, device=device)[None, ...])
        
        y = model.generate(x,
                           max_new_tokens,
                           temperature=temperature,
                           top_k=top_k)
        complete = decode(y[0].tolist())
        complete = complete.split("<|endoftext|>")[0]
        print(complete)
        prompt += f'\n{complete}'

chat()


    