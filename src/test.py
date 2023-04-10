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

device = 'cuda'

def prepare_gpt2_input(prompt, device):
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    indices = encode(prompt)
    x = (torch.tensor(indices, dtype=torch.long, device=device)[None, ...])
    return x, decode


def generate_gpt2(model, prompt, device, samples=2):
    model.eval()
    model.to(device)
    max_new_tokens = 50
    temperature = 0.9
    top_k = 200
    x, decode = prepare_gpt2_input(prompt, device)

    for k in range(samples):
        y = model.generate(x,
                           max_new_tokens,
                           temperature=temperature,
                           top_k=top_k)
        print(decode(y[0].tolist()))
        print('---------------')

prompt = """Human: You are an asshole! You are an idiot!
Assitant:"""
cfg = get_configs("gpt2-medium")

model = GPT.from_pretrained(cfg)
# model = GPT.from_checkpoint(cfg, ckpt_path="")

generate_gpt2(model, prompt, device, samples=10)

