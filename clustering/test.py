from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np


model_name = "gpt2-large"  # You can change this to any other suitable model
model = AutoModelForCausalLM.from_pretrained(model_name, is_decoder=True).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name)

def compute_token_scores(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    next_token_logits = logits[0, -1, :]
    return next_token_logits

def top_k_tokens(tokenizer, token_scores, k=3):
    top_k_idx = torch.topk(token_scores, k).indices
    top_scores = token_scores[top_k_idx].tolist()
    top_words = [tokenizer.decode([idx]) for idx in top_k_idx]
    return dict(zip(top_words, top_scores))

# prompt = "The leaf drops to the"

# token_scores = compute_token_scores(model, tokenizer, prompt)
# # score_map = top_k_tokens(tokenizer, token_scores)
# # print(score_map)

# # Convert to probabilities
# probs = torch.nn.functional.softmax(token_scores, dim=-1)
# probs_map = top_k_tokens(tokenizer, probs, k=100)
# print(probs_map)