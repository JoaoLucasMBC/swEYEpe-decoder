from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
print(torch.cuda.is_available())


class LanguageContext:
    def __init__(self):
        self.model_name = "gpt2-large"  # You can change this to any other suitable model
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, is_decoder=True).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def compute_token_scores(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        next_token_logits = logits[0, -1, :]
        return next_token_logits

    def top_k_tokens(self, token_scores, k=3):
        top_k_idx = torch.topk(token_scores, k).indices
        top_scores = token_scores[top_k_idx].tolist()
        top_words = [self.tokenizer.decode([idx]) for idx in top_k_idx]
        return dict(zip(top_words, top_scores))

    def words_and_probs(self, prompt):
        token_scores = self.compute_token_scores(prompt)
        # Convert to probabilities
        probs = torch.nn.functional.softmax(token_scores, dim=-1)
        probs_map = self.top_k_tokens(probs, k=100)
        return probs_map

    def combine_probs(self, gaze_probs, language_probs, language_weight = 0.3):
        gaze_weight = 1-language_weight
        combined_dict = {}
        for elem in gaze_probs:
            combined_dict[elem[0]] = combined_dict.get(elem[0], 0) + elem[1] * gaze_weight

        l_probs_map = [(key.strip(), language_probs[key]) for key in language_probs]
        for elem in l_probs_map:
            if (combined_dict.get(elem[0], 0) != 0):
                combined_dict[elem[0]] = combined_dict.get(elem[0], 0) + elem[1] * language_weight
        combined_prediction = list(sorted(combined_dict.items(), key = lambda x: x[1], reverse=True))
        return combined_prediction[:3]