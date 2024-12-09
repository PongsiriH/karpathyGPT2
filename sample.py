import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from torch.nn import functional as F
from model import GPT
import tiktoken
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

checkpoint_path = 'logging/model_00500.pt'
checkpoint = torch.load(checkpoint_path, map_location=device)

model = GPT(checkpoint['config'])
state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint['model'].items()}
model.load_state_dict(state_dict)
model.to(device)
model.eval()

max_length = 20
enc = tiktoken.get_encoding('gpt2')
initial_text = """The universe is"""
num_response = 5
x = torch.tensor(enc.encode(initial_text), dtype=torch.long)
x = x.unsqueeze(0).repeat(num_response, 1)
x = x.to(device)

model.eval()
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)[0] # (B, T, C) 
        logits = logits[:, -1, :] / 1.0 # (B, C)
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 10, dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat([x, xcol], dim=1)

for token in x:
    print("Generated text: ", enc.decode(token.tolist()))
    print()