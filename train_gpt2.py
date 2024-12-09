import os 
import time
import math
import signal
def sigtstp_handler(signum, frame):
    print("Ctrl+Z pressed. Use Ctrl+C to save and exit.")
signal.signal(signal.SIGTSTP, sigtstp_handler)

import torch
from torch import nn
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
if torch.cuda.is_available(): 
    torch.cuda.empty_cache() 
 
from model import GPT, GPTConfig
from Dataloader import DataLoaderShakeSpeare

seed = 1337
max_steps = 60000
lr_decay_iters = max_steps
lr_decay_iters = 10 # !! OVERWRITE
total_batch_size = 2**19 # around 0.5M 
B=16
T=1024

warmup_steps = 500
weight_decay = 0.1
lr0 = 6e-4 
max_lr = lr0 
min_lr = max_lr * 0.1

lr0 = 6e-5
max_lr = 6e-5
min_lr = 3e-5

val_freq = 100
val_loss_steps = 20
log_dir = 'logging'

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it +1) / warmup_steps
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it-warmup_steps) / (lr_decay_iters - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

ddp = int(os.environ.get('RANK', -1)) != -1 # is this ddp run?
if ddp:
    assert torch.cuda.is_available()
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
device_type = 'cuda' if 'cuda' in device else 'cpu'
assert total_batch_size % (B*T*ddp_world_size) == 0
grad_accum_steps = total_batch_size // (B*T*ddp_world_size)
if master_process:
    print(f'total desired batch size: {total_batch_size}')
    print(f'=> calculated gradiet accumulation steps: {grad_accum_steps}')

# -----------------------------------------------------------------------------
# import tiktoken
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

# -----------------------------------------------------------------------------

torch.manual_seed(seed)
if device != 'cpu': torch.cuda.manual_seed(seed)
torch.set_float32_matmul_precision('high')

# print(device)
# print('Hello I am gpu:', ddp_rank, ddp_local_rank, ddp_world_size)
# import sys; sys.exit(0)
train_loader = DataLoaderLite(B, T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
val_loader = DataLoaderLite(B, T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')
# train_loader = DataLoaderShakeSpeare(B, T, process_rank=ddp_rank, num_processes=ddp_world_size)
# val_loader = DataLoaderShakeSpeare(B, T, process_rank=ddp_rank, num_processes=ddp_world_size)

# model = GPT(GPTConfig())
checkpoint_path = 'training-2-plato_on_epoch_8000/model_08500.pt'
checkpoint = torch.load(checkpoint_path, map_location=device)
model = GPT(checkpoint['config'])
state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint['model'].items()}
model.load_state_dict(state_dict)

model.to(device)
model = torch.compile(model)
model.train()
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model
optimizer = raw_model.configure_optimizers(weight_decay, lr0, device)

# logging 
os.makedirs(log_dir, exist_ok=True)
log_train = os.path.join(log_dir, 'log_train.txt')
log_val = os.path.join(log_dir, 'log_val.txt')
with open(log_train, 'w') as f: pass
with open(log_val, 'w') as f: pass

# x, y = train_loader.next_batch()
try:
    for step in range(max_steps):
        t0 = time.time()
        last_step = step == (max_steps-1)
        
        # validation every once in a while.
        if step % val_freq == 0:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                val_loss_accum = 0
                for _ in range(val_loss_steps):
                    logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
                
            # if ddp:
            #     dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f'validation loss: {val_loss_accum.item():.4f}')
                with open(log_val, 'a') as f:
                    f.write(f'{step}, {val_loss_accum.item()}\n')
                if step > 0 and (step % 500 == 0 or last_step):
                    checkpoint_path = os.path.join(log_dir, f'model_{step:05d}.pt')
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'config': raw_model.config,
                        'step': step,
                        'val_loss': val_loss_accum.item()
                    }
                    torch.save(checkpoint, checkpoint_path)
                    
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            if ddp: model.require_backward_grad_sync = (micro_step == grad_accum_steps-1) # sync gradients of the last micro_step
            with torch.amp.autocast(device_type, torch.bfloat16):
                logits, loss = model(x, y)
                loss = loss / grad_accum_steps
            # if ddp:
            #     dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
            loss.backward()
            loss_accum += loss.detach()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        
        if device != 'cpu': torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0)
        tokens_processed = train_loader.B * train_loader.T
        token_per_sec = tokens_processed / dt * grad_accum_steps * ddp_world_size
        if master_process:
            print(f'step {step} | loss: {loss_accum.item():.4f} | lr: {lr:.2e} | norm: {norm:.4f} | dt: {dt:.2f}s | token/sec: {token_per_sec:.2f}')
            with open(log_train, 'a') as f:
                f.write(f'{step}, {loss_accum.item()}, {lr}, {norm}, {dt}, {token_per_sec}\n')
except KeyboardInterrupt:
    if master_process:
        raw_model = model.module if ddp else model
        checkpoint_path = os.path.join(log_dir, f'last.pt')
        checkpoint = {
            'model': raw_model.state_dict(),
            'config': raw_model.config,
            'step': step
        }
        torch.save(checkpoint, checkpoint_path)
finally:
    if ddp:
        destroy_process_group()
    if master_process:
        raw_model = model.module if ddp else model
        checkpoint_path = os.path.join(log_dir, f'last.pt')
        checkpoint = {
            'model': raw_model.state_dict(),
            'config': raw_model.config,
            'step': step
        }
        torch.save(checkpoint, checkpoint_path)