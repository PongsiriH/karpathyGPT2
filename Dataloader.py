import torch
import tiktoken

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, do_print=False):
        self.B = B # batch_size
        self.T = T # sequence_length
        self.process_rank = process_rank
        self.num_processes = num_processes
        
        with open('input.txt', 'r') as f: 
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        self.current_position = 0 # state.
        if do_print:
            print(f"loaded {len(self.tokens)} tokens")
            print(f"1 epoch = {len(self.tokens) // (B*T)} batches" )

    def reset(self):
        self.current_position = self.B * self.T * self.process_rank
        
    def next_batch(self):
        B, T = self.B, self.T
        i = self.current_position
        buf = self.tokens[i: i+B*T+1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_position += B*T*self.num_processes
        if self.current_position + (B*T*self.num_processes+1) > len(self.tokens):
            self.current_position = B*T*self.process_rank
        return x, y

class DataLoaderShakeSpeare(DataLoaderLite):
    pass


if __name__ == '__main__':
    train_loader = DataLoaderLite(2, 128)
    x, y = train_loader.next_batch()
    print('x', x[0])
    print('y', y[0])