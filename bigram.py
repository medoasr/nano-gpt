import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1337)

#hyperparam
block_size=8
batch_size=32
device='cuda' if torch.cuda.is_available() else 'cpu'
lr=1e-2
max_iter=3000
eval_iter=200
eval_intervals=300

#!wget  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt','r') as f:
    text=f.read()
chars=sorted(list(set(text)))
vocab_size=len(chars)
stoi={ch:i for i,ch in enumerate(chars)}
itos={i:ch for i,ch in enumerate(chars)}
encode=lambda s:[stoi[c] for c in s]
decode=lambda l:''.join([itos[i] for i in l])

data=torch.tensor(encode(text),dtype=torch.long)
n=int(.9*len(data))
train_data=data[:n]
val_data=data[n:]

def get_batch(split):
  data=train_data if split=='train' else val_data
  idx=torch.randint(len(data)-block_size,(batch_size,))
  x=torch.stack([data[i:i+block_size] for i in idx])
  y=torch.stack([data[i+1:i+block_size+1] for i in idx])
  x,y=x.to(device),y.to(device)
  return x,y
xb,yb=get_batch('train')
print(xb.shape,yb.shape)

@torch.no_grad()
def estimate_loss():
    out={}
    model.eval()
    for split in ['train','val']:
        losses=torch.zeros(eval_iter)
        for k in range(eval_iter):
            x,y=get_batch(split)
            logits,loss=model(x,y)
            losses[k]=loss.item()
        out[split]=losses.mean()
    model.train()
    return out



class BiagramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_table(idx)
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = BiagramModel(vocab_size)
model=model.to(device)
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)

for steps in range(max_iter):
    if steps%eval_intervals==0:
        losses=estimate_loss()
        print(f'step {steps}: train_loss {losses['train']}, val_loss {losses['val']}')
    xb,yb=get_batch('train')
    logits,loss=model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
context=torch.zeros(1,1,dtype=torch.long,device=device)
print(decode(model.generate(context,100)[0].tolist()))