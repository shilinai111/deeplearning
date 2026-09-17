import torch
x = torch.arange(12).reshape(3,4)
print(x)

chunks = torch.chunk(x,chunks=2,dim=0)
print(len(chunks))
print(chunks[0])
print(chunks[1])