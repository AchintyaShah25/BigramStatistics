import torch
import numpy as np
import matplotlib.pyplot as plt
words = open("names.txt", "r").read().splitlines()
b = {}

for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1

N = torch.zeros((27, 27), dtype=torch.int32)
chars = sorted(list(set("".join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
itos = {i: s for s, i in stoi.items()}
stoi["."] = 0
itos[0] = "."

for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

plt.figure(figsize=(27, 27))
plt.imshow(N, cmap='Pastel1')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color="black",
                 fontsize='xx-large', fontweight='bold')
        plt.text(j, i, N[i, j].item(), ha="center", va="top",
                 color="black", fontsize='xx-large', fontweight='book')
plt.axis('off')

P = (N + 1).float()
P /= P.sum(1, keepdim=True)
n = 0

log_likelihood = 0.0

for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        n += 1
        prob = P[ix1, ix2]
        log_prob = torch.log(prob)
        log_likelihood += log_prob

nll = - log_likelihood
print(f"log_likelihood: {log_likelihood/n}")
print(f"negative_log_likelihood: {nll/n}")

for i in range(20):
    ix = 0
    out = []
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True).item()
        xyz = np.array(p)
        out.append(itos[ix])
        if ix == 0:
            break
    print("".join(out))
