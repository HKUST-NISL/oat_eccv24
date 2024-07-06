import torch
import torch.nn as nn
import torch.nn.functional as F
from positionalEncoding import *

sigma = 2 # try 1.5, 2, 2 is the best
# W1:Random; W2:Original; W3:Original with order 15
choice = 1
saved_name = './amazon_learned_random_PE.npy'
num_rows, num_columns = 14, 86
learning_rate = 0.01
num_iterations = 10000
tolerance = 1e-3


if choice == 2:
    enc = PositionalEncoding(num_columns, 0, 'original', 0.9, maxlen=num_rows).pos_embedding
    enc = enc.reshape(11, 86)
    W = nn.Parameter(enc)
elif choice == 1:
    W = nn.Parameter(torch.randn(num_rows, num_columns))
#elif choice == 3:
#    W = nn.Parameter(torch.from_numpy(getEnc('original', 0.9, 6, True)).float())
optimizer = torch.optim.Adam([W], lr=learning_rate)

def loss_function(new_a, a, b, target_sim):
    new_sim = getCosSim(new_a, b)
    return (new_sim ** 2 - target_sim) ** 2


def getCosSim(a, b):
    cos_sim = dot(a, b) / (norm(a) * norm(b))
    return cos_sim


def getEnc(choice, alpha, center, update=True, order=15):
    embed = 86
    totali = 11
    enc = PositionalEncoding(embed, 0, choice, alpha).pos_embedding
    x = enc[:totali, 0].numpy()
    ref = x[center]
    simMatrix = np.zeros((1, totali))
    enc_updated = np.zeros((11, 86))
    # W = np.random.normal(0, 1, size=(2, 50))
    for i in range(0, totali):
        emb = x[i]
        a = getCosSim(emb, ref)

        if update:
            # new version
            order = order
            target_sim = a ** order
            result = minimize(loss_function, emb, args=(emb, ref, target_sim))
            updated_emb = result.x
            a = getCosSim(updated_emb, ref)
        simMatrix[0, i] = a
        enc_updated[i] = updated_emb
    return enc_updated


def target_gaussian(i, j, sigma):
    if i == j:
        return torch.tensor(1.0)
    else:
        return torch.exp(-(torch.tensor(i - j, dtype=torch.float32) ** 2) / (2 * sigma ** 2))


# alpha: weight of mag_loss
def magnitude_similarity_loss(W, alpha=1.0):
    norms = W.norm(p=2, dim=1)
    mean_norm = norms.mean()
    loss = ((norms - mean_norm) ** 2).mean()
    return alpha * loss


for iteration in range(num_iterations):
    optimizer.zero_grad()
    total_loss = 0

    W_normalized = F.normalize(W, p=2, dim=1)

    for i in range(num_rows):
        for j in range(num_rows):
            sim = torch.matmul(W_normalized[i].unsqueeze(0), W_normalized[j].unsqueeze(1))
            target_sim = target_gaussian(i, j, sigma)
            loss = F.mse_loss(sim, target_sim.unsqueeze(0).unsqueeze(1))
            total_loss += loss

    magnitude_loss = magnitude_similarity_loss(W)
    total_loss += magnitude_loss

    total_loss = total_loss / (num_rows * (num_rows - 1))
    total_loss.backward()
    optimizer.step()

cosine_sim_matrix = torch.zeros((num_rows, num_rows), dtype=torch.float32)
for i in range(num_rows):
    for j in range(num_rows):
        cosine_sim_matrix[i, j] = F.cosine_similarity(W_normalized[i], W_normalized[j], dim=0)

validation_matrix = torch.zeros_like(cosine_sim_matrix, dtype=torch.bool)
for i in range(num_rows):
    for j in range(num_rows):
        target_sim = target_gaussian(i, j, sigma)
        validation_matrix[i, j] = torch.isclose(cosine_sim_matrix[i, j], target_sim, atol=tolerance)

pass_rate = validation_matrix.float().mean().item()
print(f'pass rate: {pass_rate:.2%}')
print('cos_sim_Matrix:', cosine_sim_matrix)
print('Val_Matrix:', validation_matrix)

# cheack Norm
norms = W.data.norm(p=2, dim=1)
mean_norm = norms.mean()
norm_variance = norms.var()

print("Mean norm:", mean_norm.item())
print("Variance of norms:", norm_variance.item())
print("Norms of each row:", norms)

W = W.detach().cpu().numpy()
'''if choice == 1:
    np.save('learned_PE_random_20_'+str(sigma)+'.npy', W)
elif choice == 2:
    np.save('learned_PE_init_'+str(sigma)+'.npy', W)'''
np.save(saved_name, W)
