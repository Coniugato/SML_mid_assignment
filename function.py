import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from scipy.stats import norm


N_q=5000
N_p=100
mu_q=0
v_q=5
mu_p=7
v_p=1

class DistribDataset(Dataset):
    def __init__(self, data_mean, data_sdv, noise_sdv, function, n_data):
        super().__init__()
        self.data_mean=data_mean
        self.data_sdv=data_sdv
        self.noise_sdv=noise_sdv
        self.function=function
        self.n_data=n_data

    def __getitem__(self, index):
        x = np.random.normal(self.data_mean, self.data_sdv)
        y = np.random.normal(self.function(x), self.noise_sdv)
        return x, y

    def __len__(self):
        return self.n_data



q_dataset = DistribDataset(mu_q,v_q,1, lambda x: x**2+50*np.exp(-(x-mu_p)**2),N_q)
p_dataset = DistribDataset(mu_p,v_p,1, lambda x: x**2+50*np.exp(-(x-mu_p)**2),N_p)

q_dataloader = DataLoader(
    dataset=q_dataset,
    batch_size=N_q,
    shuffle=True
)

p_dataloader = DataLoader(
    dataset=p_dataset,
    batch_size=N_p,
    shuffle=True
)

def dataloadep_to_array(dataloader):
    xs=[]
    ys=[]
    for x,y in dataloader:
        xs.append(x)
        ys.append(y)
    return xs, ys

lams=[1e-2, 1e-1, 1, 1e+1, 1e+2]
sigmas=[1e-2, 1e-1, 1, 1e+1, 1e+2]

for x,y in q_dataloader:
    q_xs, q_ys = x, y
for x,y in p_dataloader:
    p_xs, p_ys = x, y



def torchize(x):
    if type(x)==np.ndarray:
        return torch.from_numpy(x)
    else:
        return x

def calc_r_model(p_xs, q_xs, lam, sigma):
    if len(q_xs.shape)<=1:
        calc_norm2=lambda x : x**2
    else:
        calc_norm2=lambda x : torch.norm(x,dim=-1)**2
    p_part_kernel=np.exp(-calc_norm2(np.repeat(p_xs.unsqueeze(1), len(q_xs),axis=1)-q_xs)/(2*sigma**2))
    G=np.matmul(p_part_kernel,p_part_kernel.T)/len(q_xs)
    h=np.exp(-calc_norm2(np.repeat(p_xs.unsqueeze(1), len(p_xs),axis=1)-p_xs)/(2*sigma**2)).mean(axis=1)
    alpha=np.linalg.solve(G+lam*np.eye(len(p_xs)),h)
    r_model=lambda x: torch.matmul(torchize(alpha), torch.exp(-calc_norm2(torch.repeat_interleave(torchize(x).unsqueeze(1), len(p_xs),dim=1)-p_xs)/(2*sigma**2)).T)
    return r_model

def validate_r_model(p_xs, q_xs, r_model):
    return (r_model(q_xs)**2).mean()/2-r_model(p_xs).mean()


n_split=5
r_models=[]
vals=[]
for lam in tqdm.tqdm(lams):
    for sigma in sigmas:
        val=0
        for idx in range(n_split):
            n_p=len(p_xs)
            p_interval=int(n_p/n_split)
            p_from=idx*p_interval
            p_to=n_p if idx==n_split-1 else (idx+1)*p_interval
            train_p_xs=torch.cat([p_xs[:p_from],p_xs[p_to:]])
            valid_p_xs=p_xs[p_from:p_to]

            n_q=len(q_xs)
            q_interval=int(n_q/n_split)
            q_from=idx*q_interval
            q_to=n_q if idx==n_split-1 else (idx+1)*q_interval
            train_q_xs=torch.cat([q_xs[:q_from],q_xs[q_to:]])
            valid_q_xs=q_xs[q_from:q_to]

            r_model=calc_r_model(train_p_xs, train_q_xs, lam, sigma)
            val+=validate_r_model(valid_p_xs, valid_q_xs, r_model)
        r_models.append(r_model)
        vals.append(val)
best_r_model=r_models[np.array(vals).argmin()]

data_distrib=lambda x: norm.pdf(x,mu_p,v_p)/norm.pdf(x,mu_q,v_q)

xgrid=np.linspace(-20,20,1000)
yideal=data_distrib(xgrid)
yestim=best_r_model(xgrid)
plt.plot(xgrid, yideal, label="theoretical")
plt.plot(xgrid, yestim, label="estimate")
plt.legend()
plt.title("density ratio")
plt.savefig("ratio.png")
plt.show()

import torch.nn as nn

class SimplpeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.li1 = nn.Linear(1,50)
        self.li2 = nn.Linear(50, 50)
        self.li3 = nn.Linear(50, 50)
        self.li4 = nn.Linear(50, 1)
        self.activ1 = nn.Sigmoid()
        self.activ2 = nn.Sigmoid()
        self.activ3 = nn.Sigmoid()

    def forward(self, x):
        x = x.float()
        x = self.li1(x)
        x = self.activ1(x)
        x = self.li2(x)
        x = self.activ2(x)
        x = self.li3(x)
        x = self.activ3(x)
        x = self.li4(x)
        return x
    
    def __call__(self, x):
        return self.forward(x)
    


def criteria(x,y):
    # MSE
    return (torch.norm(x-y,dim=1)**2).mean()

def shifted_criteria(y,t, x, r_model):
    # tが正解ラベル
    return (r_model(x)*(torch.norm(y-t,dim=1)**2)).mean()

def make1dto2d(x):
    if(len(x.shape)==1): return x.unsqueeze(1)
    else: return x

for x,y in p_dataloader:
    pv_xs, pv_ys = x, y

p_xs, p_ys, q_xs, q_ys, pv_xs, pv_ys = map(make1dto2d, [p_xs, p_ys, q_xs, q_ys, pv_xs, pv_ys])



model_shifted=SimplpeNet()
model_naive=SimplpeNet()


n_epochs=800

opt1=torch.optim.Adam(model_shifted.parameters(), lr=0.01)
opt2=torch.optim.Adam(model_naive.parameters(), lr=0.01)


train_losses_shifted=[]
train_losses_naive=[]
valid_losses_shifted=[]
valid_losses_naive=[]



for epoch in range(n_epochs):
    print("epoch "+str(epoch+1)+": ")
    shifted_loss=shifted_criteria(model_shifted(q_xs), q_ys, q_xs, r_model)
    loss=criteria(model_shifted(q_xs), q_ys)
    train_loss_shifted=loss.item()
    opt1.zero_grad()
    shifted_loss.backward()
    opt1.step()

    loss=criteria(model_shifted(pv_xs), pv_ys)
    valid_loss_shifted=loss.item()

    loss=criteria(model_naive(q_xs), q_ys)
    train_loss_naive=loss.item()
    opt2.zero_grad()
    loss.backward()
    opt2.step()

    loss=criteria(model_naive(pv_xs), pv_ys)
    valid_loss_naive=loss.item()

    print("q_loss_shifted: ", str(train_loss_shifted),"p loss shifted: ", str(valid_loss_shifted) )
    print("q_loss_naive: ", str(train_loss_naive),"p loss naive: ", str(valid_loss_naive) )
    train_losses_shifted.append(train_loss_shifted)
    train_losses_naive.append(train_loss_naive)
    valid_losses_shifted.append(valid_loss_shifted)
    valid_losses_naive.append(valid_loss_naive)

plt.title("valid loss")
#plt.plot(range(n_epochs),train_losses_naive,  label="(naive) train loss p")
plt.plot(range(n_epochs),valid_losses_naive,  label="(naive) valid loss q")
#plt.plot(range(n_epochs),train_losses_shifted,  label="(shifted) train loss p")
plt.plot(range(n_epochs),valid_losses_shifted,  label="(shifted) valid loss q")
plt.legend()
plt.savefig("simple_problem")
plt.show()

plt.scatter(q_xs, q_ys, label="train distribution")
plt.scatter(pv_xs, pv_ys, label="valid distribution")
xgrid=np.linspace(-20,20,1000)
plt.plot(xgrid, model_naive(torch.from_numpy(xgrid).unsqueeze(1)).detach().numpy(), label="naive prediction", color="green")
plt.plot(xgrid, model_shifted(torch.from_numpy(xgrid).unsqueeze(1)).detach().numpy(), label="shifted prediction", color="red")
plt.legend()
plt.savefig("prediction.png")

plt.show()