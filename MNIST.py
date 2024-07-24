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
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),  # テンソルに変換
    transforms.Normalize((0.1307,), (0.3081,))  # 平均と標準偏差で正規化
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# DataLoaderの作成
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100000, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100000, shuffle=True)

prob=0.05
n_train_p=500
n_valid_p=500
n_train_q=1000
for x,y in train_dataloader:
    mask = (y<=0) | (torch.rand(y.shape)<=prob)
    x, y = x[mask], y[mask]
    x = x.reshape((x.shape[0],-1))
    y = torch.nn.functional.one_hot(y, num_classes=10).float()
q_xs, q_ys = x[:n_train_p], y[:n_train_p]

for x,y in train_dataloader:
    x = x.reshape((x.shape[0],-1))
p_xs = x[:n_train_p]

for x,y in train_dataloader:
    x = x.reshape((x.shape[0],-1))
    y = torch.nn.functional.one_hot(y, num_classes=10).float()
pv_xs, pv_ys = x[:n_valid_p], y[:n_valid_p]

for x,y in test_dataloader:
    x = x.reshape((x.shape[0],-1))
    y = torch.nn.functional.one_hot(y, num_classes=10).float()
t_xs, t_ys = x[:n_valid_p], y[:n_valid_p]




lams=[1e-2, 1e-1, 1, 1e+1, 1e+2]
sigmas=[1e-2, 1e-1, 1, 1e+1, 1e+2]



def torchize(x):
    if type(x)==np.ndarray:
        return torch.from_numpy(x).float()
    else:
        return x.float()

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

#data_distrib=lambda x: norm.pdf(x,mu_p,v_p)/norm.pdf(x,mu_q,v_q)

#xgrid=np.linspace(-20,20,1000)
#yideal=data_distrib(xgrid)
#yestim=best_r_model(xgrid)
#plt.plot(xgrid, yideal, label="ideal")
#plt.plot(xgrid, yestim, label="estimate")
#plt.legend()
#plt.show()

import torch.nn as nn

class SimplpeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.li1 = nn.Linear(784,200)
        self.li2 = nn.Linear(200,10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.float()
        x = self.li1(x)
        x = self.sigmoid(x)
        x = self.li2(x)
        return x
    
    def __call__(self, x):
        return self.forward(x)
    


cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")

def criteria(y,t):
    return cross_entropy_loss(y,t).mean()

def shifted_criteria(y,t, x, r_model):
    # tが正解ラベル
    return (r_model(x)*cross_entropy_loss(y,t)).mean()

def make1dto2d(x):
    if(len(x.shape)==1): return x.unsqueeze(1)
    else: return x

softmax=torch.nn.functional.softmax

def calc_acc(y,t):
    return (softmax(y).argmax(dim=-1)==t.argmax(dim=-1)).sum().item()/len(y)

p_xs, q_xs, q_ys, pv_xs, pv_ys, t_xs, t_ys = map(make1dto2d, [p_xs, q_xs, q_ys, pv_xs, pv_ys, t_xs, t_ys])



model_shifted=SimplpeNet()
model_naive=SimplpeNet()


n_epochs=100

opt1=torch.optim.Adam(model_shifted.parameters(), lr=0.01)
opt2=torch.optim.Adam(model_naive.parameters(), lr=0.01)


train_losses_shifted=[]
train_losses_naive=[]
valid_losses_shifted=[]
valid_losses_naive=[]

valid_accs_naive=[]
valid_accs_shifted=[]



for epoch in range(n_epochs):
    print("epoch "+str(epoch+1)+": ")
    shifted_loss=shifted_criteria(model_shifted(q_xs), q_ys, q_xs, r_model)
    loss=criteria(model_shifted(q_xs), q_ys)
    train_loss_shifted=loss.item()
    opt1.zero_grad()
    shifted_loss.backward()
    opt1.step()

    pv_ps=model_shifted(pv_xs)
    loss=criteria(pv_ps, pv_ys)
    valid_loss_shifted=loss.item()

    valid_acc_shifted=calc_acc(pv_ps, pv_ys)

    loss=criteria(model_naive(q_xs), q_ys)
    train_loss_naive=loss.item()
    opt2.zero_grad()
    loss.backward()
    opt2.step()

    pv_ps=model_naive(pv_xs)
    loss=criteria(pv_ps, pv_ys)
    valid_loss_naive=loss.item()

    valid_acc_naive=calc_acc(pv_ps, pv_ys)

    print("train q loss shifted: ", str(train_loss_shifted),"valid p loss shifted: ", str(valid_loss_shifted), "valid accuracy", str(valid_acc_shifted) )
    print("train q loss naive: ", str(train_loss_naive),"valid p loss naive: ", str(valid_loss_naive) , "valid accuracy", str(valid_acc_naive) )
    train_losses_shifted.append(train_loss_shifted)
    train_losses_naive.append(train_loss_naive)
    valid_losses_shifted.append(valid_loss_shifted)
    valid_losses_naive.append(valid_loss_naive)
    valid_accs_naive.append(valid_acc_naive)
    valid_accs_shifted.append(valid_acc_shifted)

print("shifted test acc:", calc_acc(model_shifted(t_xs),t_ys))
print("naive test acc:", calc_acc(model_naive(t_xs),t_ys))

plt.title("valid loss")
#plt.plot(range(n_epochs),train_losses_naive,  label="(naive) train loss q")
plt.plot(range(n_epochs),valid_losses_naive,  label="(naive) valid loss p")
#plt.plot(range(n_epochs),train_losses_shifted,  label="(shifted) train loss q")
plt.plot(range(n_epochs),valid_losses_shifted,  label="(shifted) valid loss p")
plt.legend()
plt.savefig("valid_loss")
plt.show()

plt.title("valid acc")
plt.plot(range(n_epochs),valid_accs_naive,  label="(naive) valid acc p")
#plt.plot(range(n_epochs),train_losses_shifted,  label="(shifted) train loss p")
plt.plot(range(n_epochs),valid_accs_shifted,  label="(shifted) valid acc p")
plt.legend()
plt.savefig("valid_acc")
plt.show()
