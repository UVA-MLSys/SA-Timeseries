import torch
from torchmetrics import AUROC
pred = torch.tensor([0.2, 0.1, 0.15])
target = torch.tensor([1, 0, 1])
auroc = AUROC(task='binary')
print(target.dim(), target[0].dim())
print(auroc(pred, target), auroc(target, target))