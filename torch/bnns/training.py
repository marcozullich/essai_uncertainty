import torch
import ...metrics as M
from ..utils import training as T

    
    

def test_bnn(model, dataloader, metric=M.accuracy_ensemble, device=None):
    device = T.get_device(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        metric_tracker = torch.zeros([len(dataloader)])
        nums = torch.zeros_like(metrics_tracker)
        for data, target in dataloader:
            data = data.to(device)
            target = target.to(device)
            y_pred = model(data)
            
            mtr = metric(y_pred.detach(), target)
            metric_tracker.append(mtr)
            nums.append(len(data))
        
    metric_tracker *= nums
            
    print(f"Bayesian test - Metric {metric_tracker.mean(metric_tracker.sum() / nums.sum())}")
        
            
            
            