import torch
import essai_uncertainty.metrics as M
from ..utils import training as T  

def test_bnn(model, dataloader, metric="accuracy", device=None, get_confidence=False):
    device = T.get_device(device)
    model.to(device)
    model.eval()
    confidence_tensors = []
    metric_tracker = 0.0
    num_data = 0
    
    with torch.no_grad():
        
        for data, target in dataloader:
            data = data.to(device)
            target = target.to(device)
            y_pred = model(data)
            
            if metric == "accuracy":
                confidence, pred_class = y_pred.mean(0).softmax(1).max(1)
                if get_confidence:
                    confidence_tensors.append(confidence)
                metric_tracker += (pred_class == target).sum()
                num_data += len(data)
            else:
                raise NotImplementedError("Not implemented yet")
            
        
    if metric == "accuracy":
        metric_tracker /= num_data
            
    print(f"Bayesian test - {metric} {metric_tracker}")
    
    if get_confidence:
        return torch.cat(confidence_tensors)
        
            
            
            