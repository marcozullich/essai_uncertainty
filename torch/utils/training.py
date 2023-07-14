import torch

def train_full_batch_GD(
    model:torch.nn.Module,
    data:torch.Tensor,
    target:torch.Tensor,
    loss_fn:torch.nn.Module,
    optimizer:torch.optim.Optimizer,
    num_iterations:int,
    print_each:int=50
):
  model.train()
  for i in range(num_iterations):
    y_pred = model(data)

    loss = loss_fn(y_pred, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if (i + 1) % print_each == 0:
      print(f"Iteration {i+1} - loss {loss}")

def train_SGD(
    model:torch.nn.Module,
    dataloader:torch.utils.data.DataLoader,
    loss_fn:torch.nn.Module,
    optimizer:torch.optim.Optimizer,
    num_epochs:int,
    print_each_epochs:int=1
):
  model.train()
  for epoch in range(num_epochs):

    for data, target in dataloader:
      y_pred = model(data)
      loss = loss_fn(y_pred, target)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

    if (epoch + 1) % print_each_epochs == 0:
      print(f"Iteration {epoch+1} - last ite loss {loss}")
      
def test(model, dataloader, loss_fn, device=None):
  if device is None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
  model.to(device)
  model.eval()
  with torch.no_grad():
    accumulated_loss = 0.0
    correct_items = 0
    n_data = 0

    for data, target in dataloader:
      data = data.to(device)
      target = target.to(device)
      y_pred = model(data)
      loss = loss_fn(y_pred, target)
      accumulated_loss += loss.item()
      correct_items += (y_pred.max(1).indices == target).sum()
      n_data += len(data)

    print(f"TEST - loss {accumulated_loss/n_data} - accuracy {correct_items/n_data}")