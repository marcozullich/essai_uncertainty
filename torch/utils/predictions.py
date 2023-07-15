def plot_prediction(image, model, normalize=False):
  if normalize:
    image = image / 255
  model_output = model(image.unsqueeze(0))
  model_output_softmax = nn.functional.softmax(model_output, dim=1)
  _=plt.bar(x=np.arange(0,10), height=model_output_softmax.squeeze().detach().numpy())
  _=plt.xticks(np.arange(0,10))
  
def get_confidence_deterministic(y_pred):
  return y_pred.softmax(1).max(1).values

def get_confidence_on_dataset(model, dataloader):
  model.eval()
  confidence_tensors = []
  with torch.no_grad():
    for data, target in dataloader:
      y_pred = model(data)
      confidence = get_confidence_deterministic(y_pred)
      confidence_tensors.append(confidence)
  return torch.cat(confidence_tensors)