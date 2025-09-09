import torch
from torch import nn


def calc_loss_batch(input_batch, target_batch, model, device):
  input_batch, target_batch = input_batch.to(device), target_batch.to(device)
  logits = model(input_batch)
  loss = torch.nn.functional.cross_entropy(logits, target_batch)
  return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
  total_loss = 0.
  if len(data_loader) == 0:
    return float("nan")
  elif num_batches is None:
    num_batches = len(data_loader)
  else:
    num_batches = min(num_batches, len(data_loader))

  for i, (input_batch, target_batch) in enumerate(data_loader):
    if i < num_batches:
      loss = calc_loss_batch(input_batch, target_batch, model, device)
      total_loss += loss.item()
    else:
      break
  return total_loss / num_batches


def train_simple(model, train_loader, valid_loader, optimizer, device, num_epochs, eval_freq, eval_iter):
  train_losses, valid_losses = [], []
  train_accs, valid_accs = [], []

  examples_seen, global_step = 0, -1
  for epoch in range(num_epochs):
    model.train()

    for input_batch, target_batch in train_loader:
      optimizer.zero_grad()
      loss = calc_loss_batch(input_batch, target_batch, model, device)
      loss.backward()
      optimizer.step()
      examples_seen += input_batch.shape[0]
      global_step += 1

      if global_step % eval_freq == 0:
        train_loss, valid_loss = evaluate_model(
            model, train_loader, valid_loader, device, eval_iter
        )
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        print(f"Epoch: {epoch}, (Step {global_step:06d}): "
              f"Train loss: {train_loss:.3f}, Valid loss: {valid_loss:.3f}")
    train_acc = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
    valid_acc = calc_accuracy_loader(valid_loader, model, device, num_batches=eval_iter)
    print(f"Training acc: {train_acc*100:.3f} % | ", end="")
    print(f"Validation acc: {valid_acc*100:.3f} %")
    train_accs.append(train_acc)
    valid_accs.append(valid_acc)
  return train_losses, valid_losses, train_accs, valid_accs, examples_seen


def evaluate_model(model, train_loader, valid_loader, device, eval_iter):
  model.eval()
  with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
    valid_loss = calc_loss_loader(valid_loader, model, device, num_batches=eval_iter)
  model.train()
  return train_loss, valid_loss


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
  model.eval()
  correct_predictions, num_examples = 0, 0

  if num_batches is None:
    num_batches = len(data_loader)

  else:
    num_batches = min(num_batches, len(data_loader))

  for i, (input_batch, target_batch) in enumerate(data_loader):
    if i < num_batches:
      input_batch, target_batch = input_batch.to(device), target_batch.to(device)

      with torch.no_grad():
        logits = model(input_batch)[:, -1, :]
      predicted_labels = torch.argmax(logits, dim=-1)

      num_examples += predicted_labels.shape[0]
      correct_predictions += (predicted_labels == target_batch).sum().item()

    else:
      break
  return correct_predictions / num_examples