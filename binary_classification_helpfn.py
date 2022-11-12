import torch

def train_step(model: torch.nn.Module,
               dl: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               opt: torch.optim.Optimizer,
               dev: torch.device = "cpu"):
    
    # Initiate loss and accuracy count
    train_loss, train_acc = 0, 0
    
    # Model on taget device and in training mode
    model.to(dev)
    model.train()
    
    for batch, (X, y) in enumerate(dl):
        # Put data on target device
        X, y = X.to(dev), y.to(dev)
        
        # Forward pass
        y_pred = model(X).squeeze()
        
        # Loss and accuracy
        loss = loss_fn(y_pred, y)
        train_loss += loss
        y_prob = torch.round(torch.sigmoid(y_pred))
        train_acc += _accuracy_fn(y_prob, y)
        
        # Backpropagation
        opt.zero_grad()
        loss.backward()
        opt.step()
        
    # Normalise loss and accuracy
    train_loss /= len(dl)
    train_acc /= len(dl)
    print(f"\nTrain loss: {train_loss:.5f} | Train acc: {train_acc:.2f}%")


def val_step(model: torch.nn.Module,
              dl: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              opt: torch.optim.Optimizer,
              dev: torch.device = "cpu"):
    
    # Initiate loss and accuracy count
    val_loss, val_acc = 0, 0
    
    # Model on taget device and in eval mode
    model.to(dev)
    model.eval()
    
    # Inference mode
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dl):
            # Put data on target device
            X, y = X.to(dev), y.to(dev)
            
            # Forward pass
            y_pred = model(X).squeeze()
            
            # Loss and accuracy
            val_loss += loss_fn(y_pred, y)
            y_prob = torch.round(torch.sigmoid(y_pred))
            val_acc += _accuracy_fn(y_prob, y)
            
        # Normalise loss and accuracy
        val_loss /= len(dl)
        val_acc /= len(dl)
        print(f"\nValidation loss: {val_loss:.5f} | Validation acc: {val_acc:.2f}%")


def make_predictions(model: torch.nn.Module,
                     dl: torch.utils.data.DataLoader,
                     dev: torch.device = "cpu"):
    
    # Initiate predictions
    y_preds = []
    
    # Model on taget device and in eval mode
    model.to(dev)
    model.eval()
    
    # Inference mode
    with torch.inference_mode():
        for X, y in dl:
            # Put data on target device
            X, y = X.to(dev), y.to(dev)
            
            # Forward pass
            y_logit = model(X).squeeze()
            y_prob = torch.round(torch.sigmoid(y_logit))
            
            # Save prediction in list on CPU
            y_preds.append(y_prob.to("cpu"))
    
    # Turn list into tensor
    return torch.cat(y_preds)


# Help function
def _accuracy_fn(y_pred, y_true):
    return torch.eq(y_true, y_pred).sum().item() / len(y_pred) * 100
