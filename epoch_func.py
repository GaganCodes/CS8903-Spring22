"""
Add description of this file
"""

import torch

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Classes/functions specific to epoch/prediction steps
class ValueMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_loop(dataloader, model, loss_fn, optimizer, acc_fn=None, record=True, device='cpu'):
    num_batches = len(dataloader)
    train_loss = 0.0
    accuracy = 0.0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction and loss
        pred = model(X)

        loss = loss_fn(pred, y)
        train_loss += loss.item()

        if acc_fn:
            new_accuracy = acc_fn(pred, y).item()
            # Because we're tracking the max
            if accuracy < new_accuracy:
                accuracy = new_accuracy
    
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= num_batches
    if record:
        model.train_score_metric_1.append(train_loss)
        if acc_fn:
            model.train_score_metric_2.append(100*accuracy)

def test_loop(dataloader, model, loss_fn, acc_fn=None, record=True, mode='val', device='cpu'):
    num_batches = len(dataloader)
    test_loss = 0.0
    accuracy = 0.0

    # Not keeping track of gradient because we're only testing = forward pass only
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            pred = model(X)

            test_loss += loss_fn(pred, y).item()

            if acc_fn:
                new_accuracy = acc_fn(pred, y).item()
                # Because we're tracking the max
                if accuracy < new_accuracy:
                    accuracy = new_accuracy

    test_loss /= num_batches
    
    print(f"Test Error: \n Max Abs Err: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    if record and mode=='val':
        model.val_score_metric_1.append(test_loss)
        if acc_fn:
            model.val_score_metric_2.append(100*accuracy)
    if record and mode=='test':
        model.test_score_metric_1.append(test_loss)
        if acc_fn:
            model.test_score_metric_2.append(100*accuracy)

def check_conv(model, conv_criteria=0.001):
    # For stopping the epochs if loss isn't reducing by defined convergence criteria
    # Kept to 0.1% by default
    # Requires minimum 15-20 runs for robustness
    loss_current = model.val_score_metric_1[-1]
    loss_prev = model.val_score_metric_1[-10]

    return abs((loss_current-loss_prev)/loss_prev) <= conv_criteria

def check_early_stop(model):
    # For stopping the epochs if validation loss is increasing
    # For this implementation looking over 10 iterations
    loss_current = model.val_score_metric_1[-1]
    loss_prev = model.val_score_metric_1[-10]

    return loss_current > loss_prev
