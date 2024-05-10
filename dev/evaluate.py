import torch
from tqdm import tqdm
from utils import FocalBCEWithLogitLoss

def evaluate(net, val_loader, criterion, args):
    net.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        with tqdm(val_loader, desc="Evaluating", unit="batch") as t:
            for idx, batch in enumerate(t):
                input_ids = batch[0]['input_ids'].to(args.device)
                attention_mask = batch[0]['attention_mask'].to(args.device)
                labels = batch[1].to(args.device)  

                outputs = net(input_ids, attention_mask)
                if isinstance(criterion, FocalBCEWithLogitLoss):
                    y = torch.zeros(outputs.shape)
                    y[range(outputs.shape[0]), labels] = 1
                    target = y.to(args.device)
                else:
                    target = labels
                loss = criterion(outputs, target)                

                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

                if idx % 10 == 0:
                    t.set_postfix(loss=loss.item(), val_acc=correct_predictions / total_samples)

    epoch_loss = total_loss / len(val_loader)
    epoch_val_acc = correct_predictions / total_samples

    return epoch_loss, epoch_val_acc
