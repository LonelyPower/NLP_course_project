import sys
workplace_root = '/home/drew/Desktop/nlp_course_project/news_cls_recm'
sys.path.append(workplace_root)

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel

from model import Bert_BiLSTM, Bert_Naive
from evaluate import evaluate
from utils import CustomDataset, read_split_data, FocalBCEWithLogitLoss


def train(args):
    current_time = time.strftime('%Y%m%d%H%M%S', time.localtime())
    bert_tokenizer = BertTokenizer.from_pretrained(args.bert)

    train_data, train_label, val_data, val_label = read_split_data(args.data_path, args.sample_rate, args.seed)
    train_dataset = CustomDataset(train_data, train_label, bert_tokenizer, args)
    val_dataset = CustomDataset(val_data, val_label, bert_tokenizer, args)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    bert = BertModel.from_pretrained(args.bert).to(args.device)
    net = Bert_Naive(bert, args).to(args.device)
    criterion = FocalBCEWithLogitLoss().to(args.device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    max_acc = 0
    net.load_state_dict(torch.load(args.continue_training))
    for epoch in range(args.num_epochs):
        epoch_loss, epoch_train_acc = train_one_epoch(net, train_loader, optimizer, criterion, args)
        val_loss, val_acc = evaluate(net, val_loader, criterion, args)
        print(f"Epoch {epoch+1}/{args.num_epochs}, Loss: {epoch_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        if val_acc > max_acc:
            max_acc = val_acc
            torch.save(net.state_dict(), os.path.join(args.save_folder, f"{args.model_name}_{current_time}.pth"))


def train_one_epoch(net, train_loader, optimizer, criterion, args):
    net.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with tqdm(train_loader, desc="Training", unit="batch") as t:
        for batch in t:
            input_ids = batch[0]['input_ids'].to(args.device)
            attention_mask = batch[0]['attention_mask'].to(args.device)
            labels = batch[1].to(args.device)  

            optimizer.zero_grad()
            
            outputs = net(input_ids, attention_mask)
            if isinstance(criterion, FocalBCEWithLogitLoss):
                y = torch.zeros(outputs.shape)
                y[range(outputs.shape[0]), labels] = 1
                target = y.to(args.device)
            else:
                target = labels
            loss = criterion(outputs, target)
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            t.set_postfix(loss=loss.item(), train_acc=correct_predictions / total_samples)

    epoch_loss = total_loss / len(train_loader)
    epoch_train_acc = correct_predictions / total_samples

    return epoch_loss, epoch_train_acc
