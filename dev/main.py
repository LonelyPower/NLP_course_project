import os
import argparse

from train import train

def get_args():
    parser = argparse.ArgumentParser(description='Build Up the Model and Train/Test')
    parser.add_argument('--data_path', type=str, default='data', help='Path to the data')
    parser.add_argument('--mode', type=str, default='train', help='Mode')
    parser.add_argument('--bert', type=str, default='pretrained_weights/bert-base-chinese', help='Path to the data')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum length of the sequence')
    parser.add_argument('--num_classes', type=int, default=14, help='Number of classes')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--input_size', type=int, default=768, help='Input size')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size')
    parser.add_argument('--num_layers', type=int, default=8, help='Number of layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of heads')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--bidirectional', action='store_true', help='Whether to use bidirectional LSTM')    
    parser.add_argument('--save_folder', type=str, default='weights', help='Path to save the model')
    parser.add_argument('--model_name', type=str, default='bert_naive', help='Name of the model')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--sample_rate', type=int, default=0.01, help='Number of samples to use')
    parser.add_argument('--seed', type=int, default=114514, help='Seed for reproducibility')
    parser.add_argument('--continue_training', type=str, default='weights/bert_naive_20240501161942.pth', help='Continue training from a checkpoint')
    return parser.parse_args()


def main(args):
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        pass
    else:
        raise ValueError('Mode must be train or test')


if __name__ == '__main__':
    args = get_args()
    main(args)