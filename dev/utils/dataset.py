from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import BertTokenizer

from utils import read_split_data


class CustomDataset(Dataset):
    def __init__(self, data_path, label, tokenizer, args):
        super(CustomDataset, self).__init__()
        self.data_path = data_path
        self.data_class = label
        self.tokenizer: BertTokenizer = tokenizer
        self.args = args
        self.input_seqs, self.tokens = self.gen_tokens()
    
    def __len__(self):
        return len(self.data_path)
    
    def __getitem__(self, idx):
        input_seq = self.input_seqs[idx]
        label = self.data_class[idx]
        return input_seq, label
    
    def gen_tokens(self):
        tokens, inputs = [], []
        print(f"Generating tokens of {self.data_path[0].split('/')[1]}_dataset...")
        for text in tqdm(self.data_path):
            with open(text, 'r', encoding='utf-8') as f:
                text = f.read()
            encodes = self.tokenizer(
                text=text,
                max_length=self.args.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            inputs.append(encodes)
            tokens.append(self.tokenizer.tokenize(text=text))
        return inputs, tokens

# Test Case
if __name__ == '__main__':
    import sys
    sys.path.append('/home/drew/Desktop/nlp_course_project/news_cls_recm')

    import argparse
    from transformers import BertModel

    from model import Bert_BiLSTM

    data_root = 'data'
    train_data_path, train_label, val_data_path, val_label = read_split_data(data_root, 0.1, 1145)
    tokenizer = BertTokenizer.from_pretrained('pretrained_weights/bert-base-chinese')

    dataset_parser = argparse.ArgumentParser()
    dataset_parser.add_argument('--max_length', type=int, default=512)
    dataset_args = dataset_parser.parse_args()

    train_dataset = CustomDataset(train_data_path, train_label, tokenizer, dataset_args)

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', type=int, default=768)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--num_classes', type=int, default=14)
    parser.add_argument('--bidirectional', action='store_true')
    args = parser.parse_args()

    bert = BertModel.from_pretrained('pretrained_weights/bert-base-chinese')
    model = Bert_BiLSTM(bert, args)
    sample = train_dataset[0]
    output = model(sample[0]['input_ids'], sample[0]['attention_mask'])