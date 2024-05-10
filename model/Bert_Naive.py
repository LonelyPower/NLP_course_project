import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel


class Bert_Naive(nn.Module):
    def __init__(self, bert, args):
        super(Bert_Naive, self).__init__()
        self.args = args
        self.bert: BertModel = bert
        for param in self.bert.parameters():
            param.requires_grad = False
        
        self.ffn_neuron = self.args.input_size 
        self.mh_attn = nn.MultiheadAttention(
            embed_dim=self.ffn_neuron,  
            num_heads=args.num_heads,
            dropout=args.dropout_rate
        )
        self.dense = nn.Sequential(
            nn.Linear(self.ffn_neuron, 2 * self.ffn_neuron),  
            nn.ReLU(),
            nn.Dropout(args.dropout_rate),
            nn.Linear(2 * self.ffn_neuron, self.args.num_classes)
        )

    def forward(self, input_ids, attention_mask):
        bert_outputs = self.bert(input_ids=input_ids.squeeze(1), attention_mask=attention_mask.squeeze(1))
        pooled_output = bert_outputs['pooler_output'] 
        attn_output, _ = self.mh_attn(pooled_output, pooled_output, pooled_output)
        output = self.dense(attn_output)
        return output


# Test Case
if __name__ == '__main__':
    import argparse
    
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
    model = Bert_Naive(bert, args)
    input_ids = torch.randint(0, 10000, (1, 128))
    attention_mask = torch.ones(1, 128)
    output = model(input_ids, attention_mask)