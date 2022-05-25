import torch
import torch.nn as nn
from transformers import BertModel

# Create the BertClassfier class
class BERT_GRU(nn.Module):
    """Bert Model for Classification Tasks.
    """

    def __init__(self,
                 hidden_dim = 256,
                 output_dim = 3,
                 n_layers = 2,
                 bidirectional = True,
                 dropout = 0.25):
        
        super().__init__()
        
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        
        embedding_dim = self.bert.config.to_dict()['hidden_size']
        self.dropout = nn.Dropout(dropout)
        
        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers = n_layers,
                          bidirectional = bidirectional,
                          batch_first = True,
                          dropout = dropout)
        
        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

    def forward(self, input_ids, attention_mask=None):
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,attention_mask=attention_mask)
        
        # Extract the last hidden state of
        encoded_layers = outputs[0]
        
        _, hidden = self.rnn(encoded_layers)

        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
            
        logits = self.out(hidden)
        return logits