import torch
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F

# Create the BertClassfier class
class BERT_LSTM(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, hidden_dimension=256):
        
        super().__init__()
        
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        
        embedding_dim = self.bert.config.to_dict()['hidden_size']
        
        self.LSTM = nn.LSTM(embedding_dim,hidden_dimension,bidirectional=True, batch_first=True)
               
        self.out = nn.Linear(hidden_dimension * 2, 3)
            
    def forward(self, input_ids, attention_mask=None):
        
       
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
    
        # Extract the last hidden state of the token `[CLS]` for classification task
        encoded_layers = outputs[0]

        #encoded_layers = encoded_layers.permute(1, 0, 2)
        
        enc_hiddens, (last_hidden, last_cell) = self.LSTM(encoded_layers)
        output_hidden = torch.cat((enc_hiddens[:,-1, :256],enc_hiddens[:,0, 256:]),dim=-1)
        output_hidden = F.dropout(output_hidden,0.2)
        
        logits = self.out(output_hidden)
        
        return logits