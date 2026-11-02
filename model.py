"""creates the MLP for grokking test"""

import torch
import torch.nn as nn


class MLP(nn.Module): # asks custom module to inherit nn.Modules built in pytorch machinery methods etc
    def __init__(self,p,d_embed,d_hidden):
        super().__init__() # forces it to run nn.Module's setup first (super calls theparent class of which my MLP is derived)
        self.deep_stack = nn.Sequential(nn.Embedding(p,d_embed),nn.Flatten()
            ,nn.Linear(2*d_embed,d_hidden),nn.ReLU())
        self.output_layer = nn.Linear(d_hidden,p)
        self.init_weights()
        
    def forward(self,X):
        hidden = self.deep_stack(X)
        return self.output_layer(hidden)
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)               
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

# self test block:
if __name__ == "__main__":
    model = MLP(p=53, d_embed=128, d_hidden=256)
    x = torch.randint(0, 53, (10, 2))       # 10 fake pairs of symbols
    out = model(x)

    assert out.shape == (10, 53), out.shape
    n_params = sum(w.numel() for w in model.parameters())
    assert n_params == 86_197, n_params

    print("All checks passed:", out.shape, n_params, "parameters")