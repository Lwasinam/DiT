import torch
import math
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()

        self.conv = nn.Conv2d(in_channels,embed_dim, kernel_size=patch_size, stride  = patch_size)
        self.flat = nn.Flatten(start_dim=2)
              
    
    def forward(self, x):
        # (B, C, H, W) -> (B, C, H/Patch_size, W/ Patch_size
        x = self.conv(x)

        # (B, C, H/Patch, W/Patch) -> (B, C, Number_of_patch (H/Patch_size * W/Patch_size))
        x = self.flat(x)

        # (B, C, Number_of_patch) -> (B, Number_of_patch, C) # channel is the embedding_dimension
        x = x.transpose(1,2)
        return x
def generate_1d_sin_cos_embed(grid, embed_dim):
    embed_dim = embed_dim // 2 
    div_term =  torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
    print(div_term.shape)
    grid = grid.flatten().unsqueeze(-1)
    # print(grid.shape)

    # pos_embed_sin = torch.zeros(grid[0], embed_dim )
    # pos_embed_cos = torch.zeros(grid[0], embed_dim )

   

    pos_embed_sin = torch.sin(grid * div_term)
    pos_embed_cos = torch.cos(grid * div_term)

    print(pos_embed_cos.shape)
    

    return torch.cat([pos_embed_sin, pos_embed_cos], dim=1)        
def positional_encoding(patch_size, embed_dim):
    grid_size = 32 // patch_size
    grid_y = torch.arange(grid_size) 
    grid_y = grid_y.repeat(grid_size,1)
    grid_x = grid_y.transpose(0,1)

    x  =  generate_1d_sin_cos_embed(grid_x, embed_dim)
    y = generate_1d_sin_cos_embed(grid_y, embed_dim)

    # print(x.shape)

    return torch.cat([x,y],dim=1)       
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.wq = nn.Linear(embed_dim, embed_dim)
        self.wk = nn.Linear(embed_dim, embed_dim)
        self.wv  = nn.Linear(embed_dim, embed_dim)
        self.ww = nn.Linear(embed_dim, embed_dim)

    def compute_attention(self, query, key,  value, ) :
        
        attention  = (query @ key.transpose(-2,-1))/math.sqrt(self.head_dim)
        attention = attention.softmax(dim=-1)  # (batch_size, num_heads, query_len, key_len)
        attention = attention @ value 
        return attention

     

    def forward(self, x):
        wQ = self.wq(x)
        wK = self.wk(x)
        wV = self.wv(x)
        
        
        wQ = wQ.view(wQ.shape[0], wQ.shape[1], self.num_heads, self.head_dim).transpose(1,2)
        wK = wK.view(wK.shape[0], wK.shape[1], self.num_heads, self.head_dim).transpose(1,2) 
        wV = wV.view(wV.shape[0], wV.shape[1], self.num_heads, self.head_dim).transpose(1,2)

        out = self.compute_attention(wQ, wK, wV)

        out = out.transpose(1, 2).contiguous().view(out.shape[0], out.shape[1], self.embed_dim)


        return self.ww(out) 





        



# def positional_encoding(patch_size, embed_dim):
#     grid_size = 32 // patch_size
#     grid_y = torch.arange(grid_size) 
#     grid_y = grid_y.repeat(grid_size,1)
#     grid_x = grid_y.transpose(0,1)

#     x  =  generate_1d_sin_cos_embed(grid_x, embed_dim)
#     y = generate_1d_sin_cos_embed(grid_y, embed_dim)

#     # print(x.shape)

#     return torch.cat([x,y],dim=1)










print(positional_encoding(16, 512).shape)
# # Testing
# p = PatchEmbedding(in_channels=3, embed_dim=768, patch_size=16)
# x = torch.randn(2, 3, 32, 32)  # [batch_size, in_channels, height, width]
# out = p(x)
# print(out.shape) 


        