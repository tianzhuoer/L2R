import math
import pandas as pd
import torch
from torch import nn

from config import DQN_HIDDEN_SIZE, BELIEF_WINDOW_SIZE

def masked_softmax(X, attn_mask):
    
    # Set up causal masking for attention
    if attn_mask is None:
        seq_len = X.shape[-1]

        rows = torch.arange(seq_len, device=X.device).unsqueeze(1)
        cols = torch.arange(seq_len, device=X.device).unsqueeze(0)
        mask = cols >= rows
        X = X.masked_fill(mask, float(-1e7))
        return nn.functional.softmax(X, dim=-1)
    else:
        return nn.functional.softmax(X, dim=-1)

class PositionalEncoding(nn.Module):
    
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)   


class StateEmbedding(nn.Module):
    
    def __init__(self, total_dim=16, depth_dim=1, belief_len=BELIEF_WINDOW_SIZE,  
                 embed_dim=DQN_HIDDEN_SIZE, fusion_type='gated'):
        super().__init__()
        self.depth_dim = depth_dim
        self.data_dim = total_dim - depth_dim
        self.per_step_dim = total_dim
        self.fusion_type = fusion_type  # 'gated', 'attention', 'simple'
        

        self.depth_encoder = nn.Sequential(
            nn.Linear(self.depth_dim*belief_len, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        

        self.data_encoder = nn.Sequential(
            nn.Linear(self.data_dim*belief_len, DQN_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(DQN_HIDDEN_SIZE, embed_dim)
        )
        

        if fusion_type == 'gated':

            self.gate_depth = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Sigmoid()
            )
            self.gate_data = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Sigmoid()
            )
            self.fusion_layer = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            )
            
        elif fusion_type == 'attention':

            self.cross_attn_depth = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
            self.cross_attn_data = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
            self.fusion_layer = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            )
            
        else:
            self.fusion_layer = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            )
        
    def forward(self, x):

        

        s = self.per_step_dim
        depth_features = x[..., ::s]
        data_indices = []
        total_features = x.shape[-1]
        for i in range(0, total_features, s):
            data_indices.extend(range(i + 1, min(i + s, total_features)))
        data_features = x[..., data_indices]
        

        depth_embed = self.depth_encoder(depth_features)  # (..., embed_dim)
        data_embed = self.data_encoder(data_features)     # (..., embed_dim)
        

        if self.fusion_type == 'gated':

            combined = torch.cat([depth_embed, data_embed], dim=-1)
            gate_d = self.gate_depth(combined)
            gate_t = self.gate_data(combined)

            gated_features = gate_d * depth_embed + gate_t * data_embed
            state_embed = self.fusion_layer(gated_features)
            
        elif self.fusion_type == 'attention':


            if depth_embed.dim() == 2:
                depth_embed_seq = depth_embed.unsqueeze(1)
                data_embed_seq = data_embed.unsqueeze(1)
            else:
                depth_embed_seq = depth_embed
                data_embed_seq = data_embed
            

            depth_attended, _ = self.cross_attn_depth(depth_embed_seq, data_embed_seq, data_embed_seq)

            data_attended, _ = self.cross_attn_data(data_embed_seq, depth_embed_seq, depth_embed_seq)
            
            if depth_embed.dim() == 2:
                depth_attended = depth_attended.squeeze(1)
                data_attended = data_attended.squeeze(1)
            
            combined = torch.cat([depth_attended, data_attended], dim=-1)
            state_embed = self.fusion_layer(combined)
            
        else:  # 'simple'

            combined = torch.cat([depth_embed, data_embed], dim=-1)
            state_embed = self.fusion_layer(combined)
        
        return state_embed


class DotProductAttention(nn.Module):
    
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)





    def forward(self, queries, keys, values, attn_mask=None):
        d = queries.shape[-1]

        scores = torch.bmm(queries, keys.transpose(1,2)) / (float(d) ** 0.5)
        self.attention_weights = masked_softmax(scores, attn_mask)
        attention_temp = self.dropout(self.attention_weights)
        return torch.bmm(attention_temp, values)

class MultiHeadAttention(nn.Module):
    
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, data_mask=None, attn_mask=None):

        





        # (batch_size*num_heads,seq_length, num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        
        if attn_mask is not None:

            # (batch_size, num_queries, num_keys) -> (batch_size*num_heads, num_queries, num_keys)
            attn_mask = torch.repeat_interleave(
                attn_mask, repeats=self.num_heads, dim=0)


        head_of_attention = self.attention(queries, keys, values, attn_mask)


        head_concat = transpose_output(head_of_attention, self.num_heads)
        return self.W_o(head_concat)

def transpose_qkv(X, num_heads):
    



    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)



    X = X.permute(0, 2, 1, 3)


    # num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class PositionWiseFFN(nn.Module):
    
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))
    

class AddNorm(nn.Module):
    
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)
    


class DecoderBlock(nn.Module):
    
    def __init__(self,tgt_size, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)



        self.multiattention1 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, attn_mask):

        if self.training:
            batch_size, num_steps, _ = X.shape


            # dec_valid_lens = torch.arange(
            #     1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        # else:
        #     dec_valid_lens = None

        # k = X
        # q = X
        # v = X

        X1=self.multiattention1(X, X, X, attn_mask)
        Y=self.addnorm1(X, X1)
        return self.addnorm3(Y, self.ffn(Y))

class Decoder(nn.Module):
    def __init__(self, tgt_size, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 num_layers, dropout, use_bias=False):
        super(Decoder, self).__init__()
        
        self.layers = nn.ModuleList([
            DecoderBlock(tgt_size, key_size, query_size, value_size, num_hiddens,
                         norm_shape, ffn_num_input, ffn_num_hiddens,
                         num_heads, dropout, use_bias)
            for _ in range(num_layers)
        ])


    def forward(self, tgt, attn_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, attn_mask)
        return tgt



# DQN with Attention
class AttentionDQN(nn.Module):
    def __init__(self, n_observations, n_actions, num_hiddens=64, num_heads=4, num_layers=2, dropout=0.1,
                 depth_range=[-250, 0], depth_feature_idx=0):
        super(AttentionDQN, self).__init__()
        

        self.n_observations = n_observations
        self.n_actions = n_actions
        self.num_hiddens = num_hiddens
        self.depth_feature_idx = depth_feature_idx
        self.depth_range = depth_range
        


        self.state_embedding = StateEmbedding(total_dim=n_observations, depth_dim=1, embed_dim=num_hiddens)

        self.positional_encoding = PositionalEncoding(num_hiddens, dropout)

        self.decoder = Decoder(tgt_size=n_observations,
            key_size=num_hiddens, query_size=num_hiddens, value_size=num_hiddens,
            num_hiddens=num_hiddens, norm_shape=num_hiddens,
            ffn_num_input=num_hiddens, ffn_num_hiddens=num_hiddens*2,
            num_heads=num_heads, num_layers=num_layers, dropout=dropout
        )
        


        self.output_projection = nn.Linear(num_hiddens, n_actions)

    def forward(self, x, data_mask=None, attn_mask=None):

        

        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch_size, seq_len=1, n_observations)
        
        batch_size, seq_len, n_observations = x.shape
    
        

        x_proj = self.state_embedding(x)  # (batch_size, seq_len, num_hiddens)

        x_encoded=self.positional_encoding(x_proj)

        model_output = self.decoder(x_encoded, attn_mask=None)


        q_values = self.output_projection(model_output)  # (batch_size, seq_len, n_actions)
        




        # if seq_len == 1:
        #     q_values = q_values.squeeze(1)  # (batch_size, n_actions)
        # else:

        #     if attn_mask is not None:

        #         last_valid_indices = data_mask.sum(dim=1) - 1  # (batch_size,)
        #         q_values = q_values[torch.arange(batch_size), last_valid_indices]
        #     # else:

        q_values = q_values.squeeze(1)
        return q_values
        return q_values
    
