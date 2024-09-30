import torch
import math
import torch.nn as nn


class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int): 
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Matrix of size (seq_len, d_model) 
        pe = torch.zeros(seq_len, d_model)
        # Creating the pos vector of size (seq_len, 1)
        position = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # Adding the batch dimension

        self.register_buffer('pe', pe)

    def forward(self, x):
        print("Shape of x:", x.shape)
        print("Shape of positional encodings:", self.pe.shape)
        x =  x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) #multiplied
        self.gamma = nn.Parameter(torch.zeros(1)) #added

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * ((x - mean) / (std + self.eps)) + self.gamma 


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff, bias = True) #w1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model, bias = True) #w2

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.n_heads = n_heads

        assert d_model % n_heads == 0, "d_model is not divisible by n_heads"

        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, 1e-9)
        attention_scores = attention_scores.softmax(dim = -1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores


        

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.n_heads, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.n_heads, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.n_heads, self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.n_heads * self.d_k)

        return self.w_o(x)
    
class ResidualConnection(nn.Module):

    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):

    def __init__(self, features: int, layers : nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.add_norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.add_norm(x)
    

class DecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):

    def __init__(self, features: int, layers = nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.add_norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.add_norm(x)
        

class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)
    
class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        tgt = self.src_embed(tgt)
        tgt = self.src_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048):

    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    encoder_blocks = []

    for i in range(N):
        encoder_self_attention = MultiHeadAttention(d_model, h, dropout)
        encoder_feed_forward = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention, encoder_feed_forward, dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks = []

    for i in range(N):
        decoder_self_attention = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention = MultiHeadAttention(d_model, h, dropout)
        decoder_feed_forward = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention, decoder_cross_attention, decoder_feed_forward, dropout)
        decoder_blocks.append(decoder_block)
    
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(encoder,  decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    #initializing parameters

    for p in transformer.parameters():
        if p.dim() > 1: # weights have dimensions greater than 1 while dimensions of bias is one. Here we only initialize the weights and not the bias
            nn.init.xavier_uniform_(p)

    return transformer


'''
Why Xavier Uniform Initialization?
Control Variance: The goal of Xavier initialization is to keep the variance of the outputs of each neuron approximately the same across all layers. 
Without this, the variance can increase or decrease exponentially with the number of layers in a deep network, potentially leading to exploding or vanishing gradients. 
This helps in stabilizing the learning process in deep neural networks.

Facilitate Convergence: By maintaining a constant variance, it becomes easier for the optimizer to converge during training since the gradients are less likely 
to explode or vanish.

Empirical Success: Xavier initialization has been found to work well empirically with standard activation functions like tanh. 
For ReLU activations, a variant called He initialization is often more effective, as it accounts for the non-linearity of ReLU that can potentially halve the variance.

This initialization is particularly important for deep learning models like transformers that involve many layers, 
where maintaining the proper flow of gradients through the network during backpropagation is crucial for effective learning.
'''


