import torch 
import torch.nn as nn
import math

# Building the input embedding: Where the input text is converted into a format that the model can process.
class InputEmbeddings(nn.Module):
    def __init__(self, d_model:int, vocab_size:int):
        # d_model: represents the size of the embedding vector of each word
        # vocab_size: represents total number of unique tokens (words, subwords, characters, etc.) that the model can recognize and process
        
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # This module is often used to store word embeddings and retrieve them using indices.
        self.embedding = nn.Embedding(vocab_size, d_model)

    
    def forward(self,x):
        # In the embedding layers, we multiply those weights by sqrt(d_model)
        return self.embedding(x) * math.sqrt(self.d_model)


# Building the positional encoding: These encodings are added to the embeddings to provide information about the position of each word in the sentence.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        # seq_len: Defines the maximum sequence length
        # dropout: Dropout helps in regularization and prevents overfitting
        
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        self.dropout = nn.Dropout(dropout)


        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # The reason for adding this new dimension "unsqueeze(1)" is to allow for proper broadcasting during subsequent operations

        # formula from the researh paper
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))

        # Apply sin to even positions| cos to odd positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (seq_len, d_model) -> (1, seq_len, d_model)

        # We want it to be registered when saving the model
        self.register_buffer("pe", pe)
    
    def forward(self,x):
        x = x + (self.pe[:,:x.shape[1], :]).requires_grad_(False)
        # TO DO:
        # (self.pe[:,:x.shape[1], :]) -> Full Explanation in the README file
        
        # .requires_grad_(False) Ensure Positional Encodings Are Not Trainable

        return self.dropout(x)


# Building Layer Normalization: It normalizes the inputs across the features for each data point, which helps to reduce the internal covariate shift.
class LayerNormalization(nn.Module):
    def __init__(self, eps:float = 10**-6) -> None:
        # eps: epsilon
        super().__init__()
        self.eps = eps

        # Formula = y = alpha*x_hat +beta
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self,x):
        mean = x.mean(dim =-1, keepdim = True)
        std = x.std(dim =-1, keepdim = True)
        
        x_normalized = (x - mean) / (std + self.eps)

        return self.alpha * x_normalized + self.beta


# Building the Feed Forward Block: It plays a role in processing and transforming the output of the multi-head attention mechanism.
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout: float) -> None:
        # d_ff: The dimensionality of the hidden layer in the feedforward network. This is usually larger than d_model.

        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        # linear_1, which takes an input of size d_model and outputs a vector of size d_ff.

        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2
        # linear_2, which takes an input of size d_ff and outputs a vector of size d_model.
    
    def forward(self,x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
        # input tensor x [batch_size, seq_length, d_model] -> linear_1: W1 * x + B1 [batch_size, seq_length, d_ff]
        # output of linear_1 -> ReLU(x) = max(0, x) -> dropout: which randomly zeroes some of the elements in the tensor during training based on the dropout rate
        # output of dropout -> linear_2: W2 * x + B2: This transforms the tensor back to the original dimension d_model
        

# Building the Multi-Head Attention: 
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model:int, h:int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)
        
        assert d_model % h == 0, "d_model is not divisible"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model) # Wq
        self.w_k = nn.Linear(d_model, d_model) # Wk
        self.w_v = nn.Linear(d_model, d_model) # Wv

        self.w_o = nn.Linear(d_model, d_model) # Wo
    
    # To get the attention score
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # (Batch, h, seq_len, d_k) -> (Batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (Batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores * value), attention_scores

    def forward(self, q, k, v, mask):
        # mask: we mask words that we don't want to interact with each other
        query = self.w_q(q) # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
        key = self.w_q(k) # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
        value = self.w_q(v) # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)

        # Multiple heads
        # [Batch, seq_len, d_model] -> [Batch, seq_len, self.h, self.d_k] -> [Batch, self.h, seq_len, self.d_k]
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        # view: Returns a new tensor with the same data as the self tensor but of a different shape.
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (Batch, h, seq_len, d_k) -> (Batch, seq_len, h, d_k) -> (Batch, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)


# Building the Residual Connection | Add & Norm
class ResidualConnection(nn.Module):
    def __init__(self, dropout:float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


# Building the incoder block
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_borward_block: FeedForwardBlock, dropout:float) -> None:
        self.self_attention_block = self_attention_block
        self.feed_borward_block = feed_borward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x,x,x, src_mask))
        x = self.residual_connection[0](x, self.self_attention_block)

        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# Building the decoder block
class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_borward_block: FeedForwardBlock, dropout:float) -> None:
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_borward_block = feed_borward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x,x,x, tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x,encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, self.feed_borward_block)

        return x

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


# Building Projection Layer | Linear layer
class ProjectionLayer(nn.Module):
    def __init__(self, d_model:int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # (Batch, seq_len, d_model) -> (Batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)


# Building the Transformer
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
        # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    # Create the transformer

    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer