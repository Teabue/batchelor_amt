import torch
import torch.nn as nn
import torch.optim as optim
import math

class Transformer(nn.Module):
    def __init__(self, n_mel_bins, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, device):
        super(Transformer, self).__init__()
        self.device = device
        
        # We get embeddings by using a Dense layer 
        self.encoder_embedding = nn.Linear(n_mel_bins, d_model)
        # self.decoder_embedding = nn.Linear(tgt_vocab_size, d_model)        
        
        # self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = torch.any((src != -1), axis=-1).unsqueeze(1).unsqueeze(2) # special mask for spectrogram
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3) # batch_size x 1 x tgt_seq_length x 1
        seq_length = tgt.size(1)
        nopeak_mask = torch.tril(torch.ones(1, seq_length, seq_length), diagonal=0).bool()
        nopeak_mask[:, 0, 1] = True # SOS can also see the tempo
        tgt_mask = tgt_mask & nopeak_mask.to(self.device)
        return src_mask, tgt_mask
    
    def get_sequence_predictions(self, src, tgt, max_seq_length) -> torch.Tensor:
        for i in range(max_seq_length):
            output = self.forward(src, tgt, training=False)
            output = output[:,-1,:].unsqueeze(0)
            output = torch.argmax(output, dim=-1)
            if output.squeeze().item() == 2:
                # if predicted end of sequence
                return tgt
            tgt = torch.cat([tgt, output], dim=1)
        return tgt
        
    def forward(self, src, tgt, training=True):
        src_mask, tgt_mask = self.generate_mask(src, tgt)

        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt))) # (batch_size, t_seq_length, d_model)
        
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x
    
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize dimensions
        self.d_model = d_model # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.d_k = d_model // num_heads # Dimension of each head's key, query, and value
        
        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model) # Query transformation
        self.W_k = nn.Linear(d_model, d_model) # Key transformation
        self.W_v = nn.Linear(d_model, d_model) # Value transformation
        self.W_o = nn.Linear(d_model, d_model) # Output transformation
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]





    
if __name__ == "__main__":
    import yaml
    import tqdm
    from utils.vocabularies import Vocabulary
    from utils.data_loader import TransformerDataset
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    n_mel_bins = 128
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = 256 # doesn't really matter, can be set to whatever as long as its larger than the longest sequence - it's a pteprocess step for PE
    dropout = 0.1

    with open('Transformer/configs/preprocess_config.yaml', 'r') as file:
        p_config = yaml.safe_load(file)

    with open('Transformer/configs/vocab_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        
    vocab = Vocabulary(config)
    vocab.define_vocabulary(p_config['max_beats'])
    
    tgt_vocab_size = vocab.vocab_size
    
    
    transformer = Transformer(n_mel_bins, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    dataset = TransformerDataset(r'/work3/s214629/preprocessed_data_old')

    transformer.train()

    num_epochs = 50
    batch_size = 20
    for epoch in range(100):
            
        train_loader = dataset.get_split('train', batch_size=batch_size, shuffle=True)
        pbar = tqdm.tqdm(train_loader, total = len(train_loader), \
                         desc=f'Train: Loss: [{1}], Epochs: {epoch}/{num_epochs}', leave = False)    

        optimizer.zero_grad()
        losses = []
        for spectrograms, tokens in pbar:#iter(train_loader):
            spectrograms = spectrograms.to(device)
            tokens = tokens.to(device)
            
            output = transformer(src=spectrograms, tgt=tokens)
            loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tokens.contiguous().view(-1))
            loss.backward()
            optimizer.step()
            print(f"Epoch: {epoch+1}, Loss: {loss.item()}")