import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import normalize

class SubmissionSequenceDataset(Dataset):
    """Dataset for student submission sequences"""
    
    def __init__(self, sequence_data, sequence_lengths):
        self.sequences = torch.FloatTensor(sequence_data)
        self.lengths = torch.LongTensor(sequence_lengths)
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        return self.sequences[idx], self.lengths[idx]

class SequenceAutoencoder(nn.Module):
    """Sequence autoencoder model for student submission sequences"""
    
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.0):
        super(SequenceAutoencoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Encoder LSTM
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Decoder LSTM
        self.decoder = nn.LSTM(
            input_size=hidden_dim,  # Input is the hidden state
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection layer
        self.output_layer = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x, lengths):
        """
        Forward pass through the autoencoder
        
        Args:
            x (tensor): Batch of sequences [batch_size, seq_len, input_dim]
            lengths (tensor): Length of each sequence in batch
            
        Returns:
            tuple: (encoded_sequence, reconstructed_sequence)
                - encoded_sequence: [batch_size, hidden_dim]
                - reconstructed_sequence: [batch_size, seq_len, input_dim]
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Pack padded sequences for efficient computation
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # Encode the sequence
        _, (hidden, cell) = self.encoder(packed_input)
        
        # Get final hidden state as encoded representation
        encoded = hidden[-1]  # [batch_size, hidden_dim]
        
        # Use encoded vector as input for each decoder step
        decoder_input = encoded.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Decode
        decoder_output, _ = self.decoder(decoder_input)
        
        # Project to input dimension
        reconstructed = self.output_layer(decoder_output)
        
        return encoded, reconstructed

def prepare_sequence_data(student_submission_sequences, embeddings):
    """
    Prepare sequence data for autoencoder model
    
    Args:
        student_submission_sequences (dict): Dictionary mapping student IDs to their submission sequences
        embeddings (dict): Dictionary mapping node IDs to embeddings
        
    Returns:
        tuple: (padded_sequences, sequence_lengths, student_ids)
    """
    sequences = []
    lengths = []
    student_ids = []
    
    for student_id, submission_sequence in student_submission_sequences.items():
        # Create sequence of embeddings
        seq = [embeddings[node_id] for node_id in submission_sequence if node_id in embeddings]
        
        if seq:
            sequences.append(seq)
            lengths.append(len(seq))
            student_ids.append(student_id)
    
    # Find max sequence length for padding
    max_len = max(lengths)
    
    # Pad sequences
    padded_sequences = []
    for seq in sequences:
        # Pad with zeros to max_len
        padded = seq + [np.zeros_like(seq[0])] * (max_len - len(seq))
        padded_sequences.append(padded)
    
    return padded_sequences, lengths, student_ids

def train_sequence_autoencoder(student_submission_sequences, embeddings, hidden_dim=8,
                               num_layers=2, batch_size=32, num_epochs=20, learning_rate=0.001):
    """
    Train sequence autoencoder on student submission sequences
    
    Args:
        student_submission_sequences (dict): Dictionary mapping student IDs to submission sequences
        embeddings (dict): Dictionary mapping node IDs to embeddings
        hidden_dim (int): Hidden dimension size for LSTM
        num_layers (int): Number of LSTM layers
        batch_size (int): Batch size for training
        num_epochs (int): Number of training epochs
        learning_rate (float): Learning rate
        
    Returns:
        tuple: (trained_model, student_sequence_embeddings)
    """
    # Prepare sequence data
    padded_sequences, lengths, student_ids = prepare_sequence_data(student_submission_sequences, embeddings)
    
    # Create dataset and dataloader
    dataset = SubmissionSequenceDataset(padded_sequences, lengths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    embedding_dim = len(list(embeddings.values())[0])
    model = SequenceAutoencoder(input_dim=embedding_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    
    # Define optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
    
    # Train model
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        
        for sequences, seq_lengths in dataloader:
            optimizer.zero_grad()
            
            # Forward pass through autoencoder
            _, reconstructed = model(sequences, seq_lengths)
            
            # Calculate reconstruction loss
            loss = torch.mean((reconstructed - sequences)**2)

            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    # Generate sequence embeddings for all students
    model.eval()
    dataset = SubmissionSequenceDataset(padded_sequences, lengths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    student_sequence_embeddings = {}
    
    with torch.no_grad():
        batch_start = 0
        for sequences, seq_lengths in dataloader:
            # Get encoded representations
            encoded, _ = model(sequences, seq_lengths)
            
            # Map encoded vectors back to student IDs
            batch_size = encoded.size(0)
            for i in range(batch_size):
                if batch_start + i < len(student_ids):
                    student_id = student_ids[batch_start + i]
                    student_sequence_embeddings[student_id] = encoded[i].numpy()
            
            batch_start += batch_size
    
    return model, student_sequence_embeddings