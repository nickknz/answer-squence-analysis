import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class SubmissionSequenceDataset(Dataset):
    """Dataset for student submission sequences"""
    
    def __init__(self, sequence_data, sequence_lengths):
        """
        Args:
            sequence_data (list): List of padded submission sequences
            sequence_lengths (list): Original lengths of each sequence
        """
        self.sequences = torch.FloatTensor(sequence_data)
        self.lengths = torch.LongTensor(sequence_lengths)
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        return self.sequences[idx], self.lengths[idx]

class SubmissionLSTM(nn.Module):
    """LSTM model for encoding submission sequences"""
    
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.0):
        super(SubmissionLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
    def forward(self, x, lengths):
        """
        Forward pass through LSTM
        
        Args:
            x (tensor): Batch of sequences [batch_size, seq_len, input_dim]
            lengths (tensor): Length of each sequence in batch
            
        Returns:
            tensor: Final hidden state for each sequence [batch_size, hidden_dim]
        """
        # Pack padded sequences for efficient computation
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # Process with LSTM
        _, (hidden, _) = self.lstm(packed)
        
        # Get the final hidden state
        final_hidden = hidden[-1]
        
        return final_hidden

def prepare_sequence_data(student_submission_sequences, embeddings):
    """
    Prepare sequence data for LSTM model
    
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

def train_sequence_model(student_submission_sequences, embeddings, hidden_dim=128, 
                         num_layers=1, batch_size=32, num_epochs=10, learning_rate=0.001):
    """
    Train LSTM model on student submission sequences
    
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
    model = SubmissionLSTM(input_dim=embedding_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    
    # Define optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train model (autoencoder fashion)
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        
        for sequences, seq_lengths in dataloader:
            optimizer.zero_grad()
            
            # Get sequence embeddings
            sequence_embeddings = model(sequences, seq_lengths)
            
            # Now expand the decoded embeddings to compare with sequences
            reconstructed = sequence_embeddings.unsqueeze(1).expand(-1, sequences.size(1), -1)
            
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
        for i, (sequences, seq_lengths) in enumerate(dataloader):
            batch_embeddings = model(sequences, seq_lengths)
            
            # Map embeddings back to student IDs
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(student_ids))
            
            for j, student_id in enumerate(student_ids[start_idx:end_idx]):
                student_sequence_embeddings[student_id] = batch_embeddings[j].numpy()
    
    return model, student_sequence_embeddings