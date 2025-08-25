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
    
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.1):
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

        # Self-attention to focus on important parts
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, batch_first=True, dropout=0.1
        )
    
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, input_dim)
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

        
    def forward(self, x, lengths):
        """
        Forward pass through LSTM
        
        Args:
            x (tensor): Batch of sequences [batch_size, seq_len, input_dim]
            lengths (tensor): Length of each sequence in batch
            
        Returns:
            tensor: Final hidden state for each sequence [batch_size, hidden_dim]
        """

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        final_hidden = hidden[-1]
        return final_hidden
    
    def reconstruct(self, sequence_embeddings, seq_len):
        expanded = sequence_embeddings.unsqueeze(1).expand(-1, seq_len, -1)
        reconstructed = self.decoder(expanded)
        return reconstructed

def extract_student_timestamps(df):
    """
    extract timestamps for each student from DataFrame
    
    Args:
        df: DataFrame includes ['student_id', 'timestamp', ...] columns
        
    Returns:
        dict: student_id -> list of timestamps
    """
    import pandas as pd
    from datetime import datetime
    
    # check data type of timestamp column
    print(f"data type of timestamp: {df['timestamp'].dtype}")
    print(f"sample: {df['timestamp'].iloc[0]}")
    
    # make sure timestamp is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        print("convert to datetime...")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # sort by student_id and timestamp
    df_sorted = df.sort_values(['student_id', 'timestamp']).copy()
    
    # extract timestamps for each student
    student_timestamps = {}
    
    for student_id, group in df_sorted.groupby('student_id'):
        timestamps = group['timestamp'].tolist()
        student_timestamps[student_id] = timestamps
        
        # print info for students with few timestamps
        if len(student_timestamps) <= 3:
            print(f"student {student_id}: {len(timestamps)} timestamps")
            print(f"  starting time: {timestamps[0]}")
            print(f"  end time: {timestamps[-1]}")
            if len(timestamps) > 1:
                duration = (timestamps[-1] - timestamps[0]).total_seconds() / 3600
                print(f"  total time: {duration:.2f} hours")
    
    print(f"\n extract {len(student_timestamps)} students' timestamps")
    
    # summary statistics
    timestamp_counts = [len(ts) for ts in student_timestamps.values()]
    avg_submissions = sum(timestamp_counts) / len(timestamp_counts)
    
    print(f"average of submission of each stu: {avg_submissions:.1f}")
    print(f"minimum submission count: {min(timestamp_counts)}")
    print(f"maximum submission count: {max(timestamp_counts)}")
    
    return student_timestamps

def prepare_sequence_data(student_submission_sequences, embeddings, max_length=None):
    """
    Prepare sequence data for LSTM model
    
    Args:
        student_submission_sequences (dict): Dictionary mapping student IDs to their submission sequences
        embeddings (dict): Dictionary mapping node IDs to embeddings
        
    Returns:
        tuple: (padded_sequences, sequence_lengths, student_ids)
    """

    # If max_length not provided, use 90th percentile of lengths
    if max_length is None:
        lengths = [len(seq) for seq in student_submission_sequences.values()]
        max_length = int(np.percentile(lengths, 100))
        print(f"Use 75 percent max length: {max_length}")

    sequences = []
    lengths = []
    student_ids = []
    
    for student_id, submission_sequence in student_submission_sequences.items():
        # Cut submission_sequence longer than max_length
        if len(submission_sequence) > max_length:
            submission_sequence = submission_sequence[:max_length]
            print(f"student {student_id} from {len(student_submission_sequences[student_id])} cut to {max_length}")
        
        # Create sequence of embeddings
        seq = [embeddings[node_id] for node_id in submission_sequence if node_id in embeddings]
        
        if seq:
            sequences.append(seq)
            lengths.append(len(submission_sequence))
            student_ids.append(student_id)
    
    # Pad sequences
    padded_sequences = []
    for seq in sequences:
        # Pad with zeros to max_length
        padded = seq + [np.zeros_like(seq[0])] * (max_length - len(seq))
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

            reconstructed = model.reconstruct(sequence_embeddings, sequences.size(1))
            
            # Calculate reconstruction loss [batch, seq_len, input_dim]
            loss = torch.mean((reconstructed - sequences)**2)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {total_loss/len(dataloader):.8f}")
    
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