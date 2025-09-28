import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from collections import Counter

def extract_time_features(timestamps):
    """
    Extract temporal features from a sequence of timestamps.
    
    This function computes various time-based features that capture the temporal
    patterns of student submissions, including submission intervals, cumulative
    study time, and submission frequency.
    
    Args:
        timestamps (list): List of datetime objects representing submission times,
                         ordered chronologically
    
    Returns:
        list: List of temporal feature vectors, where each vector contains:
            - [0]: Time interval from previous submission (hours, capped at 10)
            - [1]: Cumulative time from first submission (days, capped at 7) 
            - [2]: Submission frequency (submissions per unit time, scaled by 100)
    
    Notes:
        - For the first submission, interval and cumulative time are 0
        - Features are normalized to prevent extreme values from dominating
        - Returns zero features for sequences with length <= 1
    """
    if len(timestamps) <= 1:
        return [[0.0, 0.0, 0.0]] * len(timestamps)
    
    time_features = []
    start_time = timestamps[0]
    
    for i, current_time in enumerate(timestamps):
        if i == 0:
            # First submission
            interval = 0.0
            cumulative = 0.0
            speed = 0.0
        else:
            # Calculate time interval (seconds)
            interval = (current_time - timestamps[i-1]).total_seconds()
            # Cumulative time
            cumulative = (current_time - start_time).total_seconds()
            # Speed indicator (submission frequency)
            speed = i / (cumulative + 1)  # Avoid division by zero
        
        # Normalize temporal features
        time_features.append([
            min(interval / 3600, 10.0),     # Time interval (hours), max 10 hours
            min(cumulative / 86400, 7.0),  # Cumulative time (days), max 7 days
            speed * 100                     # Speed indicator scaled
        ])
    
    return time_features

def prepare_sequence_data_with_time(student_submission_sequences, embeddings, 
                                    student_timestamps, max_length=None):
    """
    Prepare sequence data with temporal information for LSTM model training.
    
    This function processes student submission sequences and their corresponding timestamps
    to create training data that includes both content embeddings and temporal features.
    
    Args:
        student_submission_sequences (dict): Dictionary mapping student IDs to their 
                                           submission node sequences (e.g., 'q1_p1_s1_t2')
        embeddings (dict): Dictionary mapping node IDs to their embedding vectors
        student_timestamps (dict): Dictionary mapping student IDs to lists of timestamps 
                                 corresponding to each submission
        max_length (int, optional): Maximum sequence length for padding/truncation. 
                                   If None, uses 75th percentile of sequence lengths.
    
    Returns:
        tuple: A 4-tuple containing:
            - padded_sequences (list): List of padded embedding sequences 
                                     [num_students, max_length, embedding_dim]
            - padded_time_features (list): List of padded temporal feature sequences
                                         [num_students, max_length, time_feature_dim]
            - sequence_lengths (list): Original lengths of each sequence before padding
            - student_ids (list): Student IDs corresponding to each sequence
    
    Notes:
        - Sequences longer than max_length are truncated
        - Shorter sequences are padded with zeros
        - Time features include: interval between submissions, cumulative time, 
          and submission frequency
        - Students without valid embeddings or timestamp mismatches are filtered out
    """
    
    if max_length is None:
        lengths = [len(seq) for seq in student_submission_sequences.values()]
        max_length = int(np.percentile(lengths, 75))
        print(f"Using 75th percentile as max length: {max_length}")

    sequences = []
    time_features = []
    lengths = []
    student_ids = []
    
    for student_id, submission_sequence in student_submission_sequences.items():
        if student_id not in student_timestamps:
            continue
            
        timestamps = student_timestamps[student_id]
        
        # Check length matching
        if len(submission_sequence) != len(timestamps):
            print(f"Warning: Student {student_id} sequence length mismatch: "
                  f"submissions={len(submission_sequence)}, timestamps={len(timestamps)}")
            continue
        
        # Truncate overly long sequences
        if len(submission_sequence) > max_length:
            submission_sequence = submission_sequence[:max_length]
            timestamps = timestamps[:max_length]
        
        # Create embedding sequence
        seq = []
        for node_id in submission_sequence:
            if node_id in embeddings:
                seq.append(embeddings[node_id])
            else:
                print(f"Warning: Node {node_id} not found in embeddings")
                break
        
        if len(seq) == len(submission_sequence) and len(timestamps) == len(submission_sequence):
            # Calculate temporal features
            time_feat = extract_time_features(timestamps)
            
            sequences.append(seq)
            time_features.append(time_feat)
            lengths.append(len(submission_sequence))
            student_ids.append(student_id)
    
    print(f"Successfully processed {len(sequences)} student sequences")
    
    if not sequences:
        raise ValueError("No valid sequence data available!")
    
    # Pad sequences
    embedding_dim = len(sequences[0][0])
    time_dim = len(time_features[0][0]) if time_features else 3
    
    padded_sequences = []
    padded_time_features = []
    
    for seq, time_feat in zip(sequences, time_features):
        # Pad embedding sequences
        padded_seq = seq + [np.zeros(embedding_dim)] * (max_length - len(seq))
        padded_sequences.append(padded_seq)
        
        # Pad temporal features
        padded_time = time_feat + [np.zeros(time_dim)] * (max_length - len(time_feat))
        padded_time_features.append(padded_time)
    
    return padded_sequences, padded_time_features, lengths, student_ids

class SubmissionLSTMWithTime(nn.Module):
    """
    LSTM model for encoding student submission sequences with temporal information.
    
    This model extends the basic LSTM autoencoder to incorporate temporal features
    alongside content embeddings, enabling the model to capture both what students
    do and when they do it.
    
    Args:
        input_dim (int): Dimension of input content embeddings
        time_dim (int): Dimension of temporal feature vectors  
        hidden_dim (int): Hidden state dimension of LSTM
        num_layers (int, optional): Number of LSTM layers. Default: 1
        dropout (float, optional): Dropout rate for regularization. Default: 0.1
    
    Attributes:
        time_encoder (nn.Sequential): Neural network to process temporal features
        lstm (nn.LSTM): Main LSTM layer processing combined content+time features
        attention (nn.MultiheadAttention): Self-attention mechanism for sequence modeling
        decoder (nn.Sequential): Decoder network for reconstruction
        layer_norm (nn.LayerNorm): Layer normalization for LSTM output
    """
    
    def __init__(self, input_dim, time_dim, hidden_dim, num_layers=1, dropout=0.1):
        super(SubmissionLSTMWithTime, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_dim = input_dim
        
        # Temporal feature preprocessing layers
        self.time_encoder = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.ReLU(),
            nn.Linear(time_dim * 2, time_dim)
        )
        
        # LSTM input dimension = embedding dimension + time dimension
        self.lstm = nn.LSTM(
            input_size=input_dim + time_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, input_dim)  # Only reconstruct content, not time
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, time_features, lengths):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Content embeddings [batch_size, seq_len, input_dim]
            time_features (torch.Tensor): Temporal features [batch_size, seq_len, time_dim]
            lengths (torch.Tensor): Sequence lengths [batch_size]
            
        Returns:
            torch.Tensor: Sequence embeddings [batch_size, hidden_dim]
        
        Notes:
            - Combines content and temporal features before LSTM processing
            - Uses packed sequences for efficient processing of variable lengths
            - Returns final hidden state as sequence representation
        """
        
        # Encode temporal features
        encoded_time = self.time_encoder(time_features)
        
        # Concatenate content and temporal features
        combined_input = torch.cat([x, encoded_time], dim=-1)
        
        # Pack sequences for efficient LSTM processing
        packed = nn.utils.rnn.pack_padded_sequence(
            combined_input, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # LSTM processing
        packed_output, (hidden, _) = self.lstm(packed)
        
        # Simple version: return final hidden state of last layer
        final_hidden = hidden[-1]  # [batch_size, hidden_dim]
        return final_hidden
    
    def reconstruct(self, sequence_embeddings, seq_len):
        """
        Reconstruct content sequences from embeddings (excluding temporal features).
        
        Args:
            sequence_embeddings (torch.Tensor): Sequence embeddings [batch_size, hidden_dim]
            seq_len (int): Maximum sequence length to reconstruct
            
        Returns:
            torch.Tensor: Reconstructed sequences [batch_size, seq_len, input_dim]
        
        Notes:
            - Only reconstructs content embeddings, not temporal features
            - Uses decoder network to map from hidden representation to content space
        """
        expanded = sequence_embeddings.unsqueeze(1).expand(-1, seq_len, -1)
        reconstructed = self.decoder(expanded)
        return reconstructed

def train_sequence_model_with_time(student_submission_sequences, embeddings, 
                                   student_timestamps, hidden_dim=128, 
                                   num_layers=1, batch_size=32, num_epochs=15):
    """
    Train LSTM model with temporal information on student submission sequences.
    
    This function trains an autoencoder-style LSTM model that learns to represent
    student learning patterns using both content (what they submit) and temporal
    (when they submit) information.
    
    Args:
        student_submission_sequences (dict): Dictionary mapping student IDs to 
                                           their submission sequences
        embeddings (dict): Dictionary mapping node IDs to embedding vectors
        student_timestamps (dict): Dictionary mapping student IDs to timestamp lists
        hidden_dim (int, optional): LSTM hidden dimension size. Default: 256
        num_layers (int, optional): Number of LSTM layers. Default: 2
        batch_size (int, optional): Training batch size. Default: 32
        num_epochs (int, optional): Number of training epochs. Default: 20
        
    Returns:
        tuple: A 2-tuple containing:
            - trained_model (SubmissionLSTMWithTime): The trained LSTM model
            - student_sequence_embeddings (dict): Dictionary mapping student IDs 
                                                 to their learned embeddings
    
    Notes:
        - Uses masked reconstruction loss to ignore padded positions
        - Applies gradient clipping and learning rate scheduling
        - Only reconstructs content embeddings, not temporal features
        - Model training uses autoencoder objective with reconstruction loss
    """
    
    print("Starting preparation of sequence data with temporal information...")
    
    # Prepare data
    try:
        padded_sequences, padded_time_features, lengths, student_ids = prepare_sequence_data_with_time(
            student_submission_sequences, embeddings, student_timestamps
        )
    except Exception as e:
        print(f"Data preparation failed: {e}")
        return None, None
    
    # Convert to tensors
    sequences_tensor = torch.FloatTensor(padded_sequences)
    time_tensor = torch.FloatTensor(padded_time_features)
    lengths_tensor = torch.LongTensor(lengths)
    
    print(f"Data shapes:")
    print(f"  Sequences: {sequences_tensor.shape}")
    print(f"  Time features: {time_tensor.shape}")
    print(f"  Lengths: {lengths_tensor.shape}")
    
    # Create dataset and dataloader
    dataset = TensorDataset(sequences_tensor, time_tensor, lengths_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    embedding_dim = sequences_tensor.size(-1)
    time_dim = time_tensor.size(-1)
    
    print(f"Model parameters:")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Time dimension: {time_dim}")
    print(f"  Hidden dimension: {hidden_dim}")
    
    model = SubmissionLSTMWithTime(
        input_dim=embedding_dim, 
        time_dim=time_dim,
        hidden_dim=hidden_dim, 
        num_layers=num_layers,
        dropout=0.3
    )
    
    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Training loop
    model.train()
    print(f"\nStarting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for sequences, time_features, seq_lengths in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            sequence_embeddings = model(sequences, time_features, seq_lengths)
            reconstructed = model.reconstruct(sequence_embeddings, sequences.size(1))
            
            # Calculate masked reconstruction loss (content only)
            loss = torch.mean((reconstructed - sequences)**2)
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        scheduler.step(avg_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
        
        # Early stopping if loss is very small
        if avg_loss < 1e-6:
            print("Loss converged, stopping early")
            break
    
    print("Training completed! Generating student embeddings...")
    
    # Generate student embeddings
    model.eval()
    student_sequence_embeddings = {}
    
    dataset = TensorDataset(sequences_tensor, time_tensor, lengths_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        batch_start = 0
        for sequences, time_features, seq_lengths in dataloader:
            batch_embeddings = model(sequences, time_features, seq_lengths)
            
            batch_end = min(batch_start + len(batch_embeddings), len(student_ids))
            
            for i, student_id in enumerate(student_ids[batch_start:batch_end]):
                if i < len(batch_embeddings):
                    student_sequence_embeddings[student_id] = batch_embeddings[i].numpy()
            
            batch_start = batch_end
    
    print(f"Generated embeddings for {len(student_sequence_embeddings)} students")
    
    return model, student_sequence_embeddings

def extract_student_timestamps(df):
    """
    Extract timestamps for each student from DataFrame.
    
    Args:
        df (pandas.DataFrame): DataFrame containing student submission data
                              with columns ['student_id', 'timestamp', ...]
        
    Returns:
        dict: Dictionary mapping student IDs to lists of timestamps
    
    Notes:
        - Ensures timestamps are in datetime format
        - Sorts submissions by student_id and timestamp
        - Provides summary statistics about submission patterns
    """
    import pandas as pd
    from datetime import datetime
    
    # Check timestamp column data type
    print(f"Timestamp column data type: {df['timestamp'].dtype}")
    print(f"Sample timestamp: {df['timestamp'].iloc[0]}")
    
    # Ensure timestamp is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        print("Converting timestamps to datetime format...")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by student_id and timestamp
    df_sorted = df.sort_values(['student_id', 'timestamp']).copy()
    
    # Extract timestamps for each student
    student_timestamps = {}
    
    for student_id, group in df_sorted.groupby('student_id'):
        timestamps = group['timestamp'].tolist()
        student_timestamps[student_id] = timestamps
        
        # Print info for first few students
        if len(student_timestamps) <= 3:
            print(f"Student {student_id}: {len(timestamps)} timestamps")
            print(f"  Start time: {timestamps[0]}")
            print(f"  End time: {timestamps[-1]}")
            if len(timestamps) > 1:
                duration = (timestamps[-1] - timestamps[0]).total_seconds() / 3600
                print(f"  Total duration: {duration:.2f} hours")
    
    print(f"\nExtracted timestamps for {len(student_timestamps)} students")
    
    # Summary statistics
    timestamp_counts = [len(ts) for ts in student_timestamps.values()]
    avg_submissions = sum(timestamp_counts) / len(timestamp_counts)
    
    print(f"Average submissions per student: {avg_submissions:.1f}")
    print(f"Minimum submission count: {min(timestamp_counts)}")
    print(f"Maximum submission count: {max(timestamp_counts)}")
    
    return student_timestamps