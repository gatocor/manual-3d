import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class ImagePointDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Custom dataset for loading images (as .npy files) and ground truth points.
        
        Args:
        - root_dir: Root directory containing the subfolders 'img_origin', 'img_target', and 'pos_target'.
        - transform: Optional transform to apply to the images.
        """
        self.root_dir = root_dir
        self.img_origin_dir = os.path.join(root_dir, 'img_origin')
        self.img_target_dir = os.path.join(root_dir, 'img_target')
        self.pos_target_dir = os.path.join(root_dir, 'pos_target')
        self.transform = transform

        # List all files in the img_origin directory to use as dataset entries
        self.image_files = sorted(os.listdir(self.img_origin_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Get image filenames for img_origin and img_target
        img_origin_path = os.path.join(self.img_origin_dir, self.image_files[idx])
        img_target_path = os.path.join(self.img_target_dir, self.image_files[idx])  # Assuming matching filenames

        # Get corresponding ground truth position
        pos_target_path = os.path.join(self.pos_target_dir, self.image_files[idx])  # Assuming .txt files for points

        # Load images as NumPy arrays
        img_origin = np.load(img_origin_path)  # Shape: (depth, height, width)
        img_target = np.load(img_target_path)

        # Add a channel dimension to each image (from [D, H, W] to [C, D, H, W])
        # Assuming single-channel data, so channel dimension = 1
        img_origin = np.expand_dims(img_origin, axis=0)  # Now shape: (1, depth, height, width)
        img_target = np.expand_dims(img_target, axis=0)  # Now shape: (1, depth, height, width)

        # Convert NumPy arrays to PyTorch tensors
        img_origin = torch.tensor(img_origin, dtype=torch.float32)
        img_target = torch.tensor(img_target, dtype=torch.float32)

        # If transform is provided, apply it (for resizing, normalization, etc.)
        if self.transform:
            img_origin = self.transform(img_origin)
            img_target = self.transform(img_target)

        # Load ground truth 3D point (assuming space-separated values in .txt file)
        ground_truth_point = np.load(pos_target_path)

        # Convert ground truth point to a tensor
        ground_truth_point = torch.tensor(ground_truth_point, dtype=torch.float32)

        return img_origin, img_target, ground_truth_point

class ContractingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), pool_size=(2, 2, 2)):
        """
        Contracting block with anisotropic pooling.
        
        Args:
        - in_channels: Number of input channels.
        - out_channels: Number of output channels.
        - pool_size: Pooling size for each dimension (can be anisotropic).
        """
        super(ContractingBlock, self).__init__()

        padding_size = tuple((k - 1) // 2 for k in kernel_size)
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding_size)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding_size)
        self.pool = nn.MaxPool3d(pool_size)  # Anisotropic pooling
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x_pooled = self.pool(x)
        return x_pooled, x

class Dual3DImageNet(nn.Module):
    def __init__(self, radius):
        """
        Args:
        - radius: Shape of the input anisotropic image (e.g., (depth, height, width)).
        """
        super(Dual3DImageNet, self).__init__()
        
        self.radius = radius/2
        depth, height, width = radius
        
        # Calculate the relative scaling factors for pooling based on the smallest dimension
        min_dim = min(radius)
        scale_depth = depth / min_dim
        scale_height = height / min_dim
        scale_width = width / min_dim

        # Proportional kernel sizes based on image dimensions
        kernel_size_depth = max(3, int(np.round(3 * scale_depth)))  # Minimum size 3
        kernel_size_height = max(3, int(np.round(3 * scale_height)))
        kernel_size_width = max(3, int(np.round(3 * scale_width)))

        # print("Proportional kernel sizes:", kernel_size_depth, kernel_size_height, kernel_size_width)

        # Pool sizes based on proportional scaling
        pool_size_depth = max(1, int(np.round(scale_depth)))
        pool_size_height = max(1, int(np.round(scale_height)))
        pool_size_width = max(1, int(np.round(scale_width)))

        # print("Anisotropic pool sizes:", pool_size_depth, pool_size_height, pool_size_width)

        # Contracting path shared between both images
        self.contract1 = ContractingBlock(1, 32, kernel_size=(kernel_size_depth, kernel_size_height, kernel_size_width), pool_size=(pool_size_depth, pool_size_height, pool_size_width))
        self.contract2 = ContractingBlock(32, 64, kernel_size=(3, 3, 3), pool_size=(1, 1, 1))  # Uniform kernel size for subsequent layers
        self.contract3 = ContractingBlock(64, 128, kernel_size=(3, 3, 3), pool_size=(1, 1, 1))

        # Compute flattened size after contracting blocks (depends on input shape and pooling)
        sample_input = torch.rand(1, 1, depth, height, width)
        sample_output, _ = self.contract3(self.contract2(self.contract1(sample_input)[0])[0])
        # print(sample_output.shape)
        flattened_size = sample_output.numel()  # Calculate the flattened size for a single image

        # Final fully connected layers for outputting a 3D point
        self.fc1 = nn.Linear(flattened_size * 2, 128)  # Adjusting input size based on flattened feature size
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)  # Output 3D point
    
    def forward(self, img1, img2):
        # Apply the contracting path to the first 3D image
        img1_pooled1, _ = self.contract1(img1)
        img1_pooled2, _ = self.contract2(img1_pooled1)
        img1_features, _ = self.contract3(img1_pooled2)

        # Apply the contracting path to the second 3D image
        img2_pooled1, _ = self.contract1(img2)
        img2_pooled2, _ = self.contract2(img2_pooled1)
        img2_features, _ = self.contract3(img2_pooled2)

        # Flatten and concatenate the features from both images
        img1_flat = torch.flatten(img1_features)
        img2_flat = torch.flatten(img2_features)

        # Check dimensions
        combined_features = torch.cat([img1_flat, img2_flat])

        # Pass through fully connected layers to output a 3D point
        x = F.relu(self.fc1(combined_features))
        x = F.relu(self.fc2(x))
        output_point = self.fc3(x)
        
        return output_point
    
    def get_images(self, image, point, forward):

        img1 = get_chunk_4d_with_padding(image, point, self.radius)
        #Second point
        point2 = point.copy()
        if forward:
            point2[0] += 1
        else:
            point2[0] -= 1
        #Break if endpoint
        if point2[0] < 0 or point2[0] >= image.shape[0]:
            return point
        img2 = get_chunk_4d_with_padding(image, point2, self.radius)

    def predict(self, image, point, forward):

        img1, img2 = self.get_images(image, point, forward)

        img_origin = np.expand_dims(img1, axis=0)  # Now shape: (1, depth, height, width)
        img_target = np.expand_dims(img2, axis=0)  # Now shape: (1, depth, height, width)

        # Convert NumPy arrays to PyTorch tensors
        img_origin = torch.tensor(img_origin, dtype=torch.float32)
        img_target = torch.tensor(img_target, dtype=torch.float32)

        return self.forward(img1, img2)
    
    def train_model(self, train_loader, num_epochs=10, lr=0.001):
        """
        Train the Dual3DImageNet model.
        
        Args:
        - train_loader: DataLoader for the training data (should return img1, img2, ground_truth_point).
        - num_epochs: Number of training epochs.
        - lr: Learning rate for the optimizer.
        
        Returns:
        - The trained model.
        """
        # Define the loss function and optimizer
        criterion = nn.MSELoss()  # Mean Squared Error Loss
        optimizer = optim.Adam(self.parameters(), lr=lr)  # Adam optimizer
        
        # Loop over epochs
        for epoch in range(num_epochs):
            self.train()  # Set the model to training mode
            
            running_loss = 0.0
            
            # Loop over the training data
            for i, (img1, img2, ground_truth_point) in enumerate(train_loader):
                optimizer.zero_grad()  # Zero the gradients
                
                # Forward pass: Get the model predictions
                predicted_point = self(img1, img2)
                
                # Compute the loss (MSE between predicted point and ground truth)
                loss = criterion(predicted_point, ground_truth_point)
                
                # Backward pass: Compute gradients and update the weights
                loss.backward()
                optimizer.step()
                
                # Accumulate the loss
                running_loss += loss.item()
            
            # Print average loss for the epoch
            epoch_loss = running_loss / len(train_loader)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

        self.eval()
        
        return

    def train_step(self, image, point, ground_truth_point, forward, lr=0.001):
        """
        Perform one step of online training.
        
        Args:
        - model: The PyTorch model (Dual3DImagePointNet).
        - img1: A single 3D image (tensor).
        - img2: A second 3D image (tensor).
        - ground_truth_point: The ground truth 3D point (tensor).
        - optimizer: Optimizer for the model parameters (e.g., Adam).
        - criterion: Loss function (e.g., MSELoss).
        
        Returns:
        - loss: The computed loss for the current training step.
        """
        # Define the loss function and optimizer
        criterion = nn.MSELoss()  # Mean Squared Error Loss
        optimizer = optim.Adam(self.parameters(), lr=lr)  # Adam optimizer

        img1, img2 = self.get_images(image, point, forward)

        self.train()  # Set the model to training mode
        
        optimizer.zero_grad()  # Zero out the gradients
        
        # Forward pass: get the predicted point from the model
        predicted_point = self(img1, img2)
        
        # Compute the loss between the predicted point and the ground truth
        loss = criterion(predicted_point, ground_truth_point)
        
        # Backward pass: compute the gradients
        loss.backward()
        
        # Update model parameters
        optimizer.step()

        self.eval()
        
        return
    
    def save(self, path=None):

        torch.save(self.state_dict(), path)

    def load(self, path=None):

        self.load_state_dict(torch.load(path))

def get_chunk_4d_with_padding(image: np.ndarray, point: tuple, radius: tuple):
    """
    Extracts a chunk from a 4D image (t, z, y, x) around a given point with an anisotropic spatial radius,
    and pads with zeros if the chunk exceeds the image boundaries.
    
    Parameters:
    image (np.ndarray): 4D array representing the image (t, z, y, x).
    point (tuple): A tuple (t, z, y, x) representing the center point in time and space.
    radius (tuple): A tuple (rz, ry, rx) representing the radius in the spatial (z, y, x) dimensions.
    
    Returns:
    np.ndarray: A 4D chunk of the image with padding if necessary.
    """
    # Unpack the point and radius
    t, z, y, x = point
    rz, ry, rx = radius

    # Spatial chunk size (including the center point)
    chunk_size_z = 2 * rz + 1
    chunk_size_y = 2 * ry + 1
    chunk_size_x = 2 * rx + 1

    # Create an empty array (chunk) of the desired size filled with zeros
    chunk = np.zeros((chunk_size_z, chunk_size_y, chunk_size_x), dtype=image.dtype)

    # Calculate the bounding box for spatial dimensions, clipped to image boundaries
    z_min = max(z - rz, 0)
    z_max = min(z + rz + 1, image.shape[1])

    y_min = max(y - ry, 0)
    y_max = min(y + ry + 1, image.shape[2])

    x_min = max(x - rx, 0)
    x_max = min(x + rx + 1, image.shape[3])

    # Calculate where to place the valid region of the image in the padded chunk
    chunk_z_min = max(rz - z, 0)
    chunk_z_max = chunk_z_min + (z_max - z_min)

    chunk_y_min = max(ry - y, 0)
    chunk_y_max = chunk_y_min + (y_max - y_min)

    chunk_x_min = max(rx - x, 0)
    chunk_x_max = chunk_x_min + (x_max - x_min)

    # Insert the valid portion of the image into the zero-padded chunk
    chunk[chunk_z_min:chunk_z_max, chunk_y_min:chunk_y_max, chunk_x_min:chunk_x_max] = \
        image[t, z_min:z_max, y_min:y_max, x_min:x_max]

    return chunk

def save_chunks_and_coords_from_tracks(
        image: np.ndarray, 
        tracks: np.ndarray, 
        radius: tuple, 
        t_folder: str):
    """
    Extract and save pairs of chunks (t, t+1) and their spatial coordinates from the image for each track in the tracks data.
    
    Parameters:
    image (np.ndarray): 4D image (t, z, y, x).
    tracks (np.ndarray): Array of track points in the form (id, t, z, y, x).
    radius (tuple): A tuple (rz, ry, rx) representing the radius in the spatial (z, y, x) dimensions.
    t_folder (str): Path to the folder where chunks at time t will be saved.
    t1_folder (str): Path to the folder where chunks at time t+1 will be saved.
    t_coords_folder (str): Path to the folder where spatial points at time t will be saved.
    t1_coords_folder (str): Path to the folder where spatial points at time t+1 will be saved.
    
    Returns:
    None
    """
    # Create output folders if they don't exist
    os.makedirs(t_folder, exist_ok=True)
    os.makedirs("{}/{}".format(t_folder,"img_origin"), exist_ok=True)
    os.makedirs("{}/{}".format(t_folder,"img_target"), exist_ok=True)
    os.makedirs("{}/{}".format(t_folder,"pos_target"), exist_ok=True)

    # Sort tracks by id and then by time
    tracks = tracks[np.lexsort((tracks[:, 1], tracks[:, 0]))]

    rz, ry, rx = radius
    chunk_center = np.array([rz, ry, rx])

    # Loop through the unique track ids
    for track_id in np.unique(tracks[:, 0]):
        # Extract the track data for this specific track_id
        track_data = tracks[tracks[:, 0] == track_id]

        # Loop through the time points, and extract chunk pairs (t, t+1)
        for i in range(len(track_data) - 1):
            t, z, y, x = track_data[i, 1:].astype(int)
            t_next, z_next, y_next, x_next = track_data[i + 1, 1:].astype(int)
            track_id, t, t_next = int(track_id), int(t), int(t_next)

            # Ensure consecutive time points (t and t+1)
            if t_next == t + 1:
                # Get the chunks for time t and t+1
                chunk_t = get_chunk_4d_with_padding(image, (t, z, y, x), radius)
                chunk_t1 = get_chunk_4d_with_padding(image, (t_next, z, y, x), radius)
                # Save the chunks in separate folders for time t and t+1
                np.save("{}/{}/{}".format(t_folder,"img_origin", f"track_{track_id}_t_{t}_{t+1}.npy"), chunk_t)
                np.save("{}/{}/{}".format(t_folder,"img_target", f"track_{track_id}_t_{t}_{t+1}.npy"), chunk_t1)
                np.save("{}/{}/{}".format(t_folder,"pos_target", f"track_{track_id}_t_{t}_{t+1}.npy"), np.array([z, y, x])-chunk_center)

                # Get the chunks for time t and t+1
                chunk_t = get_chunk_4d_with_padding(image, (t, z_next, y_next, x_next), radius)
                chunk_t1 = get_chunk_4d_with_padding(image, (t_next, z_next, y_next, x_next), radius)
                # Save the chunks in separate folders for time t and t+1
                np.save("{}/{}/{}".format(t_folder, "img_origin", f"track_{track_id}_t_{t+1}_{t}.npy"), chunk_t1)
                np.save("{}/{}/{}".format(t_folder, "img_target", f"track_{track_id}_t_{t+1}_{t}.npy"), chunk_t)
                np.save("{}/{}/{}".format(t_folder, "pos_target", f"track_{track_id}_t_{t+1}_{t}.npy"), np.array([z_next, y_next, x_next])-chunk_center)
