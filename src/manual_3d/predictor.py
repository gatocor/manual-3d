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
    def __init__(self, root_dir, test=False, transform=None):
        """
        Custom dataset for loading images (as .npy files) and ground truth points.
        
        Args:
        - root_dir: Root directory containing the subfolders 'img_origin', 'img_target', and 'pos_target'.
        - transform: Optional transform to apply to the images.
        """
        self.root_dir = root_dir
        if test:
            n = "_test"
        else:
            n = ""
        self.img_origin_dir = os.path.join(root_dir, 'img_origin{}'.format(n))
        self.img_target_dir = os.path.join(root_dir, 'img_target{}'.format(n))
        self.pos_target_dir = os.path.join(root_dir, 'pos_target{}'.format(n))
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
        # self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding_size)
        self.pool = nn.MaxPool3d(pool_size)  # Anisotropic pooling
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        x_pooled = self.pool(x)
        return x_pooled, x

class Dual3DImageNet(nn.Module):
    def __init__(self, shape):
        """
        Args:
        - radius: Shape of the input anisotropic image (e.g., (depth, height, width)).
        """
        super().__init__()
        
        shape = np.array(shape).astype(int)
        self.shape = shape
        self.radius = torch.tensor(np.round((shape-1)/2).astype(int))
        depth, height, width = tuple(shape)
        
        # Calculate the relative scaling factors for pooling based on the smallest dimension
        min_dim = min(shape)
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
        self.gap = nn.AdaptiveAvgPool3d(1)

        # Compute flattened size after contracting blocks (depends on input shape and pooling)
        sample_input = torch.rand(1, depth, height, width)
        sample_output, _ = self.contract3(self.contract2(self.contract1(sample_input)[0])[0])
        # sample_output, _ = self.contract1(sample_input)
        # print(sample_output.shape)
        sample_gap = self.gap(sample_output)
        # flattened_size = sample_output.numel()  # Calculate the flattened size for a single image
        flattened_size = sample_gap.numel()  # Calculate the flattened size for a single image

        # # Final fully connected layers for outputting a 3D point
        # self.fc1 = nn.Linear(flattened_size * 2, 128)  # Adjusting input size based on flattened feature size
        # self.fc2 = nn.Linear(128, 64)
        # self.fc3 = nn.Linear(64, 3)  # Output 3D point
    
        # Final fully connected layers for outputting a 3D point
        self.fc1 = nn.Linear(flattened_size * 2, 64)  # Adjusting input size based on flattened feature size
        self.tahn1 = nn.Tanh()
        # self.fc2 = nn.Linear(64, 32)
        # self.fc3 = nn.Linear(32, 3)  # Output 3D point
        self.fc3 = nn.Linear(64, 3)  # Output 3D point
        self.tahn2 = nn.Tanh()

    def forward(self, img1, img2):
        # Apply the contracting path to the first 3D image
        img1_pooled1, _ = self.contract1(img1)
        img1_pooled2, _ = self.contract2(img1_pooled1)
        img1_features, _ = self.contract3(img1_pooled2)
        img1_gap1 = self.gap(img1_features)

        # Apply the contracting path to the second 3D image
        img2_pooled1, _ = self.contract1(img2)
        img2_pooled2, _ = self.contract2(img2_pooled1)
        img2_features, _ = self.contract3(img2_pooled2)
        img1_gap2 = self.gap(img2_features)

        # Flatten and concatenate the features from both images
        # img1_flat = torch.flatten(img1_features)
        # img2_flat = torch.flatten(img2_features)
        img1_flat = torch.flatten(img1_gap1)
        img2_flat = torch.flatten(img1_gap2)

        # Check dimensions
        combined_features = torch.cat([img1_flat, img2_flat])

        # Pass through fully connected layers to output a 3D point
        x = self.fc1(combined_features)
        x = self.tahn1(x)
        # x = F.relu(self.fc2(x))
        x = self.tahn2(self.fc3(x))
        output_point = self.radius*x
        
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
            return img1, img1, True
        img2 = get_chunk_4d_with_padding(image, point2, self.radius)

        return img1, img2, False

    def predict(self, image, point, forward, layer):

        img1, img2, stop = self.get_images(image, point, forward)
        if stop:
            return

        img_origin = np.expand_dims(img1, axis=0)  # Now shape: (1, depth, height, width)
        img_target = np.expand_dims(img2, axis=0)  # Now shape: (1, depth, height, width)

        # Convert NumPy arrays to PyTorch tensors
        img_origin = torch.tensor(img_origin, dtype=torch.float32)
        img_target = torch.tensor(img_target, dtype=torch.float32)

        new_point = self.forward(img_origin, img_target).detach().numpy()
        new_point += point[1:]
        if forward:
            t = point[0]+1
        else:
            t = point[0]-1
        layer.data = np.append([t],new_point).reshape(1,-1)
    
    def train_model(self, train_loader, test_loader=None, num_epochs=10, lr=0.001):
        """
        Train the model.
        
        Args:
        - train_loader: DataLoader for the training data (should return img1, img2, ground_truth_point).
        - test_loader: DataLoader for the test data.
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
                
                # Zero the gradients before forward pass
                optimizer.zero_grad()
                
                # Forward pass: Get the model predictions
                predicted_point = self(img1, img2)
                
                # Compute the loss (MSE between predicted point and ground truth)
                loss = criterion(predicted_point, ground_truth_point)
                
                # Backward pass: Compute gradients
                loss.backward()
                
                # Accumulate loss for this batch
                running_loss += loss.item()
            
            # After the loop, update the weights
            optimizer.step()

            # Compute average loss for this epoch
            epoch_loss = running_loss / len(train_loader)
            
            # Optionally evaluate on test data
            if test_loader is not None:
                self.eval()  # Set the model to evaluation mode
                
                running_loss_test = 0.0
                
                with torch.no_grad():  # No need to compute gradients during testing
                    for img1, img2, ground_truth_point in test_loader:
                        # Forward pass: Get the model predictions
                        predicted_point = self(img1, img2)
                        
                        # Compute the test loss
                        loss = criterion(predicted_point, ground_truth_point)
                        
                        # Accumulate test loss for this batch
                        running_loss_test += loss.item()
                
                # Compute average test loss for this epoch
                epoch_loss_test = running_loss_test / len(test_loader)
                
                # Print the training and test loss
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Test Loss: {epoch_loss_test:.4f}')
            
            else:
                # Print the training loss only
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

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

        ground_truth_point_relative = torch.tensor(ground_truth_point-point[1:], dtype=torch.float32)

        img1, img2, _ = self.get_images(image, point, forward)

        img_origin = np.expand_dims(img1, axis=0)  # Now shape: (1, depth, height, width)
        img_target = np.expand_dims(img2, axis=0)  # Now shape: (1, depth, height, width)

        # Convert NumPy arrays to PyTorch tensors
        img_origin = torch.tensor(img_origin, dtype=torch.float32)
        img_target = torch.tensor(img_target, dtype=torch.float32)

        self.train()  # Set the model to training mode
        
        optimizer.zero_grad()  # Zero out the gradients
        
        # Forward pass: get the predicted point from the model
        predicted_point = self(img_origin, img_target)
        
        # Compute the loss between the predicted point and the ground truth
        loss = criterion(predicted_point, ground_truth_point_relative)
        
        # Backward pass: compute the gradients
        loss.backward()
        
        # Update model parameters
        optimizer.step()

        self.eval()
        
        return
    
    def train_step2(self, image, point, ground_truth_point, forward, num_translations=5, lr=0.001):
        """
        Perform one step of online training with 3D data augmentation over multiple translations
        before performing backpropagation, accounting for voxel sizes proportional to the image shape.
        
        Args:
        - image: Input image tensor.
        - point: The initial 3D point in the image.
        - ground_truth_point: The ground truth 3D point (tensor).
        - forward: Forward direction.
        - num_translations: Number of translations to apply before backpropagation.
        - lr: Learning rate.
        
        Returns:
        - loss: The computed loss for the current training step.
        """
        # Define the loss function and optimizer
        criterion = nn.MSELoss()  # Mean Squared Error Loss
        optimizer = optim.Adam(self.parameters(), lr=lr)  # Adam optimizer

        # Zero out the gradients at the start
        optimizer.zero_grad()

        # Get the images (img1 and img2)
        img1, img2, _ = self.get_images(image, point, forward)

        # Convert to tensors and add a batch dimension
        img1 = torch.tensor(np.expand_dims(img1, axis=0), dtype=torch.float32)
        img2 = torch.tensor(np.expand_dims(img2, axis=0), dtype=torch.float32)

        # Get image dimensions (depth, height, width)
        img_depth, img_height, img_width = img2.shape[1], img2.shape[2], img2.shape[3]

        accumulated_loss = 0.0  # Accumulate the loss over multiple translations

        # Include the original images for training
        ground_truth_point_relative = torch.tensor(ground_truth_point - point[1:], dtype=torch.float32)  # Calculate relative ground truth

        # Use the original images for the first forward pass (no augmentation)
        self.train()  # Set the model to training mode
        predicted_point_original = self(img1, img2)  # Forward pass with original images

        # Compute the loss between the predicted point and the original ground truth point
        loss_original = criterion(predicted_point_original, ground_truth_point_relative)
        accumulated_loss += loss_original  # Add original loss to accumulated loss

        # Perform multiple translations before backpropagation
        for _ in range(num_translations):
            # Random 3D translation proportional to 10% of the image shape
            translation = (
                np.random.uniform(-0.1 * img_depth, 0.1 * img_depth),  # 10% of the depth
                np.random.uniform(-0.1 * img_height, 0.1 * img_height),  # 10% of the height
                np.random.uniform(-0.1 * img_width, 0.1 * img_width)  # 10% of the width
            )

            # Apply 3D translation to img2 using torch.roll
            img2_aug = torch.roll(img2, shifts=(int(translation[0]), int(translation[1]), int(translation[2])), dims=(1, 2, 3))

            # Transform the ground_truth_point_relative the same way as img2
            ground_truth_point_relative_aug = ground_truth_point_relative + torch.tensor(translation, dtype=torch.float32)

            # Forward pass: get the predicted point from the model with the augmented images
            predicted_point_aug = self(img1, img2_aug)  # Forward pass with augmented img2

            # Compute the loss between the predicted point and the transformed ground truth point
            loss_aug = criterion(predicted_point_aug, ground_truth_point_relative_aug)
            accumulated_loss += loss_aug  # Accumulate the loss for each augmented sample

        # Average the accumulated loss over the original image and the number of translations
        accumulated_loss /= (num_translations + 1)  # +1 to include the original image

        # Backward pass: compute the gradients using the accumulated loss
        accumulated_loss.backward()

        # Update model parameters
        optimizer.step()

        print("Loss {}".format(accumulated_loss.item()))

        self.eval()  # Switch to evaluation mode

        return accumulated_loss

    def save(self, path=None):

        torch.save(self.state_dict(), "{}/model.pt".format(path))

    def load(self, path=None):

        self.load_state_dict(torch.load("{}/model.pt".format(path)))

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
    z_min = int(max(z - rz, 0))
    z_max = int(min(z + rz + 1, image.shape[1]))

    y_min = int(max(y - ry, 0))
    y_max = int(min(y + ry + 1, image.shape[2]))

    x_min = int(max(x - rx, 0))
    x_max = int(min(x + rx + 1, image.shape[3]))

    # Calculate where to place the valid region of the image in the padded chunk
    chunk_z_min = int(max(rz - z, 0))
    chunk_z_max = int(chunk_z_min + (z_max - z_min))

    chunk_y_min = int(max(ry - y, 0))
    chunk_y_max = int(chunk_y_min + (y_max - y_min))

    chunk_x_min = int(max(rx - x, 0))
    chunk_x_max = int(chunk_x_min + (x_max - x_min))

    # Insert the valid portion of the image into the zero-padded chunk
    chunk[chunk_z_min:chunk_z_max, chunk_y_min:chunk_y_max, chunk_x_min:chunk_x_max] = \
        image[int(t), z_min:z_max, y_min:y_max, x_min:x_max]

    return chunk

def save_chunks_and_coords_from_tracks(
        image: np.ndarray, 
        tracks: np.ndarray, 
        radius: tuple, 
        t_folder: str,
        test_partition: float = 0.1
        ):
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
    os.makedirs("{}/{}".format(t_folder,"img_origin_test"), exist_ok=True)
    os.makedirs("{}/{}".format(t_folder,"img_target_test"), exist_ok=True)
    os.makedirs("{}/{}".format(t_folder,"pos_target_test"), exist_ok=True)

    # Sort tracks by id and then by time
    tracks = tracks[np.lexsort((tracks[:, 1], tracks[:, 0]))]

    rz, ry, rx = radius
    chunk_center = np.array([rz, ry, rx])

    # Loop through the unique track ids
    counter = 0
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
                if counter < (1-test_partition):
                    n = ""
                    counter += test_partition 
                elif counter < 1:
                    n = "_test"
                    counter += test_partition 
                else:
                    n = ""
                    counter = 0

                # Get the chunks for time t and t+1
                chunk_t = get_chunk_4d_with_padding(image, (t, z, y, x), radius)
                chunk_t1 = get_chunk_4d_with_padding(image, (t_next, z, y, x), radius)
                # Save the chunks in separate folders for time t and t+1
                np.save("{}/{}/{}".format(t_folder,"img_origin{}".format(n), f"track_{track_id}_t_{t}_{t+1}.npy"), chunk_t)
                np.save("{}/{}/{}".format(t_folder,"img_target{}".format(n), f"track_{track_id}_t_{t}_{t+1}.npy"), chunk_t1)
                np.save("{}/{}/{}".format(t_folder,"pos_target{}".format(n), f"track_{track_id}_t_{t}_{t+1}.npy"), np.array([z, y, x])-chunk_center)

                # Get the chunks for time t and t+1
                chunk_t = get_chunk_4d_with_padding(image, (t, z_next, y_next, x_next), radius)
                chunk_t1 = get_chunk_4d_with_padding(image, (t_next, z_next, y_next, x_next), radius)
                # Save the chunks in separate folders for time t and t+1
                np.save("{}/{}/{}".format(t_folder, "img_origin{}".format(n), f"track_{track_id}_t_{t+1}_{t}.npy"), chunk_t1)
                np.save("{}/{}/{}".format(t_folder, "img_target{}".format(n), f"track_{track_id}_t_{t+1}_{t}.npy"), chunk_t)
                np.save("{}/{}/{}".format(t_folder, "pos_target{}".format(n), f"track_{track_id}_t_{t+1}_{t}.npy"), np.array([z_next, y_next, x_next])-chunk_center)
