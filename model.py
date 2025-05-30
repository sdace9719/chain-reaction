import torch

# Automatically detect device and use GPU if available
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPOGridNet(nn.Module):
    def __init__(self, grid_size: int, num_channels: int = 2, fc_hidden_size: int = 512,load_weights=None, eval_mode=False, freeze_conv=False):
        super(PPOGridNet, self).__init__()
        # Store device and move model parameters to that device
        self.grid_size = grid_size
        self.device = device # Make sure device is defined or passed

        self.conv = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), # Output: (B, 128, G, G)
            nn.ReLU()
        )

        # Calculate the flattened size after convolution
        # For G=5, this is 128 * 5 * 5 = 3200
        conv_output_flat_size = 128 * grid_size * grid_size 

        self.fc = nn.Sequential(
            nn.Flatten(), # Flattens (B, 128, G, G) to (B, 128*G*G)
            nn.Linear(conv_output_flat_size, fc_hidden_size),
            nn.ReLU(),
            # nn.Linear(fc_hidden_size, fc_hidden_size // 2), # Optional second FC layer
            # nn.ReLU()
        )
        
        # current_fc_output_size = fc_hidden_size // 2 if using second FC layer else fc_hidden_size
        current_fc_output_size = fc_hidden_size

        self.policy_head = nn.Linear(current_fc_output_size, grid_size * grid_size)
        self.value_head = nn.Linear(current_fc_output_size, 1)

        # Move to device
        self.to(self.device)

        # Load weights if provided
        if load_weights:
            self.load_state_dict(torch.load(load_weights, map_location=self.device))
        
        # Set evaluation mode if specified
        if eval_mode:
            self.eval()
            
        # Freeze convolutional layers if specified
        if freeze_conv:
            for param in self.conv.parameters():
                param.requires_grad = False

    def get_best_action(self, obs, valid_mask):
        """Get the best action (argmax) instead of sampling"""
        with torch.no_grad():
            logits, _ = self.forward(torch.tensor(obs, dtype=torch.float32))
            masked_logits = logits.masked_fill(~valid_mask, -1e9)
            return torch.argmax(masked_logits).item()

    def forward(self, x):
        """
        Input:
            x: tensor of shape (B, C, G, G) or (C, G, G)
        Output:
            action_logits: shape (B, G*G) or (G*G)
            state_value: shape (B, 1) or (1)
        """
        # Ensure input is on the correct device
        x = x.to(self.device)
        
        # Add batch dimension if input is a single sample (C, G, G)
        # Store whether a batch dimension was added, for consistent squeezing later.
        input_was_3d = False
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # Converts (C, G, G) to (1, C, G, G)
            input_was_3d = True
        
        # x is now guaranteed to be 4D: (B, C, G, G)
        
        x_conv_output = self.conv(x)  # Output shape e.g., (B, 128, G, G)
        
        # x = self.pool(x)          # REMOVE THIS LINE
        
        # self.fc now expects the output of self.conv (e.g., (B, 128, G, G))
        # because self.fc's first layer is nn.Flatten()
        x_fc_output = self.fc(x_conv_output) # Output shape e.g., (B, 512)

        action_logits = self.policy_head(x_fc_output)  # Shape (B, G*G)
        state_value = self.value_head(x_fc_output)    # Shape (B, 1)
        
        # Remove the batch dimension if it was added at the start
        if input_was_3d:
            action_logits = action_logits.squeeze(0)  # Shape (G*G)
            state_value = state_value.squeeze(0)    # Shape (1), or scalar if further squeezed
            # If state_value was (1,1) and you want a scalar, you might do state_value.squeeze().item()
            # or ensure the value head's output is used as (1) and handled outside.
            # For consistency with PPO, (1) is often fine for a single sample's value.

        return action_logits, state_value
    

class PPOGridNet_deep_fc(nn.Module):
    def __init__(self, grid_size: int, num_channels: int = 2, load_weights=None, eval_mode=False, freeze_conv=False):
        super(PPOGridNet_deep_fc, self).__init__()
        # Store device and move model parameters to that device
        self.grid_size = grid_size
        self.device = device

        # 🧠 Shared convolutional encoder (spatial structure)
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, padding=1),  # (B, 64, G, G)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),            # (B, 128, G, G)
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),           # (B, 128, G, G)
            nn.ReLU()
        )

        # 🔄 Grid-size-independent pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Output: (B, 128, 1, 1)

        # 🔢 Fully connected trunk
        self.fc = nn.Sequential(
            nn.Flatten(),           # (B, 128)
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # 🎯 Policy head → logits for each action (grid cell)
        self.policy_head = nn.Linear(256, grid_size * grid_size)

        # 💰 Value head → scalar state value
        self.value_head = nn.Linear(256, 1)

        # tranfer to gpu
        self.conv = self.conv.to(self.device)
        self.pool = self.pool.to(self.device)
        self.fc = self.fc.to(self.device)
        self.policy_head = self.policy_head.to(self.device)
        self.value_head = self.value_head.to(self.device)

        # Load weights if provided
        if load_weights:
            self.load_state_dict(torch.load(load_weights, map_location=self.device))
        
        # Set evaluation mode if specified
        if eval_mode:
            self.eval()
            
        # Freeze convolutional layers if specified
        if freeze_conv:
            for param in self.conv.parameters():
                param.requires_grad = False
            # Also freeze the pooling layer since it's part of feature extraction
            for param in self.pool.parameters():
                param.requires_grad = False

    def get_best_action(self, obs, valid_mask):
        """Get the best action (argmax) instead of sampling"""
        with torch.no_grad():
            logits, _ = self.forward(torch.tensor(obs, dtype=torch.float32))
            masked_logits = logits.masked_fill(~valid_mask, -1e9)
            return torch.argmax(masked_logits).item()

    def forward(self, x):
        """
        Input:
            x: tensor of shape (B, C, G, G)
        Output:
            action_logits: shape (B, G*G)
            state_value: shape (B, 1)
        """
        # Ensure input is on the correct device
        x = x.to(self.device)
        
        # Add batch dimension if needed
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        
        x = self.conv(x)            # (B, 128, G, G)
        x = self.pool(x)           # (B, 128, 1, 1)
        x = self.fc(x)             # (B, 256)

        action_logits = self.policy_head(x)  # (B, G*G)
        state_value = self.value_head(x)     # (B, 1)
        
        # Remove batch dimension if it was added
        if len(x.shape) == 2:
            action_logits = action_logits.squeeze(0)
            state_value = state_value.squeeze(0)

        return action_logits, state_value
    
class PPOGridNet_deep_wide_fc(nn.Module):
    def __init__(self, grid_size: int, num_channels: int = 2, load_weights=None, eval_mode=False, freeze_conv=False):
        super(PPOGridNet_deep_wide_fc, self).__init__()
        # Store device and move model parameters to that device
        self.grid_size = grid_size
        self.device = device

        # 🧠 Shared convolutional encoder (spatial structure)
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, padding=1),  # (B, 64, G, G)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),            # (B, 128, G, G)
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),           # (B, 128, G, G)
            nn.ReLU()
        )

        # 🔄 Grid-size-independent pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Output: (B, 128, 1, 1)

        # 🔢 Fully connected trunk
        self.fc = nn.Sequential(
            nn.Flatten(),           # (B, 128)
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # 🎯 Policy head → logits for each action (grid cell)
        self.policy_head = nn.Linear(256, grid_size * grid_size)

        # 💰 Value head → scalar state value
        self.value_head = nn.Linear(256, 1)

        # tranfer to gpu
        self.conv = self.conv.to(self.device)
        self.pool = self.pool.to(self.device)
        self.fc = self.fc.to(self.device)
        self.policy_head = self.policy_head.to(self.device)
        self.value_head = self.value_head.to(self.device)

        # Load weights if provided
        if load_weights:
            self.load_state_dict(torch.load(load_weights, map_location=self.device))
        
        # Set evaluation mode if specified
        if eval_mode:
            self.eval()
            
        # Freeze convolutional layers if specified
        if freeze_conv:
            for param in self.conv.parameters():
                param.requires_grad = False
            # Also freeze the pooling layer since it's part of feature extraction
            for param in self.pool.parameters():
                param.requires_grad = False

    def get_best_action(self, obs, valid_mask):
        """Get the best action (argmax) instead of sampling"""
        with torch.no_grad():
            logits, _ = self.forward(torch.tensor(obs, dtype=torch.float32))
            masked_logits = logits.masked_fill(~valid_mask, -1e9)
            return torch.argmax(masked_logits).item()

    def forward(self, x):
        """
        Input:
            x: tensor of shape (B, C, G, G)
        Output:
            action_logits: shape (B, G*G)
            state_value: shape (B, 1)
        """
        # Ensure input is on the correct device
        x = x.to(self.device)
        
        # Add batch dimension if needed
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        
        x = self.conv(x)            # (B, 128, G, G)
        x = self.pool(x)           # (B, 128, 1, 1)
        x = self.fc(x)             # (B, 256)

        action_logits = self.policy_head(x)  # (B, G*G)
        state_value = self.value_head(x)     # (B, 1)
        
        # Remove batch dimension if it was added
        if len(x.shape) == 2:
            action_logits = action_logits.squeeze(0)
            state_value = state_value.squeeze(0)

        return action_logits, state_value