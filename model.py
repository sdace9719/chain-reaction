import torch

# Automatically detect device and use GPU if available
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPOGridNet(nn.Module):
    def __init__(self, grid_size: int, num_channels: int = 2, load_weights=None, eval_mode=False, freeze_conv=False):
        super(PPOGridNet, self).__init__()
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