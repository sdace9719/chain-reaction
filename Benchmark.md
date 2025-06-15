# Chain Reaction AI Benchmark Results

## Model Architecture

The model (PPOGridNet) is a deep neural network specifically designed for the Chain Reaction game. It combines convolutional layers for spatial feature extraction with fully connected layers for policy and value prediction. The architecture is grid-size independent, allowing the convolutional weights to be reused for larger grid sizes.

```mermaid
graph TD
    Input[Input: 2 channels x G x G] --> Conv1[Conv2D: 64 filters]
    Conv1 --> ReLU1[ReLU]
    ReLU1 --> Conv2[Conv2D: 128 filters]
    Conv2 --> ReLU2[ReLU]
    ReLU2 --> Conv3[Conv2D: 128 filters]
    Conv3 --> ReLU3[ReLU]
    ReLU3 --> Flatten[Flatten]
    Flatten --> FC1[FC: 512 units]
    FC1 --> ReLU4[ReLU]
    ReLU4 --> PolicyHead[Policy Head: G*G units]
    ReLU4 --> ValueHead[Value Head: 1 unit]
```

Key components:
- **Input Layer**: Takes a 2-channel grid representation (G x G)
- **Convolutional Layers**: 
  - 3 Conv2D layers with ReLU activation
  - Channel progression: 2 → 64 → 128 → 128
  - Kernel size: 3x3 with padding=1 to maintain grid dimensions
- **Fully Connected Layers**:
  - Flattened conv output → 512 units
  - ReLU activation
- **Dual Heads**:
  - Policy Head: Outputs action logits (G*G units)
  - Value Head: Outputs state value (1 unit)

## Training Process

The training process was conducted in multiple phases to progressively improve the model's performance:

1. **Initial Training vs Random Opponent**
   - Model was first trained against random opponents
   - Established basic gameplay understanding
   - Learned fundamental strategies

2. **Training vs Mixed Policy Pool**
   - Expanded training against a diverse set of policies
   - Policies were randomly sampled during training
   - Helped model adapt to different playing styles

3. **Top-5 Strategy Implementation**
   - Implemented a sophisticated training regime
   - Only sampled opponents from top 5 based on ELO ratings
   - Included model checkpoints in the opponent pool
   - Ensured continuous improvement against strong opponents

## Benchmark Results

The model demonstrates strong performance across different opponent types:

- **vs Random Opponents**: ~96% win rate
- **vs Custom Policy (Gemini)**: ~96% win rate
- **vs ChatGPT-4**: 93% win rate

### Training Metrics

![image](https://github.com/user-attachments/assets/2773e1c5-ec5e-4547-bcd9-e6fd41061255)


The model was trained on a 5x5 grid but the convolutional weights can be reused for larger grid sizes, making it adaptable to different game configurations.
