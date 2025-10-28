"""
T5 Model Configuration
Hyperparameters for fine-tuning T5 on QA task
"""

class T5Config:
    # Random
    SEED = 42

    #
    MAX_LENGTH = 200
    
    # Model 
    MODEL = "t5-base"  

    # Device
    DEVICE = "cude"  # or "cpu"
    
    # Training hyperparameters
    BATCH_SIZE = 16  
    LEARNING_RATE = 3e-4
    EPOCHS = 20
    WEIGHT_DECAY = 0.01

    # Training settings
    TRAIN_SIZE = 0.85

    
    # Paths
    DATA_PATH = "./data/train.csv"  

    CHECKPOINT_DIR = "./checkpoints"
    OUTPUT_DIR = "./output_model"
    
    