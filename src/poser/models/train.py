import torch
from pytorch_lightning import Trainer
from st_gcn_lstm import ST_GCN_LSTM
from _loader import ZebData  
from torch.utils.data import DataLoader

# Example configurations (modify these based on your dataset)
graph_config = {
    "center_node": 0  # Modify based on your setup
}

data_config = {
    "data_dir": "path/to/data",  # UPDATE THIS PATH
    "augment": True,
    "ideal_sample_no": 3000,
    "shift": True,
}

hparams = type("HParams", (object,), {"learning_rate": 1e-4, "batch_size": 32, "dropout": 0.3})

# Load dataset
train_data = ZebData(
    data_file="path/to/train_data.npy",   # UPDATE THIS PATH
    label_file="path/to/train_labels.npy",  # UPDATE THIS PATH
    transform="align",  
    augment=True
)

val_data = ZebData(
    data_file="path/to/val_data.npy",  
    label_file="path/to/val_labels.npy",  
    transform="align",  
    augment=False
)

# Create data loaders
train_loader = DataLoader(train_data, batch_size=hparams.batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_data, batch_size=hparams.batch_size, shuffle=False, num_workers=4)

# Initialize the model
model = ST_GCN_LSTM(
    in_channels=3, 
    num_class=5,  
    graph_cfg=graph_config,  
    data_cfg=data_config,  
    hparams=hparams
)

# Create Trainer
trainer = Trainer(max_epochs=50, gpus=1)  

# Train the model
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# Save the trained model checkpoint
trainer.save_checkpoint("st_gcn_lstm_trained.ckpt")
