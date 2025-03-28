import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from ganomaly.network import WGanomaly
from config import Config, NNCFG, WGanomalyConfig, MODEL_TYPE

class TrainWGanomaly:
    def __init__(self, model, dataloader, config):
        self.model = model
        self.dataloader = dataloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = config.epoch_count

        # Optimizers
        self.optimizer_con = torch.optim.Adam(self.model.generator.parameters(), lr=config.LEARNING_RATE, betas=(config.adam_beta1, config.adam_beta2))
        self.optimizer_adv = torch.optim.Adam(self.model.discriminator.parameters(), lr=config.LEARNING_RATE, betas=(config.adam_beta1, config.adam_beta2))
        self.optimizer_enc = torch.optim.Adam(self.model.encoder.parameters(), lr=config.LEARNING_RATE, betas=(config.adam_beta1, config.adam_beta2))
        self.optimizer_total = torch.optim.Adam(list(self.model.generator.parameters()) + list(self.model.encoder.parameters()) + list(self.model.discriminator.parameters()), lr=config.LEARNING_RATE, betas=(config.adam_beta1, config.adam_beta2))

        self.train_losses = []

    def train_epoch(self, epoch):
        epoch_loss = 0
        self.model.train()
        
        for batch_input in tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.epochs}"):
            batch_input = batch_input.to(self.device)
            outputs = self.model(batch_input)
            losses = self.model.compute_loss(batch_input, outputs)

            # Optimize Contextual Loss
            self.optimizer_con.zero_grad()
            losses["con_loss"].backward(retain_graph=True)
            self.optimizer_con.step()

            # Optimize Adversarial Loss
            self.optimizer_adv.zero_grad()
            losses["adv_loss"].backward(retain_graph=True)
            self.optimizer_adv.step()

            # Optimize Encoder Loss
            self.optimizer_enc.zero_grad()
            losses["enc_loss"].backward(retain_graph=True)  
            self.optimizer_enc.step()

            # Optimize Total Loss
            self.optimizer_total.zero_grad()
            losses["loss"].backward()
            self.optimizer_total.step()

            epoch_loss += losses["loss"].item()

        avg_epoch_loss = epoch_loss / len(self.dataloader)
        self.train_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_epoch_loss:.6f}")

    def train(self):
        for epoch in range(self.epochs):
            self.train_epoch(epoch)

        return self.train_losses

    def save_model(self, config):
        """Save the trained model."""
        model_path = os.path.join(config.MODEL_PATH, f"{self.model.model_id}.pt")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_id': self.model.model_id,
            'epoch_count': self.epochs,
            'learning_rate': config.LEARNING_RATE,
            'batch_size': config.batch_size,
            'optimizer': self.optimizer_con.__class__.__name__.lower(),
            'training_loss': self.train_losses
        }, model_path)

        print(f"Model saved to {model_path}")


def train(cfg):
    """Initialize and train the WGanomaly model."""
    nncfg = NNCFG()
    wganomalyconfig = WGanomalyConfig()
    nncfg.argParser(cfg)
    train_data = cfg.DATASET.get_train_data()
    train_tensors = torch.stack(train_data)
    dataset = TensorDataset(train_tensors)  
    dataloader = DataLoader(dataset, batch_size=nncfg.batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if cfg.MODEL_TYPE == MODEL_TYPE.WGANOMALY:
        model = WGanomaly(wganomalyconfig).to(device)
        trainer = TrainWGanomaly(model, dataloader, nncfg)
        train_losses = trainer.train()
        trainer.save_model(cfg)
