import torch
from torch.cuda import device
from tqdm import tqdm
import pytorch_lightning as pl
from config import input_size, hidden_size, output_size, num_classes, batch_size, num_workers,min_epochs, max_epochs, data_dir, accelerator, devices,learning_rate
from model import Model
from dataset import MNISTDataModule
from callbacks import CustomCallback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.strategies import DeepSpeedStrategy


torch.set_float32_matmul_precision('medium') #precision for matrix multiplication, can be 'high', 'medium', or 'low'
#This setting can help improve performance on certain hardware, 
#especially GPUs, by allowing for faster computations with a trade-off in numerical precision ******

if __name__ == '__main__':
    logger = pl.loggers.TensorBoardLogger("tb_logs", name="mnist_model")

# Strategy for Mult. GPU training
    strategy = DeepSpeedStrategy
# Profiler - compute, time management;
    profiler = pl.profilers.PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_logs/profiler1"),
        schedule=torch.profiler.schedule(skip_first=10, wait=1, wramp=1, active=20),
    )

    model = Model(input_size = input_size, 
                  learning_rate = learning_rate, 
                  hidden_size = hidden_size, 
                  output_size = output_size, 
                  num_classes = num_classes)
    model.to(device)

    datamodule = MNISTDataModule(data_dir = data_dir, 
                         batch_size = batch_size, 
                         num_workers = num_workers)

    trainer = pl.Trainer(min_epochs = min_epochs,
                          max_epochs = max_epochs,
                          strategy=strategy ,
                          devices = devices, 
                          accelerator = accelerator, 
                          logger = logger,
                          profiler = profiler,
                          callbacks = [model.early_stopping()],
                          # OR
                          #callbacks = [CustomCallback()],
                          )
    trainer.fit(model, datamodule)
    trainer.validate(model, datamodule)
    trainer.test(model, datamodule)
    # OR
    #trainer.save_checkpoint("model.ckpt")
    #trainer.eval()--for inference
    # Store Weights __only__
    # torch.save(model.state_dict(), "wieghts_only.pt")
