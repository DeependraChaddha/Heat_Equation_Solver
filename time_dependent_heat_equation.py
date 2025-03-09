import os
import torch
import torch.nn as nn
import torch.optim  # Used for optimizer
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter  # Used for animation
from typing import Callable
import tqdm
import pickle

class TimeDependentHeatEquationSolver:
  def __init__(self,
               nx:int,
               ny:int,
               nt:int,
               x_range:tuple,
               y_range:tuple,
               t_range:tuple,
               device:str): #can put 'cuda' to use gpu if gpu available, if gpu not available, by default cpu will be used irrespective of what is mentioned as device
    self.nx=nx
    self.ny=ny
    self.nt=nt
    self.x_range=x_range
    self.y_range=y_range
    self.t_range=t_range
    self.device=device if torch.cuda.is_available() else 'cpu'
    self.model_checkpoint_path = None
    print('Instance of Class TimeDependentHeatEquationSolver made.')

  def make_grid(self):
    x = torch.linspace(self.x_range[0], self.x_range[1], self.nx).reshape(-1, 1)
    y = torch.linspace(self.y_range[0], self.y_range[1], self.ny).reshape(-1, 1)
    t = torch.linspace(self.t_range[0], self.t_range[1], self.nt).reshape(-1, 1)
    X,Y,T = torch.meshgrid(x.squeeze(), y.squeeze(), t.squeeze(), indexing='ij')
    X=X.requires_grad_(True)
    Y=Y.requires_grad_(True)
    T=T.requires_grad_(True)
    self.X=X.to(self.device)
    self.Y=Y.to(self.device)
    self.T=T.to(self.device)
    return self.X,self.Y,self.T

  def save_checkpoint(self, model, optimizer, loss, model_name, hyperparameters):
    self.checkpoint_dir = f"checkpoint_{model_name}_nx{self.nx}_ny{self.ny}_nt{self.nt}_epochs{hyperparameters['epochs']}"
    os.makedirs(self.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(self.checkpoint_dir, "model_checkpoint.pkl")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "hyperparameters": hyperparameters
    }
    torch.save(checkpoint, checkpoint_path)
    self.model_checkpoint_path = checkpoint_path

  def train_step(self, model: nn.Module, X: torch.Tensor, Y: torch.Tensor, T: torch.Tensor, loss_fn: Callable, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler = None):
      '''Performs one epoch of training process.
      If calling this function explicitly, make
      sure model and optimizer are in the same
      device as tensors. Do not use loss functions
      with learnable parameters, if used manually
      make sure they are stored in correct device'''

      # Forward Pass
      prediction = model(X, Y, T, self.nx, self.ny, self.nt)
      # Calculate Loss
      loss = loss_fn(model=model, x=X, y=Y, t=T, nx=self.nx, ny=self.ny, nt=self.nt)
      # Set Optimizer Zero Grad
      optimizer.zero_grad()
      # Backward pass
      loss.backward()
      # Optimizer Step
      optimizer.step()
      # Scheduler step
      if scheduler:
          try:
              scheduler.step()
          except TypeError:  # To handle both types of schedulers
              scheduler.step(loss.item())
      return X, Y, T, prediction, loss.item(), model.state_dict()

  def train(self, model, loss_fn, optimizer, epochs, model_name, get_loss_curve=False, scheduler=None):
    try:
        X, Y, T = self.X, self.Y, self.T
        X.requires_grad_(True)
        Y.requires_grad_(True)
        T.requires_grad_(True)
        model = model.to(self.device)
        best_loss = float('inf')

        loss_values = []
        plt.ioff()
        for epoch in tqdm.tqdm(range(epochs)):
          X, Y, T, Z, loss, model_params = self.train_step(model, X, Y, T, loss_fn, optimizer, scheduler)
          loss_values.append(loss)
          if loss < best_loss:
              best_loss = loss
              self.save_checkpoint(model, optimizer, loss, model_name, {"epochs": epochs})

        print(f"Best Loss Achieved: {best_loss}")
        if get_loss_curve:
          plt.figure(figsize=(8, 5))
          plt.plot(range(1, epochs + 1), loss_values, marker='o', linestyle='-', color='r', label="Loss")
          plt.xlabel("Epoch")
          plt.ylabel("Loss")
          plt.title("Loss Curve Over Epochs")
          plt.legend()
          plt.grid()
          loss_curve_path = os.path.join(self.checkpoint_dir, "loss_curve.png")
          plt.savefig(loss_curve_path)
          print(f"Loss curve saved at {loss_curve_path}")
          plt.show()

        return self.model_checkpoint_path
    except AttributeError:
        print("Make grid first, Call make_grid method")

  def animate_final_prediction(self, model_class, model_checkpoint_path=None):
    try:
        if model_checkpoint_path is None:
            if self.model_checkpoint_path is None:
                raise AttributeError("Model checkpoint not found. Please train the model first by calling the train function.")
            model_checkpoint_path = self.model_checkpoint_path

        checkpoint = torch.load(model_checkpoint_path, map_location=self.device)
        model = model_class().to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        with torch.no_grad():
          Z_all = model(self.X, self.Y, self.T, self.nx, self.ny, self.nt)  # Compute for all time steps

        def update(frame):
          '''This function clears the graph,
             evaluates the model prediciton
             for a particular time, and plots
             the surface, then, plots the axes.'''
          ax.clear()

          ax.plot_surface(self.X.detach().cpu().numpy()[:,:,frame], self.Y.detach().cpu().numpy()[:,:,frame], Z_all.detach().cpu().numpy()[:,:,frame], cmap="viridis")
          ax.set_xlabel("X-axis")
          ax.set_ylabel("Y-axis")
          ax.set_zlabel("T")
          ax.set_title(f"Predicted Heat Distribution at Time {frame}")

        ani = FuncAnimation(fig, update, frames=self.nt, repeat=False)
        # Ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Define file path and save animation
        animation_path = os.path.join(self.checkpoint_dir, "heat_distribution.gif")
        ani.save(animation_path, writer=PillowWriter(fps=10))

        # Store path in self.model_checkpoint_path
        self.model_checkpoint_path = animation_path
        print(f"Animation saved at: {animation_path}")

        # Close figure to free memory
        plt.close(fig)

        plt.show()
    except AttributeError as e:
        print(e)
