#Make imports
import os
import torch
import torch.nn as nn
import torch.optim  # Used for optimizer
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation  # Used for animation
from typing import Callable
import tqdm
import pickle
class HeatEquationSolver:
  def __init__(self,
               nx:int,
               ny:int,
               x_range:tuple,
               y_range:tuple,
               device:str): #can put 'cuda' to use gpu if gpu available, if gpu not available, by default cpu will be used irrespective of what is mentioned as device
    self.nx=nx
    self.ny=ny
    self.x_range=x_range
    self.y_range=y_range
    self.device=device if torch.cuda.is_available() else 'cpu'
    print('Instance of Class HeatEquationSolver made, call make_imports method to get all required libraries')

  def make_grid(self):

    '''Takes no input only makes
       a mesh grid from instance
       attributes.
    '''
    #Make linear space of x and y and reshape to the shape(nx,1) and (ny,1) respectively (Reshaping is done so that in case of matrix multiplication or broadcasting no error is encountered)
    x = torch.linspace(self.x_range[0], self.x_range[1], self.nx).reshape(-1, 1)
    y = torch.linspace(self.y_range[0], self.y_range[1], self.ny).reshape(-1, 1)

    #Make meshgrid
    X,Y=torch.meshgrid(x.squeeze(),y.squeeze(),indexing='ij')
    X=X.reshape(self.nx,self.ny).requires_grad_(True)
    Y=Y.reshape(self.nx,self.ny).requires_grad_(True)

    #Send data to selected device and assign X,Y
    self.X=X.to(self.device)
    self.Y=Y.to(self.device)

    #return X and Y
    return self.X,self.Y

  def save_checkpoint(self, model, optimizer, loss, model_name, hyperparameters):
    '''Checkpoints model with best loss value'''
    self.checkpoint_dir = f"checkpoint_{model_name}_nx{self.nx}_ny{self.ny}_epochs{hyperparameters['epochs']}"
    os.makedirs(self.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(self.checkpoint_dir, "model_checkpoint.pkl")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "hyperparameters": hyperparameters
    }
    # Saving the checkpoint to the file
    torch.save(checkpoint, checkpoint_path)
  def train_step(self,
                 model:nn.Module,
                 X:torch.Tensor,
                 Y:torch.Tensor,
                 loss_fn:Callable,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler._LRScheduler = None):
    '''Performs one epoch of training process.
    If calling this function explicitly, make
    sure model and optimizer are in the same
    device as tensors. Do not use loss functions
    with learnable parameters, if used manually
    make sure they are stored in correct device'''

    #Forward Pass
    prediction=model(X,Y,self.nx,self.ny)
    #Calculate Loss
    loss=loss_fn(model=model, x=X,y=Y, nx=self.nx,ny=self.ny)
    #Set Optimizer Zero Grad
    optimizer.zero_grad()
    #Backward pass
    loss.backward()
    #Optimizer Step
    optimizer.step()
    #scheduler step
    if scheduler:
      try:
          scheduler.step()
      except TypeError: #To handle both types of schedulers
          scheduler.step(loss.item())
    return  X, Y, prediction, loss.item(), model.state_dict()


  def train(self, model, loss_fn, optimizer, epochs, model_name, animate_training=False, get_loss_curve=False, scheduler=None):
    try:
        X, Y = self.X, self.Y
        X.requires_grad_(True)
        Y.requires_grad_(True)
        model = model.to(self.device)
        best_loss = float('inf')

        loss_values = []
        frames = []
        plt.ioff()#Switching off plt interative mode
        '''Training Loop'''
        for epoch in tqdm.tqdm(range(epochs)):
          X, Y, Z, loss, model_params = self.train_step(model, X, Y, loss_fn, optimizer, scheduler)
          loss_values.append(loss)
          '''Checkpointing'''
          if loss < best_loss:
              best_loss = loss
              self.save_checkpoint(model, optimizer, loss, model_name, {"epochs": epochs})
          '''Saving frames for animation'''
          if animate_training:
              frames.append((X.detach().cpu().numpy().copy(), Y.detach().cpu().numpy().copy(), Z.detach().cpu().numpy().copy(), epoch))

        print(f"Best Loss Achieved: {best_loss}")
        '''Plot Loss Curve'''
        if get_loss_curve:
          print(f"Plotting Loss Curve")
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

        '''Animate Training Process'''
        if animate_training:
          print(f"Animating the training process")
          fig = plt.figure(figsize=(8, 6))
          ax = fig.add_subplot(111, projection='3d')

          def update(frame_idx):
            X_np, Y_np, Z_np, epoch = frames[frame_idx]
            ax.cla()
            ax.plot_surface(X_np, Y_np, Z_np, cmap='viridis')
            ax.set_xlabel("X-axis")
            ax.set_ylabel("Y-axis")
            ax.set_zlabel("T")
            ax.set_title(f"Epoch: {epoch+1}")

          ani = FuncAnimation(fig, update, frames=len(frames), interval=500)
          animation_path = os.path.join(f"checkpoint_{model_name}_nx{self.nx}_ny{self.ny}_epochs{epochs}", "training_animation.gif")
          fps = max(1, len(frames) // 15) #Setting fps so that animation ends in 15 seconds irrespective of number of frames
          ani.save(animation_path, writer="pillow", fps=fps)
          print(f"Animation saved at {animation_path}")

        return Z, loss_values
    except AttributeError:
        print("Make grid first, Call make_grid method")
  def load_and_evaluate_model(self, model_class, checkpoint_path):
    """
    Loads a model from a saved checkpoint, evaluates it on X, Y, and plots a 3D graph.

    Args:
        model_class (type): The model class (e.g., MyModel, not a string).
        checkpoint_path (str): Path to the saved model checkpoint (.pkl file).
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=self.device)

    # Create an instance of the provided model class
    model = model_class().to(self.device)

    # Load model parameters
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Evaluate model on grid
    with torch.no_grad():
        Z = model(self.X, self.Y, self.nx, self.ny)

    # Plot the results
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(self.X.detach().cpu().numpy(), self.Y.detach().cpu().numpy(), Z.detach().cpu().numpy(), cmap="viridis")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("T")
    ax.set_title("Predicted Heat Distribution")
    plt.show()

