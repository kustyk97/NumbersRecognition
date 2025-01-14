# %%
import torch
import random

# %%
SEED = 19
BATCH_SIZE = 32

# %%

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
print(f"Torch version: {torch.__version__}")
print((f"CUDA: {device}"))

# %%
from torchvision import datasets, transforms

# %%
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='../data', train=False, download=True, transform=transform)

# %%
len(train_dataset)

# %%
from torch.utils.data import DataLoader

# %%
print(f"Batch size: {BATCH_SIZE}")

# %%
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# %%
from model import Classifier
model = Classifier().to(device=device)

# %% [markdown]
# ### training loop

# %%
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

# %%
from tqdm.auto import tqdm

# %%
import torch.utils.data.dataloader


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.dataloader,
               loss_fn: torch.nn.Module) -> list:
    
    model.train()    
    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)
        
        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_acc, train_loss

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.dataloader,
              loss_fn: torch.nn.Module) -> list:
    model.eval() 
    
    test_loss, test_acc = 0, 0
    
    # Turn on inference context manager
    with torch.inference_mode():
        for batch, (X, y) in enumerate(test_loader):
            # Send data to target device
            X, y = X.to(device), y.to(device)
    
            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
            
    test_loss = test_loss / len(test_loader)
    test_acc = test_acc / len(test_loader)
    return test_acc, test_loss

    

# %%

NUM_EPOCHS = 10

# %%
import pandas as pd

results_df = pd.DataFrame(columns=["epochs", "train_acc", "train_loss", "test_acc", "test_loss"])

from timeit import default_timer as timer 
start_time = timer()

for epoch in tqdm(range(NUM_EPOCHS)):
    # train
    train_acc, train_loss = train_step(model=model,
                                       dataloader=train_loader,
                                       loss_fn=loss_fn)
    # test
    test_acc, test_loss = test_step(model=model,
                                    dataloader=test_loader,
                                    loss_fn=loss_fn)
    row = pd.DataFrame({"epoch": [epoch], 
                       "train_loss": [train_loss], 
                       "train_acc": [train_acc], 
                       "test_loss": [test_loss], 
                       "test_acc": [test_acc]})
    results_df = pd.concat([results_df, row] , ignore_index=True)

    print(
            f"epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

end_time = timer()

print(f"Total time of training {end_time-start_time:.3f} seconds")


# %%
results_df

# %%
results_df.to_csv("../results/results.csv", sep=";")

# %%
torch.save({
    'model_state_dict': model.state_dict(),
    'class_names': train_dataset.classes
}, '../models/model_with_classes.pth')

# %%

print(f"Total time of training {end_time-start_time:.3f} seconds")

# %%



