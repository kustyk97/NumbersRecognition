# %%
import random
import torch
from torch.utils.data import DataLoader, datasets
from torchvision import datasets, transforms
from tqdm.auto import tqdm
import pandas as pd
from timeit import default_timer as timer 
from model import Classifier
import argparse
SEED = 19

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Torch version: {torch.__version__}")
print((f"CUDA: {device}"))

"""Parse the arguments from the command line."""
def get_args():
    parser = argparse.ArgumentParser(description="Train a neural network model to classify images of numbers.")
    parser.add_argument("--path_to_save", type=str, default="../models/model_with_classes.pth", help="The path to save the trained model.")
    parser.add_argument("--batch_size", type=int, default=32, help="The batch size to use for training.")
    parser.add_argument("--num_epochs", type=int, default=10, help="The number of epochs to train the model.")
    parser.add_argument("--lr", type=float, default=1e-3, help="The learning rate to use for training.")
    return parser.parse_args()

"""
Performs a single training step for the given model.
Args:
    model (torch.nn.Module): The neural network model to be trained.
    dataloader (torch.utils.data.dataloader): DataLoader providing the training data.
    loss_fn (torch.nn.Module): Loss function to compute the loss.
    optimizer (torch.optim.Optimizer): Optimizer to update the model parameters.
Returns:
    list: A list containing the training accuracy and training loss.
"""
def train_step(model: torch.nn.Module,
                dataloader: torch.utils.data.dataloader,
                loss_fn: torch.nn.Module,
                optimizer) -> list:
        
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

"""
Performs a single testing step for the given model.
Args:
    model (torch.nn.Module): The neural network model to be tested.
    dataloader (torch.utils.data.dataloader): DataLoader providing the testing data.
    loss_fn (torch.nn.Module): Loss function to compute the loss.
Returns:
    list: A list containing the testing accuracy and testing loss.
"""
def test_step(model: torch.nn.Module,
            dataloader: torch.utils.data.dataloader,
            loss_fn: torch.nn.Module) -> list:
    model.eval() 
    
    test_loss, test_acc = 0, 0
    
    # Turn on inference context manager
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)
    
            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
            
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_acc, test_loss

def main():
    args = get_args()
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.num_epochs
    path_to_save = args.path_to_save
    lr = args.lr

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='../data', train=False, download=True, transform=transform)

    print(f"Batch size: {BATCH_SIZE}")
    print(f"Train dataset length: {len(train_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = Classifier().to(device=device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    results_df = pd.DataFrame(columns=["epochs", "train_acc", "train_loss", "test_acc", "test_loss"])

    start_time = timer()

    for epoch in tqdm(range(NUM_EPOCHS)):
        # train
        train_acc, train_loss = train_step(model=model,
                                        dataloader=train_loader,
                                        loss_fn=loss_fn,
                                        optimizer=optimizer)
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
    results_df.to_csv("../results/results.csv", sep=";")
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': train_dataset.classes
    }, path_to_save)


if __name__ == "__main__":
    main()



