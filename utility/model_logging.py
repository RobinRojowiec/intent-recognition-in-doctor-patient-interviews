# load and save models as checkpoints
import csv
import os

import torch


class TrainingProtocol:
    def __init__(self, file_path: str, log_name: str, headers: []):
        """
        logs loss and accuracy per epoch in a csv
        :param file_path:
        :param log_name:
        """
        self.file_path = file_path + log_name + ".csv"
        self.headers = ["epoch", "train_loss", "dev_loss"] + headers

        # clear logs
        with open(self.file_path, 'w+', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(self.headers)

    def log_epoch(self, epoch: int, loss: float, dev_loss: float, *measures):
        with open(self.file_path, "a", newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow([epoch, loss, dev_loss] + list(measures))


def load_snapshot(model, optimizer=None, directory="./snapshots"):
    class_name = model.__class__.__name__
    checkpoints = []
    for _, _, files in os.walk(directory):
        for file in files:
            if file.startswith(class_name):
                checkpoints.append(file)

    if len(checkpoints) > 0:
        file_name = sorted(checkpoints, reverse=True)[0]
        checkpoint = torch.load(directory + "\\" + file_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        best_acc = float(file_name.split("_")[-1].replace(".model", ""))
        print("\nloaded model at epoch " + str(epoch) + " with acc=" + str(best_acc) + "!\n")

        return epoch, loss, best_acc

    else:
        return 1, 0, 0


def save_snapshot(model, optimizer, epoch, loss, acc, directory):
    checkpoint_file: str = model.__class__.__name__ + "_epoch_" + str(epoch) + "_acc_" + str(
        round(acc * 10000) / 10000) + ".model"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, directory + "\\" + checkpoint_file)
    print("\nstored model!\n")

    for _, _, files in os.walk(directory):
        for file in files:
            if file != checkpoint_file and file.startswith(model.__class__.__name__):
                os.remove(directory + "\\" + file)
