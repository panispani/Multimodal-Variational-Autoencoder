import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model import MVAE
from MNISTparameters import Z_DIMS, IMAGESIZE, BATCH_SIZE, EPOCHS, ANNEAL_EPOCHS, L_RATE, BATCH_LOGGING_INTERVAL, MULTIPLIER_IMAGE, MULTIPLIER_LABEL, N_MINI_BATCHES
from MNISTparameters import train_loader, test_loader

def elbo_loss(recon_image, image, recon_label, label, mean, logvar,
              lambda_image=1.0, lambda_label=1.0, anneal_factor=1):
    image_bce = 0
    if recon_image is not None and image is not None:
        image_bce = torch.sum(binary_cross_entropy_of_logits(
            recon_image.view(-1, IMAGESIZE),
            image.view(-1, IMAGESIZE)), dim=1)

    label_bce = 0
    if recon_label is not None and label is not None:
        label_bce = torch.sum(cross_entropy_of_logits(recon_label, label), dim=1)

    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)
    ELBO = torch.mean(lambda_image * image_bce + lambda_label * label_bce + anneal_factor * KLD)
    return ELBO


def binary_cross_entropy_of_logits(input, target):
    return torch.clamp(input, 0) - input * target + torch.log(1 + torch.exp(-torch.abs(input)))


def cross_entropy_of_logits(input, target):
    log_input = F.log_softmax(input, dim=1)
    y_onehot = Variable(torch.zeros(input.shape))
    y_onehot = y_onehot.scatter(1, target.unsqueeze(1), 1)
    return -y_onehot * log_input

def train(epoch):
    model.train()
    total_loss = 0
    total_examples = 0

    for batch_idx, (image, label) in enumerate(train_loader):
        # Linearly increase from 0 to 1 on each epoch
        anneal_factor = min(1.0, float(epoch - 1) / ANNEAL_EPOCHS)

        image = Variable(image)
        label = Variable(label)
        batch_size = len(image)

        # Refresh
        optimizer.zero_grad()
        recon_image_1, recon_label_1, mean_1, logvar_1 = model(image, label)
        recon_image_2, recon_label_2, mean_2, logvar_2 = model(image)
        recon_image_3, recon_label_3, mean_3, logvar_3 = model(label=label)
        # Compute ELBO
        joint_loss = elbo_loss(recon_image_1, image, recon_label_1, label, mean_1, logvar_1,
                               lambda_image=MULTIPLIER_IMAGE, lambda_label=MULTIPLIER_LABEL, anneal_factor=anneal_factor)
        image_loss = elbo_loss(recon_image_2, image, None, None, mean_2, logvar_2,
                               lambda_image=MULTIPLIER_IMAGE, lambda_label=MULTIPLIER_LABEL, anneal_factor=anneal_factor)
        label_loss  = elbo_loss(None, None, recon_label_3, label, mean_3, logvar_3,
                               lambda_image=MULTIPLIER_IMAGE, lambda_label=MULTIPLIER_LABEL, anneal_factor=anneal_factor)
        train_loss = joint_loss + image_loss + label_loss
        total_loss += train_loss.item() * batch_size
        total_examples += batch_size
        train_loss.backward()
        optimizer.step()
        if batch_idx % BATCH_LOGGING_INTERVAL == 0:
            print('Epoch: {} [{:5}/{}]      Loss: {:11.6f}       Annealing-Factor: {:.5f}'.format(
                epoch, batch_idx * len(image), len(train_loader.dataset), total_loss / total_examples, anneal_factor))

    print('######## Epoch: {}\tLoss: {:.6f} ########'.format(epoch, total_loss / total_examples))


def test(epoch):
    model.eval()
    total_loss = 0
    total_examples = 0

    for batch_idx, (image, label) in enumerate(test_loader):

        with torch.no_grad():
            image = Variable(image)
            label  = Variable(label)
        batch_size = len(image)

        recon_image_1, recon_label_1, mean_1, logvar_1 = model(image, label)
        recon_image_2, recon_label_2, mean_2, logvar_2 = model(image)
        recon_image_3, recon_label_3, mean_3, logvar_3 = model(label=label)

        joint_loss = elbo_loss(recon_image_1, image, recon_label_1, label, mean_1, logvar_1)
        image_loss = elbo_loss(recon_image_2, image, None, None, mean_2, logvar_2)
        label_loss = elbo_loss(None, None, recon_label_3, label, mean_3, logvar_3)
        test_loss = joint_loss + image_loss + label_loss
        total_loss += test_loss.item() * batch_size
        total_examples += batch_size


    print('######## Test Loss: {} ########'.format(total_loss / total_examples))
    return total_loss / total_examples

def save_model():
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, os.path.join(folder, './bestmodel'))
    print('Model saved!')


# Save the model every 5 epochs
if __name__ == "__main__":
    model = MVAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=L_RATE)

    # Directory to save results
    folder = './models'
    if not os.path.isdir(folder):
        os.mkdir(folder)

    # Train
    train(1)
    test_loss = test(1)
    best_loss = test_loss
    save_model()
    for epoch in range(2, EPOCHS + 1):
        train(epoch)
        test_loss = test(epoch)
        if test_loss < best_loss:
            best_loss = test_loss
            save_model()
