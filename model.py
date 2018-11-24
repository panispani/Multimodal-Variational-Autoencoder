import torch
import torch.nn as nn
from torch.autograd import Variable
from MNISTparameters import ImageEncoder, ImageDecoder, LabelEncoder, LabelDecoder, Z_DIMS

class MVAE(nn.Module):
    def __init__(self):
        super(MVAE, self).__init__()
        self.image_encoder = ImageEncoder()
        self.image_decoder = ImageDecoder()
        self.label_encoder = LabelEncoder()
        self.label_decoder = LabelDecoder()

    def reparametrize(self, means, logvar):
        if self.training:
            eps = Variable(torch.Tensor(means.shape).normal_())
            return means + eps * logvar.mul(0.5).exp_()
        else:
            return means

    def prior_expert(self, size):
        # N(0, 1)
        means = Variable(torch.zeros(size))
        logvar = Variable(torch.zeros(size))
        return means, logvar

    # Mix gaussians
    def product_of_experts(self, means, logvar):
        P = 1.0 / torch.exp(logvar)
        Psum = P.sum(dim=0)
        prod_means = torch.sum(means * P, dim=0) / Psum
        prod_logvar = torch.log(1.0 / Psum)
        return prod_means, prod_logvar

    def forward(self, image=None, label=None):
        means, logvar = self.encode_modalities(image, label)
        z = self.reparametrize(means, logvar)
        # Reconstruct
        decoded_img = self.image_decoder(z)
        decoded_lbl = self.label_decoder(z)
        return decoded_img, decoded_lbl, means, logvar

    def encode_modalities(self, image=None, label=None):
        if (image is not None):
            batch_size = image.size(0)
        else:
            batch_size = label.size(0)

        # Initialization
        means, logvar = self.prior_expert((1, batch_size, Z_DIMS))

        # Support for weak supervision setting
        if image is not None:
            img_mean, img_logvar = self.image_encoder(image)
            means = torch.cat((means, img_mean.unsqueeze(0)))
            logvar = torch.cat((logvar, img_logvar.unsqueeze(0)))

        if label is not None:
            lbl_mean, lbl_logvar = self.label_encoder(label)
            means = torch.cat((means, lbl_mean.unsqueeze(0)))
            logvar = torch.cat((logvar, lbl_logvar.unsqueeze(0)))

        # Combine the gaussians
        means, logvar = self.product_of_experts(means, logvar)
        return means, logvar
