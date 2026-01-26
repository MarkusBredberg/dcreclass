
import torch
import torch.nn as nn


#
# Custom Loss Functions
#

def d_loss_bce(real_pred, fake_pred, real_label_val=0.9, fake_label_val=0.0):
    """Standard BCEWithLogitsLoss for the Discriminator."""
    bce = nn.BCEWithLogitsLoss()
    # real_label_val can be 1.0 or 0.9 for label smoothing
    real_labels = torch.full_like(real_pred, real_label_val)
    fake_labels = torch.full_like(fake_pred, fake_label_val)
    loss_real = bce(real_pred, real_labels)
    loss_fake = bce(fake_pred, fake_labels)
    return loss_real + loss_fake

def g_loss_bce(fake_pred, real_label_val=0.9):
    """Standard BCEWithLogitsLoss for the Generator."""
    bce = nn.BCEWithLogitsLoss()
    # For generator, we want D(fake) to be close to 1
    labels = torch.full_like(fake_pred, real_label_val)
    return bce(fake_pred, labels)


def d_loss_mse(real_pred, fake_pred):
    """MSE loss on raw logits (uncommon but sometimes used)."""
    mse = nn.MSELoss()
    # Typically for MSE-based DCGAN, we set real=1, fake=0
    real_labels = torch.ones_like(real_pred)
    fake_labels = torch.zeros_like(fake_pred)
    loss_real = mse(real_pred, real_labels)
    loss_fake = mse(fake_pred, fake_labels)
    return loss_real + loss_fake

def g_loss_mse(fake_pred, target_val=None):
    """MSE loss for the generator to push D(fake) to 1."""
    mse = nn.MSELoss()
    if target_val is None:
        labels = torch.ones_like(fake_pred)
    else:
        labels = torch.full_like(fake_pred, target_val)
    return mse(fake_pred, labels)


def d_loss_lsgan(real_pred, fake_pred):
    """
    LSGAN Discriminator loss:
      0.5 * E[(D(real) - 1)^2] + 0.5 * E[(D(fake) - 0)^2]
    """
    real_loss = 0.5 * torch.mean((real_pred - 1.0)**2)
    fake_loss = 0.5 * torch.mean((fake_pred - 0.0)**2)
    return real_loss + fake_loss

def g_loss_lsgan(fake_pred):
    """
    LSGAN Generator loss:
      0.5 * E[(D(fake) - 1)^2]
    """
    return 0.5 * torch.mean((fake_pred - 1.0)**2)


def d_loss_hinge(real_pred, fake_pred):
    """
    Hinge loss for Discriminator:
      E[relu(1 - D(real))] + E[relu(1 + D(fake))]
    Usually D(real) wants to be > +1, D(fake) wants to be < -1
    """
    loss_real = torch.mean(nn.ReLU()(1.0 - real_pred))
    loss_fake = torch.mean(nn.ReLU()(1.0 + fake_pred))
    return loss_real + loss_fake

def g_loss_hinge(fake_pred):
    """
    Hinge loss for Generator:
      -E[D(fake)]
    (Generator wants D(fake) to be as large as possible)
    """
    return -torch.mean(fake_pred)


#
# For WGAN (without gradient penalty):
#  - D/critic tries to maximize: E[D(real)] - E[D(fake)]
#  - G tries to maximize E[D(fake)] or equivalently minimize -E[D(fake)]
# Usually you'd add weight clipping or a gradient penalty for stability.
# This minimal example does not include gradient penalty.
#
def d_loss_wgan(real_pred, fake_pred):
    """
    WGAN critic loss = - ( E[D(real)] - E[D(fake)] ) = ( E[D(fake)] - E[D(real)] )
    Because we usually do .backward() on a "loss" we want to minimize.
    So we minimize negative of the objective we want to maximize.
    """
    return -(torch.mean(real_pred) - torch.mean(fake_pred))

def g_loss_wgan(fake_pred):
    """Generator wants to maximize D(fake), so we minimize -mean(D(fake))."""
    return -torch.mean(fake_pred)


def pick_gan_loss_functions(gen_loss='MSE', disc_loss='BCE', real_label_val=1.0, fake_label_val=0.0):
    if gen_loss == "BCE":
        gen_loss_fn = lambda fake: g_loss_bce(fake, real_label_val)
    elif gen_loss == "MSE":
        gen_loss_fn = g_loss_mse
    elif gen_loss == "LSGAN":
        gen_loss_fn = g_loss_lsgan
    elif gen_loss == "HINGE":
        gen_loss_fn = g_loss_hinge
    elif gen_loss == "WGAN":
        gen_loss_fn = g_loss_wgan
    else:
        raise ValueError(f"Unknown generator loss type: {gen_loss}")
    
    if disc_loss == "BCE":
        disc_loss_fn = lambda real, fake: d_loss_bce(real, fake, real_label_val, fake_label_val)
    elif disc_loss == "MSE":
        disc_loss_fn = d_loss_mse
    elif disc_loss == "LSGAN":
        disc_loss_fn = d_loss_lsgan
    elif disc_loss == "HINGE":
        disc_loss_fn = d_loss_hinge
    elif disc_loss == "WGAN":
        disc_loss_fn = d_loss_wgan
    else:
        raise ValueError(f"Unknown discriminator loss type: {disc_loss}")
    
    return  gen_loss_fn, disc_loss_fn,
