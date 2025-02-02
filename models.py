import torch
import math

import torch.nn as nn
import torch.nn.functional as F

from typing import List
from typing import Tuple


class VarianceScheduler:
    def __init__(self, beta_start: int=0.0001, beta_end: int=0.02, num_steps: int=1000, interpolation: str='linear') -> None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.num_steps = num_steps

        # find the beta valuess by linearly interpolating from start beta to end beta
        if interpolation == 'linear':
            # TODO: complete the linear interpolation of betas here
            self.betas = torch.linspace(beta_start, beta_end, num_steps).to(device)
        elif interpolation == 'quadratic':
            # TODO: complete the quadratic interpolation of betas here
            self.betas = (torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_steps) ** 2).to(device)
        else:
            raise Exception('[!] Error: invalid beta interpolation encountered...')
        

        # TODO: add other statistics such alphas alpha_bars and all the other things you might need here
        
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(device)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.sqrt_alpha_bars)

    def add_noise(self, x:torch.Tensor, time_step:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x.device

        # TODO: sample a random noise
        noise = torch.randn_like(x, device=device)

        # TODO: construct the noisy sample
        sqrt_alpha_bar = self.sqrt_alpha_bars[time_step].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bars = self.sqrt_one_minus_alpha_bars[time_step].view(-1, 1, 1, 1)
        noisy_input = sqrt_alpha_bar * x + sqrt_one_minus_alpha_bars * noise

        return noisy_input, noise


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int) -> None:
      super().__init__()

      self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        # TODO: compute the sinusoidal positional encoding of the time
        device = time.device
        half_dim = self.dim // 2
        
        #w_k = 1 / (torch.pow(10000.0, 2 * torch.arange(half_dim, device=device, dtype=torch.float32) / self.dim))
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * - embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        if self.dim % 2 == 1:
            embeddings = torch.nn.functional.pad(embeddings, (0, 1, 0, 0))

        return embeddings


class UNet(nn.Module):
    def __init__(self, in_channels: int=1, 
                 down_channels: List=[64, 128, 128, 128, 128], 
                 up_channels: List=[128, 128, 128, 128, 64], 
                 time_emb_dim: int=128,
                 num_classes: int=10) -> None:
        super().__init__()

        # NOTE: You can change the arguments received by the UNet if you want, but keep the num_classes argument
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.time_emb_dim = time_emb_dim
        self.down_channels, self.up_channels = down_channels, up_channels
        self.height, self.width = 32, 32

        # TODO: time embedding layer
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.LeakyReLU()
        )

        # TODO: define the embedding layer to compute embeddings for the labels
        self.class_emb = nn.Embedding(num_classes, time_emb_dim)

        # define your network architecture here
        self.down_blocks = nn.ModuleList()
        self.time_down_layers, self.conv_down_layer = nn.ModuleList(), nn.ModuleList()
        in_ch = in_channels
        for i in range(1, len(down_channels) + 1):
            self.down_blocks.append(nn.Sequential(
                nn.Conv2d(in_ch, down_channels[i-1], kernel_size=3, padding=1),
                nn.GroupNorm(8, down_channels[i-1]),
                nn.LeakyReLU()
            ))
            self.time_down_layers.append(self.time_embedding_layer(down_channels[i-1]))
            self.conv_down_layer.append(self.conv_block(down_channels[i-1], down_channels[i-1]))
            in_ch = down_channels[i-1]
        
        self.pool = nn.MaxPool2d(2)


        self.up_blocks = nn.ModuleList()
        self.time_up_layers, self.conv_up_layer = nn.ModuleList(), nn.ModuleList()
        in_ch = down_channels[-1]
        for i in range(1, len(up_channels) + 1):
            self.up_blocks.append(nn.Sequential(
                nn.Conv2d(in_ch + up_channels[i-1], up_channels[i-1], kernel_size=3, padding=1),
                nn.GroupNorm(8, up_channels[i-1]),
                nn.LeakyReLU()
            ))
            self.time_up_layers.append(self.time_embedding_layer(up_channels[i-1]))
            self.conv_up_layer.append(self.conv_block(up_channels[i-1], up_channels[i-1]))
            in_ch = up_channels[i-1]

        self.upsp = nn.Upsample(scale_factor=2, mode='nearest')

        self.final = nn.Conv2d(up_channels[-1], in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        device = x.device
        
        # TODO: embed time
        t = self.time_mlp(timestep.to(device))
        
        # TODO: handle label embeddings if labels are avaialble
        l = self.class_emb(label.to(label))
        
        emb = torch.cat([t, l], dim = 1)
        
        # TODO: compute the output of your network
        residuals = []
        out = x
        #print(x.shape)
        for i in range(1, len(self.down_channels) + 1):
            down = self.down_blocks[i-1]
            te, cb = self.time_down_layers[i-1], self.conv_down_layer[i-1]
            
            out = down(out)

            t = te(emb)
            out = out + t[:, :, None, None]
            
            out = cb(out)
            out, res = self.pool(out), out
            
            residuals.append(res)
        
        for i in range(1, len(self.up_channels) + 1):
            up = self.up_blocks[i-1]
            te, cb = self.time_up_layers[i-1], self.conv_up_layer[i-1]

            res = residuals.pop()

            out = self.upsp(out)
            out = torch.cat([out, res], dim=1)

            out = up(out)

            t = te(emb)
            out = out + t[:, :, None, None]

            out = cb(out)

        out = self.final(out)
        return out
    
    def time_embedding_layer(self, dim_out):
        return nn.Sequential(
            nn.Linear(self.time_emb_dim * 2, dim_out),
            nn.GroupNorm(1, dim_out),
            nn.LeakyReLU()
        )
    
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.LeakyReLU()
        )


class VAE(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 height: int=32, 
                 width: int=32, 
                 mid_channels: List=[32, 32, 32], 
                 latent_dim: int=32, 
                 num_classes: int=10) -> None:
        
        super().__init__()

        self.height = height
        self.width = width
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # NOTE: self.mid_size specifies the size of the image [C, H, W] in the bottleneck of the network
        self.mid_size = [mid_channels[-1], height // (2 ** len(mid_channels)), width // (2 ** len(mid_channels))]
        
        # NOTE: You can change the arguments of the VAE as you please, but always define self.latent_dim, self.num_classes, self.mid_size
        
        # TODO: handle the label embedding here
        self.class_emb = nn.Linear(num_classes, height * width)
        self.data_emb = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # TODO: define the encoder part of your network
        encoder_layer = []
        in_ch = in_channels + 1
        for out_ch in mid_channels:
            encoder_layer.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU()])
            in_ch = out_ch
        #vae has 4 downsampling layers, for lddpm needs to have 3
        self.encoder = nn.Sequential(*encoder_layer)
        
        flattened_size = self.mid_size[0] * self.mid_size[1] * self.mid_size[2]

        # TODO: define the network/layer for estimating the mean
        self.mean_net = nn.Linear(flattened_size, latent_dim)
        
        # TODO: define the networklayer for estimating the log variance
        self.logvar_net = nn.Linear(flattened_size, latent_dim)

        # TODO: define the decoder part of your network
        self.decoder_input = nn.Linear(latent_dim + num_classes, flattened_size)

        decoder_layer = []
        for i in range(len(mid_channels)-1, 0, -1):
            decoder_layer.extend([
                nn.ConvTranspose2d(mid_channels[i], mid_channels[i-1], kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(mid_channels[i-1]),
                nn.LeakyReLU()])
        self.decoder = nn.Sequential(*decoder_layer)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(mid_channels[0], mid_channels[0], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(mid_channels[0]),
            nn.LeakyReLU(),
            nn.Conv2d(mid_channels[0], in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.lddpm = False
    
    def encode(self, x: torch.Tensor) -> List[torch.tensor]:
        out = self.encoder(x)
        
        out_ = torch.flatten(out, start_dim=1)

        mean = self.mean_net(out_)
        logvar = self.logvar_net(out_)

        return [out, mean, logvar]

    def decode(self, sample: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # TODO: use you decoder to decode a given sample and their corresponding labels
        x = torch.cat([sample, labels], dim=1)

        x = self.decoder_input(x)
        
        x = x.view(x.size(0), self.mid_size[0], self.mid_size[1], self.mid_size[2])
        
        out = self.decoder(x)
        out = self.final_layer(out)
        
        return out
    
    def forward(self, x: torch.Tensor, label: torch.Tensor) -> List[torch.Tensor]:
        # TODO: compute the output of the network encoder
        label_oh = F.one_hot(label, num_classes=self.num_classes).float()
        emb_class = self.class_emb(label_oh)
        emb_class = emb_class.view(-1, self.height, self.width).unsqueeze(1)
        emb_input = self.data_emb(x)
        
        # TODO: estimating mean and logvar
        emb_x = torch.cat([emb_input, emb_class], dim=1)
        _, mean, logvar = self.encode(emb_x)
        
        # TODO: computing a sample from the latent distribution
        sample = self.reparameterize(mean, logvar)
        
        # TODO: decoding the sample
        out = self.decode(sample, label_oh)
        
        return out, mean, logvar

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # TODO: implement the reparameterization trick: sample = noise * std + mean
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std)
        sample = (noise * std) + mean
        
        return sample
    
    @staticmethod
    def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # TODO: compute the binary cross entropy between the pred (reconstructed image) and the traget (ground truth image)
        loss = F.binary_cross_entropy(pred, target, reduction='sum')
        #loss = F.binary_cross_entropy_with_logits(pred, target, reduction='sum')
        return loss
       
    @staticmethod
    def kl_loss(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # TODO: compute the KL divergence
        kl_div = torch.mean(-0.5 * torch.sum(1 + logvar - (mean **2) - logvar.exp(), dim=1), dim =0)
        return kl_div

    @torch.no_grad()
    def generate_sample(self, num_samples: int, device=torch.device('cuda'), labels: torch.Tensor=None):
        if labels is not None:
            assert len(labels) == num_samples, 'Error: number of labels should be the same as number of samples!'
            labels = labels.to(device)
        else:
            # randomly consider some labels
            labels = torch.randint(0, self.num_classes, [num_samples,], device=device)

        # TODO: sample from standard Normal distrubution
        noise = torch.randn(num_samples, self.latent_dim, device=device)

        # TODO: decode the noise based on the given labels
        labels_oh = F.one_hot(labels, num_classes=self.num_classes).float()
        out = self.decode(noise, labels_oh)

        return out


class DDPM(nn.Module):
    def __init__(self, network: nn.Module, var_scheduler: VarianceScheduler) -> None:
        super().__init__()

        self.var_scheduler = var_scheduler
        self.network = network
        self.num_steps = self.var_scheduler.num_steps

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: uniformly sample as many timesteps as the batch size
        t = torch.randint(0, self.var_scheduler.num_steps, (x.shape[0],), device=x.device)

        # TODO: generate the noisy input
        noisy_input, noise = self.var_scheduler.add_noise(x, t)

        # TODO: estimate the noise
        estimated_noise = self.network(noisy_input, t, label)
        
        # TODO: compute the loss (either L1, or L2 loss)
        loss = F.mse_loss(noise, estimated_noise) #F.l1_loss

        return loss

    @torch.no_grad()
    def recover_sample(self, noisy_sample: torch.Tensor, estimated_noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        # TODO: implement the sample recovery strategy of the DDPM
        beta = self.var_scheduler.betas[timestep].view(-1, 1, 1, 1)
        alpha = self.var_scheduler.alphas[timestep].view(-1, 1, 1, 1)
        alpha_bar = self.var_scheduler.alpha_bars[timestep].view(-1, 1, 1, 1)
        
        sample = torch.sqrt(1 / alpha) * (noisy_sample - (((1 - alpha) * estimated_noise) / torch.sqrt(1 - alpha_bar)))

        if timestep.min() > 0:
            noise = torch.randn_like(noisy_sample)
            sample += torch.sqrt(beta) * noise

        return sample

    @torch.no_grad()
    def generate_sample(self, num_samples: int, device: torch.device=torch.device('cuda'), labels: torch.Tensor=None):
        if labels is not None and self.network.num_classes is not None:
            assert len(labels) == num_samples, 'Error: number of labels should be the same as number of samples!'
            labels = labels.to(device)
        elif labels is None and self.network.num_classes is not None:
            labels = torch.randint(0, self.network.num_classes, [num_samples,], device=device)
        else:
            labels = None

        # TODO: apply the iterative sample generation of the DDPM
        sample = torch.randn(num_samples, self.network.in_channels, self.network.height, self.network.width, device=device)
        
        for t in reversed(range(self.var_scheduler.num_steps)):
            timestep = torch.full((num_samples,), t, device=device, dtype=torch.long)
            estimated_noise = self.network(sample, timestep, labels)
            sample = self.recover_sample(sample, estimated_noise, timestep)

        return sample


class DDIM(nn.Module):
    def __init__(self, network: nn.Module, var_scheduler: VarianceScheduler) -> None:
        super().__init__()

        self.var_scheduler = var_scheduler
        self.network = network
    
    def forward(self, x: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: uniformly sample as many timesteps as the batch size
        t = torch.randint(0, self.var_scheduler.num_steps, (x.shape[0],), device=x.device)

        # TODO: generate the noisy input
        noisy_input, noise = self.var_scheduler.add_noise(x, t)

        # TODO: estimate the noise
        estimated_noise = self.network(noisy_input, t, label)

        # TODO: compute the loss
        loss = F.mse_loss(estimated_noise, noise)

        return loss
    
    @torch.no_grad()
    def recover_sample(self, noisy_sample: torch.Tensor, estimated_noise: torch.Tensor, timestep: torch.Tensor, eta: float=0.0) -> torch.Tensor:
        # TODO: apply the sample recovery strategy of the DDIM
        alpha_bar_t = self.var_scheduler.alpha_bars[timestep].view(-1, 1, 1, 1)
        alpha_bar_prev = self.var_scheduler.alpha_bars[timestep - 1].view(-1, 1, 1, 1) if timestep.min() > 0 \
                        else torch.ones_like(alpha_bar_t).view(-1, 1, 1, 1)

        sigma = eta * torch.sqrt(((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * ((1 - alpha_bar_t) / alpha_bar_prev))
        
        pred_x0 = (noisy_sample - (torch.sqrt(1 - alpha_bar_t) * estimated_noise)) / torch.sqrt(alpha_bar_t)
        dir_xt = torch.sqrt(1 - alpha_bar_prev - (sigma ** 2)) * estimated_noise

        sample = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt

        if eta > 0:
            noise = torch.randn_like(noisy_sample)
            sample += sigma * noise

        return sample
    
    @torch.no_grad()
    def generate_sample(self, num_samples: int, device: torch.device=torch.device('cuda'), labels: torch.Tensor=None, eta: float=0.0):
        if labels is not None and self.network.num_classes is not None:
            assert len(labels) == num_samples, 'Error: number of labels should be the same as number of samples!'
            labels = labels.to(device)
        elif labels is None and self.network.num_classes is not None:
            labels = torch.randint(0, self.network.num_classes, [num_samples,], device=device)
        else:
            labels = None

        # TODO: apply the iterative sample generation of DDIM (similar to DDPM)
        sample = torch.randn(num_samples, self.network.in_channels, self.network.height, self.network.width, device=device)

        for t in reversed(range(self.var_scheduler.num_steps)):
            timestep = torch.full((num_samples,), t, device=device, dtype=torch.long)
            estimated_noise = self.network(sample, timestep, labels)
            sample = self.recover_sample(sample, estimated_noise, timestep, eta)

        return sample
