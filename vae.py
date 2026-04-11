import gymnasium as gym
import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn

from models.decoder import RewardDecoder
from models.encoder import RNNEncoder
from utils.storage_vae import RolloutStorageVAE

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VaribadVAE:
    """
    VAE of VariBAD:
    - has an encoder and reward decoder
    - can compute the ELBO loss
    - can update the VAE (encoder+decoder)
    """
    
    def __init__(self, args, get_iter_idx):
        self.args = args
        self.get_iter_idx = get_iter_idx
        
        # initialise the encoder
        self.encoder = RNNEncoder(
            args=self.args,
            hidden_size=self.args.encoder_gru_hidden_size,
            latent_dim=self.args.latent_dim,
            action_dim=self.args.action_dim,
            action_embed_dim=self.args.action_embedding_size,
            state_dim=self.args.state_dim,
            state_embed_dim=self.args.state_embedding_size,
            reward_size=1,
            reward_embed_size=self.args.reward_embedding_size,
        ).to(device)
        
        # initialise reward decoder
        self.reward_decoder = RewardDecoder(
            layers=self.args.reward_decoder_layers,
            latent_dim=self.args.latent_dim,
            num_states=self.args.num_states,
        ).to(device)
        
        if self.args.disable_decoder:
            self.reward_decoder = None
        
        # initialise rollout storage for the VAE update (this differs from the data that the on-policy RL algorithm uses)
        self.rollout_storage = RolloutStorageVAE(num_processes=self.args.num_processes,
                                                max_trajectory_len=self.args.max_trajectory_len,
                                                max_num_rollouts=self.args.size_vae_buffer,
                                                state_dim=self.args.state_dim,
                                                action_dim=self.args.action_dim
                                                )
        
        # initalise optimiser for the encoder and decoders
        decoder_params = []
        if not self.args.disable_decoder:
            if self.args.decode_reward:
                decoder_params.extend(self.reward_decoder.parameters())
        self.optimiser_vae = torch.optim.Adam([*self.encoder.parameters(), *decoder_params], lr=self.args.lr_vae)
    
    def compute_rew_reconstruction_loss(self, latent, next_obs, reward, return_predictions=False):
        """ Compute reward reconstruction loss. (No reduction of loss along batch dimension is done here; sum/avg has to be done outside) """
        rew_pred = self.reward_decoder(latent)
        if self.args.rew_pred_type == 'categorical':
            rew_pred = F.softmax(rew_pred, dim=-1)
        elif self.args.rew_pred_type == 'bernoulli':
            rew_pred = torch.sigmoid(rew_pred)
        
        env = gym.make(self.args.env_name)
        env_task = env.unwrapped if hasattr(env, 'unwrapped') else env
        state_indices = env_task.task_to_id(next_obs).to(device)
        if state_indices.dim() < rew_pred.dim():
            state_indices = state_indices.unsqueeze(-1)
        rew_pred = rew_pred.gather(dim=-1, index=state_indices)
        rew_target = (reward == 1).float()
        if self.args.rew_pred_type in ['categorical', 'bernoulli']:
            loss_rew = F.binary_cross_entropy(rew_pred, rew_target, reduction='none').mean(dim=-1)
        else:
            raise NotImplementedError
        
        if return_predictions:
            return loss_rew, rew_pred
        else:
            return loss_rew
    
    def compute_kl_loss(self, latent_mean, latent_logvar):
        prior_mean = torch.zeros_like(latent_mean[:1])
        prior_logvar = torch.zeros_like(latent_logvar[:1])
        
        all_means = torch.cat([prior_mean, latent_mean], dim=0)
        all_logvars = torch.cat([prior_logvar, latent_logvar], dim=0)
        
        q_t = torch.distributions.Normal(all_means[1:], torch.exp(0.5 * all_logvars[1:]))
        q_prev = torch.distributions.Normal(all_means[:-1], torch.exp(0.5 * all_logvars[:-1]))
        # https://arxiv.org/pdf/1811.09975.pdf
        # KL(N(mu,E)||N(m,S)) = 0.5 * (log(|S|/|E|) - K + tr(S^-1 E) + (m-mu)^T S^-1 (m-mu)))
        return torch.distributions.kl.kl_divergence(q_t, q_prev).sum(dim=-1)
    
    def compute_loss(self, latent_mean, latent_logvar, vae_next_obs, vae_rewards, trajectory_lens):
        """
        Computes the VAE loss for the given data.
        Handles variable trajectory lengths via masks over ELBO/reconstruction terms.
        """
        # Keep max padded length, then mask out invalid terms per trajectory.
        max_traj_len = np.max(trajectory_lens)
        latent_mean = latent_mean[:max_traj_len + 1]
        latent_logvar = latent_logvar[:max_traj_len + 1]
        vae_next_obs = vae_next_obs[:max_traj_len]
        vae_rewards = vae_rewards[:max_traj_len]
        
        # take one sample for each ELBO term
        latent_samples = torch.distributions.Normal(latent_mean, torch.exp(0.5 * latent_logvar)).rsample()
        
        num_elbos = latent_samples.shape[0]
        num_decodes = vae_next_obs.shape[0]
        
        # expand the state/rew/action inputs to the decoder (to match size of latents)
        # shape will be: [num tasks in batch] x [num elbos] x [len trajectory (reconstrution loss)] x [dimension]
        dec_next_obs = vae_next_obs.unsqueeze(0).expand((num_elbos, *vae_next_obs.shape))
        dec_rewards = vae_rewards.unsqueeze(0).expand((num_elbos, *vae_rewards.shape))
        
        # expand the latent (to match the number of state/rew/action inputs to the decoder)
        # shape will be: [num tasks in batch] x [num elbos] x [len trajectory (reconstrution loss)] x [dimension]
        dec_embedding = latent_samples.unsqueeze(0).expand((num_decodes, *latent_samples.shape)).transpose(1, 0)
        
        # compute reconstruction loss for this trajectory (for each timestep that was encoded, decode everything and sum it up)
        # shape: [num_elbo_terms] x [num_reconstruction_terms] x [num_trajectories]
        rew_reconstruction_loss = self.compute_rew_reconstruction_loss(dec_embedding, dec_next_obs, dec_rewards)
        lens = torch.as_tensor(trajectory_lens, device=device, dtype=torch.long)
        decode_mask = (torch.arange(num_decodes, device=device).unsqueeze(1) < lens.unsqueeze(0))
        elbo_mask = (torch.arange(num_elbos, device=device).unsqueeze(1) <= lens.unsqueeze(0))
        valid_pair_mask = elbo_mask.unsqueeze(1) & decode_mask.unsqueeze(0)  # [E, D, B]

        rew_reconstruction_loss = rew_reconstruction_loss * valid_pair_mask.float()
        # We sum ELBO and reconstruction terms per task, then average across tasks.
        rew_reconstruction_loss = rew_reconstruction_loss.sum(dim=(0, 1)).mean()
        
        # compute the KL term for each ELBO term of the current trajectory
        # shape: [num_elbo_terms] x [num_trajectories]
        if not self.args.disable_kl_term:
            kl_loss = self.compute_kl_loss(latent_mean, latent_logvar)
            kl_loss = (kl_loss * elbo_mask.float()).sum(dim=0).mean()
        else:
            kl_loss = 0
        return rew_reconstruction_loss, kl_loss

    def compute_vae_loss(self, update=False, pretrain_index=None):
        """ Returns the VAE loss """
        if not self.rollout_storage.ready_for_update():
            return 0
        
        if self.args.disable_decoder and self.args.disable_kl_term:
            return 0
        
        # get a mini-batch. vae_next_obs will be of size: max trajectory len x num trajectories x dimension of observations
        vae_next_obs, vae_actions, vae_rewards, trajectory_lens = self.rollout_storage.get_batch(batchsize=self.args.vae_batch_num_trajs)
        # pass through encoder (outputs will be: (max_traj_len+1) x number of rollouts x latent_dim -- includes the prior!)
        _, latent_mean, latent_logvar, _ = self.encoder(actions=vae_actions,
                                                        states=vae_next_obs,
                                                        rewards=vae_rewards,
                                                        hidden_state=None,
                                                        return_prior=True,
                                                        detach_every=self.args.tbptt_stepsize if hasattr(self.args, 'tbptt_stepsize') else None,
                                                        )
        rew_reconstruction_loss, kl_loss = self.compute_loss(
            latent_mean, latent_logvar, vae_next_obs, vae_rewards, trajectory_lens
        )
        
        # VAE loss = KL loss + reward reconstruction
        # take average (this is the expectation over p(M))
        elbo_loss = (rew_reconstruction_loss + self.args.kl_weight * kl_loss).mean()
        
        if update:
            self.optimiser_vae.zero_grad()
            elbo_loss.backward()
            # clip gradients
            if self.args.encoder_max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.encoder.parameters(), self.args.encoder_max_grad_norm)
            if self.args.decoder_max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.reward_decoder.parameters(), self.args.decoder_max_grad_norm)
            # update
            self.optimiser_vae.step()
        
        return elbo_loss
