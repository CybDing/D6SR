"""
Soft Actor-Critic (SAC) with automatic entropy tuning.

Minimal pure-PyTorch implementation for continuous action spaces.
Twin Q-networks, squashed Gaussian policy, circular replay buffer.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def peek_checkpoint(path):
    """Read obs_dim, act_dim, hidden from a saved checkpoint without building networks."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    obs_dim = ckpt.get('obs_dim')
    act_dim = ckpt.get('act_dim')
    # Fallback: infer from weight shapes
    if obs_dim is None:
        obs_dim = next(v.shape[1] for k, v in ckpt['actor'].items()
                       if k == 'trunk.0.weight')
    if act_dim is None:
        act_dim = next(v.shape[0] for k, v in ckpt['actor'].items()
                       if k == 'mean_head.weight')
    hidden = ckpt.get('hidden')
    return {'obs_dim': obs_dim, 'act_dim': act_dim, 'hidden': hidden}
LOG_STD_MIN, LOG_STD_MAX = -20, 2


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(256, 256)):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GaussianPolicy(nn.Module):
    """Squashed Gaussian policy: tanh(mu + std * noise)."""

    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev, act_dim)
        self.log_std_head = nn.Linear(prev, act_dim)

    def forward(self, obs):
        h = self.trunk(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs):
        """Sample action and compute log_prob (with tanh squashing correction)."""
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        x = dist.rsample()  # reparameterized sample
        action = torch.tanh(x)
        # Log prob with tanh squashing: log pi(a|s) = log pi(u|s) - sum(log(1 - tanh^2(u)))
        log_prob = dist.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, mean

    def deterministic(self, obs):
        mean, _ = self.forward(obs)
        return torch.tanh(mean)


class TwinQ(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.q1 = MLP(obs_dim + act_dim, 1, hidden)
        self.q2 = MLP(obs_dim + act_dim, 1, hidden)

    def forward(self, obs, action):
        sa = torch.cat([obs, action], dim=-1)
        return self.q1(sa), self.q2(sa)


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, capacity=500000):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)

    def push(self, obs, action, reward, next_obs, done):
        i = self.ptr
        self.obs[i] = obs
        self.act[i] = action
        self.rew[i] = reward
        self.next_obs[i] = next_obs
        self.done[i] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.obs[idx]).to(DEVICE),
            torch.from_numpy(self.act[idx]).to(DEVICE),
            torch.from_numpy(self.rew[idx]).unsqueeze(1).to(DEVICE),
            torch.from_numpy(self.next_obs[idx]).to(DEVICE),
            torch.from_numpy(self.done[idx]).unsqueeze(1).to(DEVICE),
        )

    def __len__(self):
        return self.size

    def clear(self):
        """Reset buffer, discarding all stored transitions."""
        self.ptr = 0
        self.size = 0


class SAC:
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden=(128, 128),
        lr_actor=3e-4,
        lr_critic=1e-4,
        lr_alpha=3e-4,
        gamma=0.99,
        tau=0.005,
        buffer_size=500000,
        batch_size=128,
        target_entropy=None,
        grad_clip=1.0,
        utd_ratio=2,
        init_alpha=1.0,
    ):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.act_dim = act_dim
        self.grad_clip = grad_clip
        self.utd_ratio = utd_ratio

        # Networks (128,128 sufficient for 15-dim obs / 5-dim act)
        self.actor = GaussianPolicy(obs_dim, act_dim, hidden).to(DEVICE)
        self.critic = TwinQ(obs_dim, act_dim, hidden).to(DEVICE)
        self.critic_target = TwinQ(obs_dim, act_dim, hidden).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Separate learning rates: critic slower for stability
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Auto-tune entropy coefficient
        self.target_entropy = target_entropy if target_entropy is not None else -act_dim
        # init_alpha < 1 starts exploration gently — critical for residual RL
        # on strong actuators, where the default alpha=1 entropy phase floods
        # the buffer with catastrophic actions before the critic can object.
        self.log_alpha = torch.tensor(
            [float(np.log(init_alpha))], requires_grad=True, device=DEVICE)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr_alpha)

        # Replay buffer
        self.buffer = ReplayBuffer(obs_dim, act_dim, buffer_size)

        # Logging
        self.train_step = 0
        self._update_counter = 0

    @property
    def alpha(self):
        return self.log_alpha.exp().item()

    def select_action(self, obs, deterministic=False):
        """Select action given observation (numpy). Returns numpy action in [-1, 1]."""
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)
            if deterministic:
                action = self.actor.deterministic(obs_t)
            else:
                action, _, _ = self.actor.sample(obs_t)
            return action.cpu().numpy().flatten()

    def update(self, update_actor=True):
        """Multiple critic updates per actor update (UTD ratio) with gradient clipping.

        update_actor=False runs only the critic (+ target) updates — used to
        pretrain the critic before the actor starts moving, which shallows the
        early performance dip caused by maximizing an untrained Q."""
        if len(self.buffer) < self.batch_size:
            return {}

        info = {}
        for _ in range(self.utd_ratio):
            obs, act, rew, next_obs, done = self.buffer.sample(self.batch_size)
            alpha = self.log_alpha.exp().detach()

            # --- Critic update (every step) ---
            with torch.no_grad():
                next_act, next_log_prob, _ = self.actor.sample(next_obs)
                q1_targ, q2_targ = self.critic_target(next_obs, next_act)
                q_targ = torch.min(q1_targ, q2_targ) - alpha * next_log_prob
                target = rew + (1.0 - done) * self.gamma * q_targ

            q1, q2 = self.critic(obs, act)
            critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

            self.critic_optim.zero_grad()
            critic_loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
            self.critic_optim.step()

            # Soft target update after each critic step
            for p, p_targ in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_targ.data.mul_(1 - self.tau).add_(self.tau * p.data)

            info['critic_loss'] = critic_loss.item()

        if not update_actor:
            self.train_step += 1
            return info

        # --- Actor update (once per call) ---
        obs_actor = self.buffer.sample(self.batch_size)[0]
        new_act, log_prob, _ = self.actor.sample(obs_actor)
        q1_new, q2_new = self.critic(obs_actor, new_act)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (alpha * log_prob - q_new).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optim.step()

        # --- Alpha update ---
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.train_step += 1
        info['actor_loss'] = actor_loss.item()
        info['alpha'] = self.alpha
        return info

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha.data,
            'actor_optim': self.actor_optim.state_dict(),
            'critic_optim': self.critic_optim.state_dict(),
            'alpha_optim': self.alpha_optim.state_dict(),
            'train_step': self.train_step,
            'obs_dim': self.actor.trunk[0].in_features,
            'act_dim': self.act_dim,
            'hidden': tuple(m.out_features for m in self.actor.trunk if isinstance(m, nn.Linear)),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        # Detect architecture from checkpoint and rebuild if needed
        ckpt_hidden = ckpt.get('hidden')
        if ckpt_hidden is None:
            # Old checkpoint without saved config — infer from weight shapes
            ckpt_hidden = tuple(
                v.shape[0] for k, v in ckpt['actor'].items()
                if k.startswith('trunk.') and k.endswith('.weight')
            )
        cur_hidden = tuple(m.out_features for m in self.actor.trunk if isinstance(m, nn.Linear))
        if ckpt_hidden != cur_hidden:
            print(f"  Rebuilding networks: checkpoint hidden={ckpt_hidden} != current {cur_hidden}")
            obs_dim = ckpt.get('obs_dim', self.actor.trunk[0].in_features)
            act_dim = ckpt.get('act_dim', self.act_dim)
            self.actor = GaussianPolicy(obs_dim, act_dim, ckpt_hidden).to(DEVICE)
            self.critic = TwinQ(obs_dim, act_dim, ckpt_hidden).to(DEVICE)
            self.critic_target = TwinQ(obs_dim, act_dim, ckpt_hidden).to(DEVICE)
            self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
            self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=1e-4)

        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.critic_target.load_state_dict(ckpt['critic_target'])
        self.log_alpha.data = ckpt['log_alpha']
        self.train_step = ckpt.get('train_step', 0)
