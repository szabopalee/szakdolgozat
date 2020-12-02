import numpy as np
import gym
import time
from gym.spaces import Box, Discrete

import os
import os.path as osp, time, atexit, os

import scipy
from scipy import signal

from skimage.color import rgb2gray

import warnings

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical 		# lehet, hogy nem kell?

def setup_logger_kwargs(exp_name):
    ymd_time = time.strftime("%Y-%m-%d_")
    relpath = ''.join([ymd_time, exp_name])

    data_dir = osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))),'data')
    logger_kwargs = dict(output_dir=osp.join(data_dir, relpath), 
                         exp_name=exp_name)
    return logger_kwargs

class Logger:
    def __init__(self, output_dir=None, output_fname='progress.txt', exp_name=None):
        self.output_dir = output_dir or "/tmp/experiments/%i"%int(time.time())
        if osp.exists(self.output_dir):
            print("Warning: Log dir %s already exists! Storing info there anyway."%self.output_dir)
        else:
            os.makedirs(self.output_dir)
        self.output_file = open(osp.join(self.output_dir, output_fname), 'w')
        atexit.register(self.output_file.close)
        print("Logging data to %s"%self.output_file.name)
        self.first_row=True
        self.log_headers = []
        self.log_current_row = {}
        self.exp_name = exp_name

    def log(self, msg):
        print(msg)

    def log_tabular(self, key, val):
        if self.first_row:
            self.log_headers.append(key)
        else:
            assert key in self.log_headers, "Trying to introduce a new key %s that you didn't include in the first iteration"%key
        assert key not in self.log_current_row, "You already set %s this iteration. Maybe you forgot to call dump_tabular()"%key
        self.log_current_row[key] = val

    def save_state(self, state_dict, itr=None):
        fname = 'vars.pkl' if itr is None else 'vars%d.pkl'%itr
        try:
            joblib.dump(state_dict, osp.join(self.output_dir, fname))
        except:
            self.log('Warning: could not pickle state_dict.')
        if hasattr(self, 'pytorch_saver_elements'):
            self._pytorch_simple_save(itr)

    def setup_pytorch_saver(self, what_to_save):
        self.pytorch_saver_elements = what_to_save

    def _pytorch_simple_save(self, itr=None):
        assert hasattr(self, 'pytorch_saver_elements'), \
            "First have to setup saving with self.setup_pytorch_saver"
        fpath = 'pyt_save'
        fpath = osp.join(self.output_dir, fpath)
        fname = 'model' + ('%d'%itr if itr is not None else '') + '.pt'
        fname = osp.join(fpath, fname)
        os.makedirs(fpath, exist_ok=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.save(self.pytorch_saver_elements, fname)

    def dump_tabular(self):
        vals = []
        key_lens = [len(key) for key in self.log_headers]
        max_key_len = max(15,max(key_lens))
        keystr = '%'+'%d'%max_key_len
        fmt = "| " + keystr + "s | %15s |"
        n_slashes = 22 + max_key_len
        print("-"*n_slashes)
        for key in self.log_headers:
            val = self.log_current_row.get(key, "")
            valstr = "%8.3g"%val if hasattr(val, "__float__") else val
            print(fmt%(key, valstr))
            vals.append(val)
        print("-"*n_slashes, flush=True)
        if self.output_file is not None:
            if self.first_row:
                self.output_file.write("\t".join(self.log_headers)+"\n")
            self.output_file.write("\t".join(map(str,vals))+"\n")
            self.output_file.flush()
        self.log_current_row.clear()
        self.first_row=False

class EpochLogger(Logger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_dict = dict()

    def store(self, **kwargs):
        for k,v in kwargs.items():
            if not(k in self.epoch_dict.keys()):
                self.epoch_dict[k] = []
            self.epoch_dict[k].append(v)

    def log_tabular(self, key, val=None, with_min_and_max=False, average_only=False):
        if val is not None:
            super().log_tabular(key,val)
        else:
            v = self.epoch_dict[key]
            vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape)>0 else v
            #stats = mpi_statistics_scalar(vals, with_min_and_max=with_min_and_max)
            mean = np.sum(vals) / len(vals)
            std = np.sum((vals - mean)**2) / len(vals)
            Max = np.max(vals)
            Min = np.min(vals)
            super().log_tabular(key if average_only else 'Average' + key, mean)
            if not(average_only):
                super().log_tabular('Std'+key, std)
            if with_min_and_max:
                super().log_tabular('Max'+key, Max)
                super().log_tabular('Min'+key, Min)
        self.epoch_dict[key] = []

    def get_stats(self, key):
        v = self.epoch_dict[key]
        vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape)>0 else v
        mean = np.sum(vals) / len(vals)
        std = np.sum((vals - mean)**2) / len(vals)
        return mean, std

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

class PPOBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        print(rews.shape)
        print(vals.shape)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = scipy.signal.lfilter([1], [1, float(-self.gamma * self.lam)], deltas[::-1], axis=0)[::-1]
        # the next line computes rewards-to-go, to be targets for the value function
        rews2 = rews[1:]
        self.ret_buf[path_slice] = scipy.signal.lfilter([1], [1, float(-self.gamma)], rews2[::-1], axis=0)[::-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0

        adv_buf_np = np.array(self.adv_buf, dtype=np.float32)
        adv_mean = np.sum(adv_buf_np) / len(adv_buf_np)
        adv_std = np.sum((adv_buf_np - adv_mean)**2) / len(adv_buf_np)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf, adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}

class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

class GaussianActor(Actor):

    def __init__(self, act_dim):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        
        self.mu_net = nn.Sequential(
            nn.Conv2d(1,6,4,2),
            nn.ReLU(),
            nn.Conv2d(6,16,4,2),
            nn.ReLU(),
            nn.Conv2d(16,32,4,2),
            nn.ReLU(),
            nn.Conv2d(32,64,4,2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, act_dim)
        )

    def _distribution(self, obs):
        mu = self.mu_net(obs)			# mean
        std = torch.exp(self.log_std)	# standard deviation
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.v_net = nn.Sequential(
            nn.Conv2d(1,6,4,2),
            nn.ReLU(),
            nn.Conv2d(6,16,4,2),
            nn.ReLU(),
            nn.Conv2d(16,32,4,2),
            nn.ReLU(),
            nn.Conv2d(32,64,4,2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, obs):
       return torch.squeeze(self.v_net(obs), -1)

class ActorCritic(nn.Module):
    def __init__(self, action_space):
        super().__init__()

        self.pi = GaussianActor(action_space.shape[0])

        self.value  = Critic()

    def step(self, obs):
        obs = torch.unsqueeze(obs, 0)                  # obs megfelelo formara hozasa
        obs = torch.unsqueeze(obs, 0).view(1,1,96,96)
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            action = pi.sample()
            logp_action = self.pi._log_prob_from_distribution(pi, action)
            value = self.value(obs)
        return action.numpy(), value.numpy(), logp_action.numpy()

    def act(self, obs):
        return self.step(obs)[0]

def PPO(env_fn, steps_per_epoch=4000, epochs=30, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=45, train_v_iters=45, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=2):

    logger = EpochLogger(**logger_kwargs)

    np.random.seed()

    env = env_fn()
    obs_dim = (96, 96)       # env.observation_space.shape (ha rgb adatot használnánk és nem grayscale-t)
    act_dim = env.action_space.shape

    ac = ActorCritic(env.action_space)

    # Training folytatasahoz
    #ac = torch.load('./data/ppo/ppo_s0/pyt_save/model.pt')

    # PPO buffer
    buf = PPOBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)

    # Fuggveny a policy loss kiszamitasara
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = ac.pi(obs.view(4000,1,96,96), act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Fuggveny a value loss szamitasra
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.value(obs.view(4000,1,96,96)) - ret)**2).mean()

    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.value.parameters(), lr=vf_lr)

    logger.setup_pytorch_saver(ac)

    def update():
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Policy training (gradient descent)
        for i in range(train_pi_iters):
            print('Policy training with gradient descent %d/%d'%(i+1, train_pi_iters))
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = pi_info['kl']
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
            loss_pi.backward()
            pi_optimizer.step()

        logger.store(StopIter=i)

        # Value function tanulas
        for i in range(train_v_iters):
            print('Value function learning %d/%d'%(i+1, train_v_iters))
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            vf_optimizer.step()

        # Log
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))

    start_time = time.time()
    obs, ep_ret, ep_len = env.reset(), 0, 0
    obs = rgb2gray(obs)

    # Main loop: tapasztalatgyujtes + update/log
    for epoch in range(epochs):
        for t in range(steps_per_epoch):
            action, value, logp = ac.step(torch.as_tensor(obs.copy(), dtype=torch.float32))
            
            #env.render()
            next_obs, reward, done, _ = env.step(action[0])
            next_obs = rgb2gray(next_obs)
            ep_ret += reward
            ep_len += 1

            buf.store(obs, action, reward, value[0], logp[0])
            logger.store(VVals=value[0])

            obs = next_obs

            timeout = ep_len == max_ep_len
            terminal = done or timeout
            epoch_ended = t == steps_per_epoch-1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                if timeout or epoch_ended:
                    _, value, _ = ac.step(torch.as_tensor(obs.copy(), dtype=torch.float32))
                else:
                    value = [0]
                buf.finish_path(value[0])   # advantage számítás
                if terminal:
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                obs, ep_ret, ep_len = env.reset(), 0, 0
                obs = rgb2gray(obs)

        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

        update()

        # Log
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CarRacing-v0')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    logger_kwargs = setup_logger_kwargs(args.exp_name)

    PPO(lambda : gym.make(args.env), gamma=args.gamma, steps_per_epoch=args.steps, epochs=args.epochs, logger_kwargs=logger_kwargs)