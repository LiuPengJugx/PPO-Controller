import os
import torch
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tianshou.policy import DDPGPolicy
from tianshou.utils import BasicLogger
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
from tianshou.trainer import offpolicy_trainer
from tianshou.exploration import GaussianNoise
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.utils.net.continuous import Actor, Critic
from my_actor import MyActor
from controller.environment.multenv_v2 import Multenv2
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='Pid')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--exploration-noise', type=float, default=0.15)
    parser.add_argument('--epoch', type=int, default=3)
    parser.add_argument('--step-per-epoch', type=int, default=1000)
    parser.add_argument('--step-per-collect', type=int, default=8)
    parser.add_argument('--update-per-step', type=float, default=0.125)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[128, 128])
    parser.add_argument('--training-num', type=int, default=2)
    parser.add_argument('--test-num', type=int, default=3)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--rew-norm', action="store_true", default=False)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--resume-path', type=str, default='log/Pid/ddpg/policy3000.pth',
                        help='the path of agent pth file '
                             'for resuming from a pre-trained agent')
    parser.add_argument(
        '--device', type=str,
        default='cuda:3' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    return args


def run_ddpg(args=get_args()):
    torch.set_num_threads(10)  # we just need only one thread for NN
    # get state and action's shape
    env = Multenv2()
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    # make environment(train_env and test_env)
    # you can also use tianshou.env.SubprocVectorEnv
    # train_envs = gym.make(args.task)
    train_envs = DummyVectorEnv(
        [lambda: Multenv2() for _ in range(args.training_num)])
    # test_envs = gym.make(args.task)
    test_envs = DummyVectorEnv(
        [lambda: Multenv2() for _ in range(args.test_num)])

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # pretrained: define the network(actor and critic)
    net = Net(args.state_shape, hidden_sizes=args.hidden_sizes,
              device=args.device)
    actor = MyActor(net, args.action_shape,max_action=1,
                  device=args.device).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net = Net(args.state_shape, args.action_shape,
              hidden_sizes=args.hidden_sizes, concat=True, device=args.device)
    critic = Critic(net, device=args.device).to(args.device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    policy = DDPGPolicy(
        actor, actor_optim, critic, critic_optim,
        tau=args.tau, gamma=args.gamma,
        exploration_noise=GaussianNoise(sigma=args.exploration_noise),
        reward_normalization=args.rew_norm,
        estimation_step=args.n_step, action_space=env.action_space)
    # collector
    train_collector = Collector(
        policy, train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True)
    test_collector = Collector(policy, test_envs)
    # log
    log_path = os.path.join(args.logdir, args.task, 'ddpg')
    writer = SummaryWriter(log_path)
    logger = BasicLogger(writer)

    def stop_fn(mean_rewards):
        return mean_rewards >= 1

    # trainer
    result = offpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.step_per_collect, args.test_num, args.batch_size,
        update_per_step=args.update_per_step,logger=logger,stop_fn=stop_fn)
    # assert stop_fn(result['best_reward'])
    torch.save(policy.state_dict(), args.resume_path)

    # test the trained policy file!
    if __name__ == '__main__':
        pprint.pprint(result)
        # Let's watch its performance!
        env = Multenv2()
        policy.eval()
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")

def run_pretrained_model(args=get_args()):
    env = Multenv2()
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    # pretrained: define the network(actor and critic)
    net = Net(args.state_shape, hidden_sizes=args.hidden_sizes,
              device=args.device)
    actor = Actor(net, args.action_shape, max_action=1,
                  device=args.device).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net = Net(args.state_shape, args.action_shape,
              hidden_sizes=args.hidden_sizes, concat=True, device=args.device)
    critic = Critic(net, device=args.device).to(args.device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    policy = DDPGPolicy(
        actor, actor_optim, critic, critic_optim,
        tau=args.tau, gamma=args.gamma,
        exploration_noise=GaussianNoise(sigma=args.exploration_noise),
        reward_normalization=args.rew_norm,
        estimation_step=args.n_step, action_space=env.action_space)
    policy.load_state_dict(torch.load(args.resume_path))
    policy.eval()
    collector=Collector(policy,env)
    result=collector.collect(n_episode=1)
    rews, lens = result["rews"], result["lens"]
    # Epoch: 20
    # Final reward: 0.14329073670657616, length: 2.0
    # [2784592.0, 1365077.0]
    # [[257, 140], [698, 120]]
    # Average query cost: 250.52336392175803

    # Epoch: 50
    # Final reward: 0.13984537022439325, length: 2.0
    # io_cost: [2430311.0, 1656680.0]
    # action list: [[228, 151], [695, 487]]
    # Average query cost: 246.73937454721082

    # Epoch: 100
    # Final reward: -0.04382986598033567, length: 15.0
    # io_cost: [1943753.0, 1309161.0, 11292.0, 653912.0, 8352.0, 7230.0, 5465.0, 15570.0, 15595.0]
    # action list: [[174, 140], [207, 207], [353, 154], [354, 333], [480, 183], [481, 51], [482, 177], [483, 55], [484, 168],
    #        [485, 72], [486, 134], [487, 157], [488, 102], [489, 154], [490, 103]]
    # Average querycost: 239.69632938903646

    # Epoch: 500
    # Final reward: 0.02486363331460302, length: 11.0
    # io_cost: [13219201.0, 797568.0, 991101.0, 712622.0, 905605.0, 810094.0, 655305.0, 430583.0, 604321.0, 432963.0,
    #           144377.0]
    # action list: [[38, 29], [88, 50], [138, 50], [188, 50], [238, 50], [288, 50], [338, 50], [388, 50], [438, 50], [488, 50],
    #        [538, 50]]
    # Average query cost: 584.5939771547248

    print(f"Final reward: {rews.mean()}, length: {lens.mean()}")
    print('io_cost : ',env.io_cost)
    print('action list : ',env.action_list)
    print(f"Average query cost: {sum(env.io_cost) / sum([sql['feature'].frequency for sql in env.w.sql_list])}")
if __name__ == '__main__':
    run_ddpg()
    # run_pretrained_model()