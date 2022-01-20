import os
import torch
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tianshou.policy import PPOPolicy
from tianshou.utils import BasicLogger
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
from tianshou.trainer import onpolicy_trainer
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.utils.net.discrete import Actor, Critic
from controller.environment.env_v2 import Env2


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='Pid')
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--step-per-epoch', type=int, default=10000)
    parser.add_argument('--episode-per-collect', type=int, default=20)
    parser.add_argument('--repeat-per-collect', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[64,64,64])
    parser.add_argument('--training-num', type=int, default=5)
    parser.add_argument('--test-num', type=int, default=3)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str,
        default='cuda:4' if torch.cuda.is_available() else 'cpu')
    # ppo special
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.0)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--rew-norm', type=int, default=1)
    parser.add_argument('--dual-clip', type=float, default=None)
    parser.add_argument('--value-clip', type=int, default=1)
    parser.add_argument('--resume-path', type=str, default='log/Pid/ppo/policy3000.pth',
                        help='the path of agent pth file '
                             'for resuming from a pre-trained agent')
    args = parser.parse_known_args()[0]
    return args


def run_ppo(args=get_args()):
    torch.set_num_threads(10)  # we just need only one thread for NN
    env = Env2()
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    # you can also use tianshou.env.SubprocVectorEnv
    # train_envs = gym.make(args.task)
    train_envs = DummyVectorEnv(
        [lambda: Env2() for _ in range(args.training_num)])
    # test_envs = gym.make(args.task)
    test_envs = DummyVectorEnv(
        [lambda: Env2() for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # pretrained
    net = Net(args.state_shape, hidden_sizes=args.hidden_sizes,
              device=args.device)
    actor = Actor(net, args.action_shape, device=args.device).to(args.device)
    critic = Critic(net,device=args.device).to(args.device)
    # orthogonal initialization
    for m in list(actor.modules()) + list(critic.modules()):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    optim = torch.optim.Adam(set(
        actor.parameters()).union(critic.parameters()), lr=args.lr)
    dist=torch.distributions.Categorical
    policy = PPOPolicy(
        actor, critic, optim, dist,
        max_grad_norm=args.max_grad_norm,
        eps_clip=args.eps_clip,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        gae_lambda=args.gae_lambda,
        reward_normalization=args.rew_norm,
        dual_clip=args.dual_clip,
        value_clip=args.value_clip,
        action_space=env.action_space)
    # collector
    train_collector = Collector(
        policy, train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True)
    test_collector = Collector(policy, test_envs)
    # log
    log_path = os.path.join(args.logdir, args.task, 'ppo')
    writer = SummaryWriter(log_path)
    logger = BasicLogger(writer)

    # def save_fn(policy):
    #     torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(mean_rewards):
        return mean_rewards >= 1

    # trainer
    result = onpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.repeat_per_collect, args.test_num, args.batch_size,
        episode_per_collect=args.episode_per_collect, stop_fn=stop_fn,
        logger=logger)
    torch.save(policy.state_dict(), args.resume_path)
    # assert stop_fn(result['best_reward'])
    if __name__ == '__main__':
        pprint.pprint(result)
        # Let's watch its performance!
        env = Env2()
        env.seed(1)
        policy.eval()
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")
        print('io_cost : ', env.io_cost)
        print('action list : ', env.action_list)
        print(f"Average query cost: {sum(env.io_cost) / sum([sql['feature'].frequency for sql in env.w.sql_list])}")
def run_pretrained_model(args=get_args()):
    env = Env2()
    env.seed(2)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    # pretrained
    net = Net(args.state_shape, hidden_sizes=args.hidden_sizes,
              device=args.device)
    actor = Actor(net, args.action_shape, device=args.device).to(args.device)
    critic = Critic(net, device=args.device).to(args.device)
    # orthogonal initialization
    for m in list(actor.modules()) + list(critic.modules()):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    optim = torch.optim.Adam(set(
        actor.parameters()).union(critic.parameters()), lr=args.lr)
    dist = torch.distributions.Categorical
    policy = PPOPolicy(
        actor, critic, optim, dist,
        max_grad_norm=args.max_grad_norm,
        eps_clip=args.eps_clip,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        gae_lambda=args.gae_lambda,
        reward_normalization=args.rew_norm,
        dual_clip=args.dual_clip,
        value_clip=args.value_clip,
        action_space=env.action_space)
    policy.load_state_dict(torch.load((args.resume_path)))
    policy.eval()
    collector=Collector(policy,env)
    result=collector.collect(n_episode=1)
    rews, lens = result["rews"], result["lens"]
    # Epoch: 50
    # Final reward: 0.09647231143803948, length: 4.0
    # io_cost : [888278.0, 1605844.0, 1067045.0, 122706.0]
    # action list : [90, 282, 474, 852]
    # Average query cost: 222.4023786524994

    # Epoch: 100
    # Final reward: 0.06414208471697519, length: 4.0
    # io_cost : [1479314.0, 1163028.0, 715591.0, 334705.0]
    # action list : [153, 298, 443, 635]
    # Average query cost: 222.9315382757788

    # Epoch: 100  ， 3000query-steam.csv
    # Final reward: 0.06205486659681806, length: 2.0
    # io_cost: [5666667.0, 2105087.0]
    # action list: [293, 578]
    # Average query cost: 230.56795324413326

    # Final reward: 5.636998928974383, length: 214.0
    # io_cost: [9634388.0, 68883.0, 88338.0, 65703.0, 89100.0, ... 10078.0, 7114.0]
    # action list: [26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 81, 83, 85, 87, 89, 91, 93, 95, 120, 122,
    #        124, 126, 128, 130, 132, 134, 136, ..., 496, 498]
    # Average query cost: 503.5351134846462

    # Final reward: 7.078512456182265, length: 498.0
    # action list: [4, 5, 7, 8, 9, 10, 12, 13, 17, 26, 28, 29, 30
    # Average query cost: 257.3725856697819
    print(f"Final reward: {rews.mean()}, length: {lens.mean()}")
    print('io_cost : ',env.io_cost)
    print('action list : ',env.action_list)
    print(f"Average query cost: {sum(env.io_cost) / sum([sql['feature'].frequency for sql in env.w.sql_list])}")
if __name__ == '__main__':
    run_ppo()
    # run_pretrained_model()