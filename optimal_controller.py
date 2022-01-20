import os
import torch
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from controller.environment.env import Env
from tianshou.policy import DQNPolicy
from tianshou.utils import BasicLogger
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
from tianshou.trainer import offpolicy_trainer
from tianshou.data import Collector, VectorReplayBuffer,PrioritizedVectorReplayBuffer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='Pid')
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--eps-test', type=float, default=0.05)
    parser.add_argument('--eps-train', type=float, default=0.1)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-3)  # learning rate
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--target-update-freq', type=int, default=320)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--step-per-epoch', type=int, default=100)
    parser.add_argument('--step-per-collect', type=int, default=10)
    parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--hidden-sizes', type=int,
                        nargs='*', default=[128, 128, 128])
    parser.add_argument('--training-num', type=int, default=3)
    parser.add_argument('--test-num', type=int, default=5)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--prioritized-replay',
                        action="store_true", default=False)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--beta', type=float, default=0.4)
    parser.add_argument('--watch', default=False, action='store_true',
                        help='no training, '
                             'watch the play of pre-trained models')
    parser.add_argument('--resume-path', type=str, default='log/Pid/dqn/policy3000.pth',
                        help='the path of agent pth file '
                             'for resuming from a pre-trained agent')
    parser.add_argument(
        '--save-buffer-name', type=str,
        default="./expert_DQN_controller.pkl")
    parser.add_argument(
        '--device', type=str,
        default='cuda:4' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    return args


def run_ql(args=get_args()):
    """
    设置学习策略
    :param conditions:
    :param table:
    :param min_block_size:
    :param args:
    :return:
    """
    """1. Make an Environment"""
    torch.set_num_threads(10)  # for poor CPU
    env = Env()
    # env = gym.make(args.task)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n

    """2. Setup Multi-environment Wrapper"""
    # you can also use tianshou.env.SubprocVectorEnv
    train_envs = DummyVectorEnv(
        [lambda: Env() for _ in range(args.training_num)])
    test_envs = DummyVectorEnv(
        [lambda: Env() for _ in range(args.test_num)])
    # set random seed for own environment
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    """3. Build the Network"""
    # pretrained
    net = Net(args.state_shape,args.action_shape, hidden_sizes=args.hidden_sizes,
              device=args.device).to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)

    """4. Setup Policy"""
    policy = DQNPolicy(
        model=net,optim=optim,discount_factor=args.gamma,estimation_step=args.n_step,
        target_update_freq=args.target_update_freq)

    # buffer
    if args.prioritized_replay:
        buf = PrioritizedVectorReplayBuffer(
            args.buffer_size, buffer_num=len(train_envs),
            alpha=args.alpha, beta=args.beta)
    else:
        buf = VectorReplayBuffer(args.buffer_size, buffer_num=len(train_envs))

    """5. Setup Collector"""
    # collector
    train_collector = Collector(policy, train_envs, buf, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # policy.set_eps(1)
    train_collector.collect(n_step=args.batch_size * args.training_num,random=True)

    """6. Train Policy with a Trainer"""
    # log
    log_path = os.path.join(args.logdir, args.task, 'dqn')
    # def save_fn(policy):
    #     # 保存神经网络的训练模型参数
    #     torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))
    # def stop_fn(mean_rewards):
    #     return mean_rewards >= 50000
    # def train_fn(epoch, env_step):
    #     # eps annnealing, just a demo
    #     if env_step <= 10000:
    #         policy.set_eps(args.eps_train)
    #     elif env_step <= 50000:
    #         eps = args.eps_train - (env_step - 10000) / \
    #               40000 * (0.9 * args.eps_train)
    #         policy.set_eps(eps)
    #     else:
    #         policy.set_eps(0.1 * args.eps_train)
    # def test_fn(epoch, env_step):强化学习训练重复的action
    #     policy.set_eps(args.eps_test)
    writer = SummaryWriter(log_path)
    logger = BasicLogger(writer)

    def train_fn(epoch, env_step):
        # eps annnealing, just a demo
        if env_step <= 10000:
            # episode
            policy.set_eps(args.eps_train)
        elif env_step <= 50000:
            eps = args.eps_train - (env_step - 10000) / \
                40000 * (0.9 * args.eps_train)
            policy.set_eps(eps)
        else:
            policy.set_eps(0.1 * args.eps_train)

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)
    # trainer
    # 跑完整个训练数据集，叫做一个epoch。一个epoch包含多个episode。一个episode完成一次模型验证，保存最优模型，简单来说就是多少个step进行一次模型验证。
    offpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=3, step_per_epoch=500, step_per_collect=10,
        update_per_step=0.1, episode_per_test=20, batch_size=64,
        train_fn=train_fn,test_fn=test_fn,
        stop_fn=lambda mean_rewards: mean_rewards >= 0.5,logger=logger)
    torch.save(policy.state_dict(),  args.resume_path)

    # result = offpolicy_trainer(
    #     policy, train_collector, test_collector, args.epoch,
    #     args.step_per_epoch, args.step_per_collect, args.test_num,
    #     args.batch_size, update_per_step=args.update_per_step, train_fn=train_fn,
    #     test_fn=test_fn, stop_fn=stop_fn, save_fn=save_fn, logger=logger)

    ## self-defined Policy Trainer :pre-collect at least 5000 transitions with random action before training
    # train_collector.collect(n_step=5000, random=True)
    # policy.set_eps(0.1)
    # for i in range(int(1e6)):  # total step
    #     collect_result = train_collector.collect(n_step=10)
    #     if collect_result['rews'].mean() >= env.spec.reward_threshold or i % 1000 == 0:
    #         policy.set_eps(0.05)
    #         result = test_collector.collect(n_episode=100)
    #         if result['rews'].mean() >= env.spec.reward_threshold:
    #             print(f'Finished training! Test mean returns: {result["rews"].mean()}')
    #             break
    #         else:
    #             policy.set_eps(0.1)
    #     losses = policy.update(64, train_collector.buffer)

    # assert stop_fn(result['best_reward'])

    """7. Watch the Agent's Performance"""
    # save buffer in pickle format, for imitation learning unittest
    # VectorReplayBuffer：It is used for storing transition from different environments yet keeping the order of time.
    # buf = VectorReplayBuffer(args.buffer_size, buffer_num=len(test_envs))
    # collector = Collector(policy, test_envs, buf)
    # collector.collect(n_step=args.buffer_size)

    """8. Save Policy"""
    # pickle.dump(buf, open(args.save_buffer_name, "wb"))
    print('end.')


def get_agents(args=get_args()) :
    env = Env()
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    net = Net(args.state_shape, args.action_shape, hidden_sizes=args.hidden_sizes,
              device=args.device).to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    agent_learn = DQNPolicy(
        model=net, optim=optim, discount_factor=args.gamma, estimation_step=args.n_step,
        target_update_freq=args.target_update_freq)
    agent_learn.load_state_dict(torch.load(args.resume_path))
    return agent_learn, optim

def watch(args=get_args()):
    env = Env()
    env.seed(1)
    policy, optim = get_agents(args)
    policy.eval()
    collector = Collector(policy, env)
    result = collector.collect(n_episode=1)
    rews, lens = result["rews"], result["lens"]
    # Final reward: 0.21710652942719805, length: 4.0
    # Average query cost: 255.10589229654673

    # Final reward: 0.14353004530958552, length: 2.0
    # Average query cost: 254.71172422120262
    print(f"Final reward: {rews.mean()}, length: {lens.mean()}")
    print(env.io_cost)
    print(env.action_list)
    print(f"Average query cost: {sum(env.io_cost)/sum([sql['feature'].frequency for sql in  env.w.sql_list])}")

if __name__ == '__main__':
    # run_ql()
    watch()