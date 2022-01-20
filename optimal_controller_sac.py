import os
import torch
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tianshou.utils import BasicLogger
from tianshou.env import SubprocVectorEnv
from tianshou.utils.net.common import Net
from tianshou.policy import DiscreteSACPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.data import Collector, VectorReplayBuffer
# from env_v2 import Env2
from controller.environment.env_v2_skew import Env2


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='Pid')
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--buffer-size', type=int, default=30000)
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--alpha-lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--auto-alpha', action="store_true", default=False)
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--step-per-epoch', type=int, default=20000)
    parser.add_argument('--step-per-collect', type=int, default=10)
    parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[128, 128])
    parser.add_argument('--training-num', type=int, default=3)
    parser.add_argument('--test-num', type=int, default=5)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.0)
    parser.add_argument('--rew-norm', action="store_true", default=False)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument(
        '--device', type=str,
        default='cuda:4' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume-path', type=str, default='log/Pid/sac/policy3000_v2.pth',
                        help='the path of agent pth file '
                             'for resuming from a pre-trained agent')
    args = parser.parse_known_args()[0]
    return args


def run_sac(args=get_args()):
    torch.set_num_threads(10)  # we just need only one thread for NN
    env = Env2()
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    train_envs = SubprocVectorEnv(
        [lambda: Env2() for _ in range(args.training_num)])
    # test_envs = gym.make(args.task)
    test_envs = SubprocVectorEnv(
        [lambda: Env2() for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # pretrained
    net = Net(args.state_shape, hidden_sizes=args.hidden_sizes,
              device=args.device)
    actor = Actor(net, args.action_shape,
                  softmax_output=False, device=args.device).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net_c1 = Net(args.state_shape, hidden_sizes=args.hidden_sizes,
                 device=args.device)
    critic1 = Critic(net_c1, last_size=args.action_shape,
                     device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    net_c2 = Net(args.state_shape, hidden_sizes=args.hidden_sizes,
                 device=args.device)
    critic2 = Critic(net_c2, last_size=args.action_shape,
                     device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    # better not to use auto alpha in CartPole
    if args.auto_alpha:
        target_entropy = 0.98 * np.log(np.prod(args.action_shape))
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)
    policy = DiscreteSACPolicy(
        actor, actor_optim, critic1, critic1_optim, critic2, critic2_optim,
        args.tau, args.gamma, args.alpha, estimation_step=args.n_step,
        reward_normalization=args.rew_norm)
    # collector
    train_collector = Collector(
        policy, train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True)
    test_collector = Collector(policy, test_envs)
    # log
    log_path = os.path.join(args.logdir, args.task, 'sac')
    writer = SummaryWriter(log_path)
    logger = BasicLogger(writer)

    def stop_fn(mean_rewards):
        return mean_rewards >= 10

    # trainer
    result = offpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.step_per_collect, args.test_num,
        args.batch_size, stop_fn=stop_fn,  logger=logger,
        update_per_step=args.update_per_step, test_in_train=False)
    torch.save(policy.state_dict(), args.resume_path)
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
    actor = Actor(net, args.action_shape,
                  softmax_output=False, device=args.device).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net_c1 = Net(args.state_shape, hidden_sizes=args.hidden_sizes,
                 device=args.device)
    critic1 = Critic(net_c1, last_size=args.action_shape,
                     device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    net_c2 = Net(args.state_shape, hidden_sizes=args.hidden_sizes,
                 device=args.device)
    critic2 = Critic(net_c2, last_size=args.action_shape,
                     device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    # better not to use auto alpha in CartPole
    if args.auto_alpha:
        target_entropy = 0.98 * np.log(np.prod(args.action_shape))
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    policy = DiscreteSACPolicy(
        actor, actor_optim, critic1, critic1_optim, critic2, critic2_optim,
        args.tau, args.gamma, args.alpha, estimation_step=args.n_step,
        reward_normalization=args.rew_norm)

    policy.load_state_dict(torch.load((args.resume_path)))
    policy.eval()
    collector=Collector(policy,env)
    result=collector.collect(n_episode=1)
    rews, lens = result["rews"], result["lens"]

    print(f"Final reward: {rews.mean()}, length: {lens.mean()}")
    print('io_cost : ',env.io_cost)
    print('action list : ',env.action_list)
    print(f"Average query cost: {sum(env.io_cost) / sum([sql['feature'].frequency for sql in env.w.sql_list])}")
if __name__ == '__main__':
    run_sac()
    # run_pretrained_model()


# Env2  w1:w1=1:1.5,
# Epoch:10*10000
# Final reward: 29.02755433245335, length: 498.0
# io_cost :  [33360, 7136.0, 7872, 14696.0, 129998.0, 168885.0, 48243.0, 29493.0, 133856.0, 47192.0, 172395.0, 105694.0, 40496.0, 10368, 480, 4800, 36354.0, 34418.0, 42094.0, 44821.0, 58131.0, 26929.0, 64195.0, 17619.0, 7358.0, 27207.0, 2496, 11424, 15459.0, 49864.0, 32217.0, 52288.0, 74433.0, 32202.0, 34694.0, 29266.0, 20205.0, 6146.0, 1104.0, 5600, 4176, 2310.0, 1152, 1957.0, 4800, 5424.0, 43710.0, 50389.0, 24039.0, 65961.0, 42269.0, 20048.0, 44749.0, 28683.0, 7104, 6132.0, 6485.0, 7968, 744.0, 4800, 8362.0, 25540.0, 28138.0, 33472.0, 27691.0, 59366.0, 24911.0, 28567.0, 48961.0, 30768.0, 17446.0, 7922.0, 5248, 13612.0, 4128, 13177.0, 16760.0, 52426.0, 83996.0, 38412.0, 59118.0, 34274.0, 14184.0, 20213.0, 30074.0, 2640, 2157.0, 4704, 3600, 4800, 5848.0, 9076.0, 15999.0, 16950.0, 22037.0, 20601.0, 63517.0, 59058.0, 47492.0, 23149.0, 24495.0, 5579.0, 9956.0, 4953.0, 4800, 2451.0, 4800, 2400, 4800, 7220.0, 28349.0, 70337.0, 52358.0, 24725.0, 24996.0, 43457.0, 21480.0, 10685.0, 15700.0, 3094.0, 3200, 4800, 3824.0, 7662.0, 17609.0, 79702.0, 39565.0, 35438.0, 42587.0, 56514.0, 25864.0, 11410.0, 10864.0, 3200, 4800, 3273.0, 1072.0, 16048.0, 23088, 38023.0, 39718.0, 55195.0, 26993.0, 27447.0, 56108.0, 8560.0, 8430.0, 17158.0, 18796.0, 12720, 4800, 4224.0, 34606.0, 27856.0, 51639.0, 78850.0, 46119.0, 31470.0, 38878.0, 18606.0, 32868.0, 1680, 4800, 8832, 13301.0, 29457.0, 44501.0, 20130.0, 31822.0, 144518.0, 24348.0, 8224.0, 4896.0, 5002.0, 13440, 4800, 26631.0, 27500.0, 38820.0, 43472.0, 35233.0, 47068.0, 41690.0, 26997.0, 30888.0, 15642.0, 22240, 4800, 5073.0, 9227.0, 10384.0, 32412.0, 37396.0, 39154.0, 60216.0, 35484.0, 47241.0, 25889.0, 27172.0, 28493.0, 30957.0, 4968.0, 5472, 3840, 2267.0, 12264.0, 35599.0, 52191.0, 35646.0, 24328.0, 27408.0, 41833.0, 33210.0, 27566.0, 16562.0, 16002.0, 2784, 2880, 4800, 12136.0, 11977.0, 18000.0, 36732.0, 20392.0, 25759.0, 24250.0, 19235.0, 25876.0, 8319.0, 6544.0, 2048.0, 10969.0, 4800, 5550.0, 6685.0, 6971.0, 16224.0, 25123.0, 38255.0, 28833.0, 30572.0, 18783.0, 12534.0, 12460.0, 3556.0, 8296.0, 2880, 4800, 1392.0, 1197.0, 19354.0, 16839.0, 15163.0, 24835.0, 26478.0, 18165.0, 34985.0, 12630.0, 10832, 672, 4992, 2080, 4800, 4157.0, 16480, 10905.0, 26242.0, 17242.0, 34284.0, 36001.0, 20620.0, 21435.0, 21152.0, 7364.0, 1376, 8931.0, 2110.0, 160, 4800, 24190.0, 34144.0, 35445.0, 24111.0, 26294.0, 8780.0, 9780.0, 10759.0, 4192, 3892.0, 2080, 4800, 4864, 5849.0, 20761.0, 15051.0, 39802.0, 18993.0, 21035.0, 16254.0, 9193.0, 30428.0, 15518.0, 14658.0, 3840, 4800, 3360, 1176.0, 6528, 10740.0, 10322.0, 37636.0, 34474.0, 21312, 45500.0, 18009.0, 13368.0, 6722.0, 9600, 960, 4800, 8332.0, 18959.0, 22765.0, 20358.0, 18366.0, 14697.0, 28481.0, 30857.0, 19325.0, 3400.0, 5758.0, 972.0, 4800, 7712.0, 13217.0, 18936.0, 27291.0, 15200, 25397.0, 14826.0, 13535.0, 31551.0, 8627.0, 8912.0, 960, 4800, 3117.0, 4544, 14186.0, 24967.0, 8917.0, 22037.0, 28877.0, 42126.0, 15915.0, 7720.0, 8845.0, 906.0, 0]
# action list :  [[3, 1], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 1], [10, 0], [11, 0], [12, 1], [14, 0], [15, 0], [16, 0], [17, 0], [19, 1], [26, 0], [27, 1], [28, 1], [29, 1], [30, 1], [32, 1], [33, 0], [34, 0], [36, 1], [37, 0], [44, 0], [45, 0], [46, 1], [47, 0], [48, 1], [49, 0], [50, 0], [51, 1], [52, 0], [53, 0], [54, 0], [55, 0], [56, 0], [57, 0], [58, 0], [59, 0], [60, 0], [62, 0], [63, 1], [65, 0], [66, 0], [68, 1], [69, 0], [70, 0], [71, 1], [72, 0], [73, 1], [74, 1], [75, 0], [76, 0], [77, 1], [78, 0], [79, 0], [80, 1], [85, 0], [86, 0], [87, 0], [88, 0], [89, 0], [90, 0], [91, 0], [92, 0], [93, 0], [94, 1], [95, 0], [96, 0], [99, 0], [104, 0], [105, 0], [106, 0], [107, 0], [108, 0], [109, 1], [110, 1], [111, 1], [112, 1], [113, 0], [114, 0], [115, 1], [116, 0], [117, 0], [118, 0], [119, 0], [120, 1], [122, 0], [124, 0], [125, 1], [126, 0], [127, 0], [128, 0], [129, 1], [130, 0], [131, 0], [132, 0], [133, 0], [134, 0], [135, 0], [137, 0], [138, 0], [139, 0], [140, 1], [142, 0], [143, 1], [145, 0], [146, 0], [147, 0], [149, 1], [150, 1], [151, 0], [152, 0], [153, 0], [154, 0], [155, 0], [157, 1], [158, 0], [159, 1], [164, 0], [165, 0], [166, 0], [167, 0], [169, 1], [170, 1], [171, 0], [172, 1], [173, 1], [174, 0], [175, 0], [176, 0], [177, 1], [183, 0], [184, 0], [185, 0], [186, 0], [187, 1], [188, 0], [189, 0], [190, 1], [191, 0], [192, 0], [194, 1], [195, 0], [196, 0], [197, 1], [198, 0], [199, 1], [205, 0], [206, 1], [207, 1], [208, 1], [209, 0], [211, 1], [212, 0], [213, 0], [214, 1], [215, 1], [217, 0], [218, 1], [224, 0], [225, 0], [226, 0], [227, 1], [228, 0], [229, 0], [230, 1], [233, 1], [234, 1], [235, 0], [236, 0], [237, 0], [238, 1], [245, 0], [246, 1], [247, 1], [248, 0], [249, 1], [250, 1], [251, 1], [252, 1], [253, 0], [254, 0], [255, 1], [256, 1], [259, 0], [264, 0], [265, 0], [266, 1], [267, 0], [268, 1], [269, 1], [270, 1], [271, 1], [272, 0], [273, 0], [274, 0], [276, 0], [278, 1], [281, 0], [283, 1], [284, 0], [285, 0], [286, 0], [287, 0], [288, 1], [289, 0], [290, 0], [291, 0], [292, 0], [293, 0], [294, 0], [295, 0], [296, 0], [297, 0], [298, 1], [304, 0], [306, 0], [307, 0], [308, 0], [309, 0], [310, 1], [311, 0], [312, 0], [313, 0], [315, 0], [316, 0], [317, 0], [318, 0], [319, 1], [323, 0], [325, 1], [326, 0], [327, 1], [328, 0], [329, 0], [330, 0], [331, 0], [332, 0], [333, 0], [334, 1], [335, 0], [336, 0], [337, 0], [338, 1], [342, 0], [345, 0], [346, 0], [347, 0], [348, 0], [349, 0], [350, 1], [351, 1], [352, 0], [354, 0], [355, 1], [356, 0], [357, 0], [358, 1], [359, 1], [364, 0], [365, 0], [366, 1], [367, 0], [368, 1], [369, 0], [370, 0], [371, 0], [372, 0], [373, 1], [374, 1], [375, 0], [376, 0], [378, 1], [379, 0], [380, 1], [388, 0], [389, 0], [390, 1], [391, 0], [392, 0], [393, 0], [394, 0], [395, 1], [396, 1], [397, 0], [398, 0], [399, 1], [404, 0], [405, 0], [406, 0], [407, 0], [408, 0], [409, 0], [410, 0], [411, 0], [412, 0], [413, 0], [414, 1], [415, 0], [417, 1], [418, 1], [420, 0], [424, 0], [425, 0], [426, 0], [427, 1], [428, 1], [429, 0], [430, 0], [431, 0], [433, 1], [434, 0], [435, 0], [436, 0], [439, 0], [440, 1], [446, 0], [447, 0], [448, 0], [449, 0], [450, 0], [451, 0], [452, 0], [453, 1], [454, 0], [455, 0], [456, 0], [458, 0], [459, 1], [465, 0], [466, 1], [467, 1], [468, 1], [469, 1], [470, 0], [471, 0], [472, 0], [473, 0], [474, 0], [475, 0], [477, 0], [478, 1], [482, 0], [485, 0], [486, 0], [487, 0], [488, 0], [489, 0], [490, 0], [491, 0], [492, 0], [493, 0], [494, 0], [495, 0], [496, 1]]
# Average query cost: 239.1008455718736


# steps:10* 20000
# Final reward: 21.923700367796627, length: 498.0
# io_cost :  [111680.0, 7872, 14696.0, 129998.0, 168885.0, 48243.0, 29493.0, 133856.0, 47192.0, 166749.0, 5358.0, 140550.0, 40496.0, 10368, 480, 4800, 36354.0, 34418.0, 42094.0, 44821.0, 12164.0, 44631.0, 26897.0, 64195.0, 5549.0, 10931.0, 8816, 10179.0, 6400, 1280, 4800, 2304, 11424, 15459.0, 49864.0, 85790.0, 71734.0, 32202.0, 34694.0, 29266.0, 20205.0, 6146.0, 1104.0, 5600, 4176, 3910.0, 2437.0, 2136.0, 8336.0, 9012.0, 30895.0, 50450.0, 24039.0, 65961.0, 42269.0, 20048.0, 44749.0, 28683.0, 7104, 6132.0, 6485.0, 7968, 744.0, 4800, 8362.0, 25540.0, 28138.0, 33472.0, 27691.0, 59366.0, 24911.0, 28567.0, 48961.0, 30768.0, 17446.0, 5099.0, 2638.0, 4800, 13612.0, 4128, 13177.0, 16760.0, 52426.0, 83996.0, 38412.0, 59118.0, 34274.0, 14184.0, 20213.0, 30074.0, 2640, 2157.0, 4704, 3600, 4800, 5848.0, 9076.0, 15999.0, 16950.0, 22037.0, 20601.0, 63517.0, 59058.0, 47492.0, 23149.0, 24495.0, 5579.0, 3312, 5666.0, 4761.0, 4800, 2451.0, 4800, 2400, 4800, 7220.0, 28349.0, 30292.0, 34075.0, 48080.0, 24725.0, 24996.0, 43457.0, 21480.0, 10685.0, 1052.0, 13368.0, 1548.0, 3200, 4800, 3824.0, 7662.0, 17609.0, 28408.0, 50045.0, 40783.0, 35438.0, 42587.0, 56514.0, 25864.0, 11410.0, 10864.0, 3200, 4800, 3273.0, 1072.0, 16048.0, 23088, 38023.0, 39718.0, 55195.0, 26993.0, 27447.0, 25154.0, 29234.0, 8412.0, 8430.0, 17158.0, 18796.0, 12720, 4800, 30330.0, 27856.0, 51639.0, 38634.0, 37038.0, 48458.0, 31470.0, 38878.0, 18606.0, 26953.0, 5408, 2240, 4800, 8832, 13301.0, 29457.0, 44501.0, 20130.0, 31822.0, 58403.0, 53541.0, 26008.0, 23796.0, 8224.0, 4896.0, 5002.0, 13440, 4800, 26631.0, 27500.0, 38820.0, 43472.0, 35233.0, 47068.0, 41690.0, 26997.0, 30888.0, 15642.0, 22240, 4800, 11725.0, 6424.0, 32412.0, 37396.0, 39154.0, 60216.0, 35484.0, 47241.0, 25889.0, 27172.0, 24006.0, 4589.0, 17008.0, 9432.0, 4800, 5472, 3840, 2267.0, 12264.0, 35599.0, 52191.0, 35646.0, 24328.0, 27408.0, 41833.0, 33210.0, 27566.0, 16562.0, 16002.0, 2784, 2880, 4800, 14168.0, 11977.0, 18000.0, 36732.0, 20392.0, 25759.0, 43485.0, 14084.0, 27516.0, 2480.0, 10969.0, 4800, 1920, 4800, 6685.0, 6971.0, 16224.0, 25123.0, 38255.0, 69644.0, 12426.0, 27192.0, 23303.0, 17223.0, 15163.0, 24835.0, 26478.0, 18165.0, 34985.0, 12630.0, 23067.0, 4157.0, 16480, 49270.0, 34844.0, 57555.0, 21920.0, 28132.0, 6889.0, 2110.0, 160, 4800, 6080, 16387.0, 24190.0, 69221.0, 25783.0, 49013.0, 4384, 5972.0, 7107.0, 4864, 26610.0, 54022.0, 19377.0, 46098.0, 27558.0, 50973.0, 9640.0, 37636.0, 34474.0, 65577.0, 18009.0, 20090.0, 4000, 6560, 3046.0, 27291.0, 22765.0, 80201.0, 31049.0, 70116.0, 24677.0, 40789.0, 15018.0, 45086.0, 8627.0, 8912.0, 960, 4800, 3117.0, 4544, 14186.0, 24967.0, 30954.0, 28877.0, 42126.0, 32672.0, 906.0, 0]
# action list :  [[4, 1], [5, 0], [6, 0], [7, 0], [8, 0], [9, 1], [10, 0], [11, 0], [12, 1], [13, 0], [14, 0], [15, 0], [16, 0], [17, 0], [18, 1], [26, 0], [27, 1], [28, 1], [29, 1], [30, 1], [31, 0], [32, 0], [33, 0], [34, 0], [35, 0], [36, 0], [37, 1], [38, 0], [39, 0], [40, 1], [44, 0], [45, 0], [46, 1], [47, 0], [48, 1], [50, 1], [51, 1], [52, 0], [53, 0], [54, 0], [55, 0], [56, 0], [57, 0], [58, 0], [59, 0], [62, 1], [64, 0], [65, 0], [66, 1], [67, 0], [68, 0], [69, 0], [70, 0], [71, 1], [72, 0], [73, 1], [74, 1], [75, 0], [76, 0], [77, 1], [78, 0], [79, 0], [80, 1], [85, 0], [86, 0], [87, 0], [88, 0], [89, 0], [90, 0], [91, 0], [92, 0], [93, 0], [94, 1], [95, 0], [96, 0], [97, 0], [99, 1], [104, 0], [105, 0], [106, 0], [107, 0], [108, 0], [109, 1], [110, 1], [111, 1], [112, 1], [113, 0], [114, 0], [115, 1], [116, 0], [117, 0], [118, 0], [119, 0], [120, 1], [122, 0], [124, 0], [125, 1], [126, 0], [127, 0], [128, 0], [129, 1], [130, 0], [131, 0], [132, 0], [133, 0], [134, 0], [135, 0], [136, 0], [137, 0], [138, 0], [139, 0], [140, 1], [142, 0], [143, 1], [145, 0], [146, 0], [147, 0], [148, 0], [149, 0], [150, 0], [151, 0], [152, 0], [153, 0], [154, 0], [155, 0], [156, 0], [157, 0], [158, 0], [159, 1], [164, 0], [165, 0], [166, 0], [167, 0], [168, 0], [169, 1], [170, 1], [171, 0], [172, 1], [173, 1], [174, 0], [175, 0], [176, 0], [177, 1], [183, 0], [184, 0], [185, 0], [186, 0], [187, 1], [188, 0], [189, 0], [190, 1], [191, 0], [192, 0], [193, 0], [194, 1], [195, 0], [196, 0], [197, 1], [198, 0], [199, 1], [206, 0], [207, 0], [208, 1], [209, 0], [210, 0], [211, 0], [212, 1], [213, 0], [214, 1], [215, 1], [216, 0], [217, 0], [218, 1], [224, 0], [225, 0], [226, 0], [227, 1], [228, 0], [229, 0], [230, 1], [231, 0], [232, 0], [233, 1], [234, 1], [235, 0], [236, 0], [237, 0], [239, 1], [245, 0], [246, 1], [247, 1], [248, 0], [249, 1], [250, 1], [251, 1], [252, 1], [253, 0], [254, 0], [255, 1], [256, 1], [259, 0], [265, 0], [266, 0], [267, 0], [268, 1], [269, 1], [270, 1], [271, 1], [272, 0], [273, 0], [274, 0], [275, 0], [276, 0], [277, 0], [278, 1], [281, 0], [283, 1], [284, 0], [285, 0], [286, 0], [287, 0], [288, 1], [289, 0], [290, 0], [291, 0], [292, 0], [293, 0], [294, 0], [295, 0], [296, 0], [297, 0], [298, 1], [305, 0], [306, 1], [307, 0], [308, 0], [309, 0], [310, 1], [311, 0], [313, 0], [314, 0], [317, 0], [318, 0], [320, 1], [322, 0], [323, 1], [325, 0], [326, 0], [327, 1], [328, 0], [329, 0], [330, 0], [333, 0], [334, 0], [338, 0], [347, 0], [348, 0], [349, 0], [350, 1], [351, 1], [352, 0], [354, 0], [355, 1], [364, 0], [365, 0], [366, 1], [369, 0], [370, 0], [372, 0], [373, 0], [375, 1], [378, 0], [379, 0], [382, 1], [385, 0], [386, 0], [388, 0], [389, 0], [391, 0], [392, 0], [396, 0], [397, 0], [401, 0], [404, 1], [405, 0], [407, 0], [409, 0], [410, 0], [413, 0], [414, 0], [427, 0], [428, 0], [429, 0], [430, 0], [433, 0], [434, 0], [436, 0], [437, 0], [440, 0], [446, 0], [448, 0], [449, 0], [453, 0], [454, 0], [468, 0], [469, 0], [471, 0], [472, 0], [474, 0], [475, 0], [477, 0], [478, 1], [483, 0], [485, 0], [486, 0], [487, 0], [488, 0], [490, 0], [491, 0], [492, 0], [495, 0], [496, 1]]
# Average query cost: 240.0597834149236

# 3000query-steam.csv
# New Environment v2 ( Add Skew )

# policy3000_v2_skew.pth
# Final reward: 0.12644493480740415, length: 498.0
# io_cost :  [273082.0, 320.0, 8017566.0]
# action list :  [[5, 1], [7, 0]]
# Average query cost: 245.98629283489097

# (IF No Add Skew )
# policy3000_v2.pth
# Final reward: 0.24453781512605044, length: 498.0
# io_cost :  [33360, 352.0, 448.0, 8482307.0]
# action list :  [[2, 1], [5, 0], [7, 0]]
# Average query cost: 252.67666518320723


# 1500query-steam.csv
