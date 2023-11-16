"""
Generate example episodes.

Instructions:
Add

"""
import os
import common
import numpy as np
import elements
import collections
import datetime
import pathlib
from dreamerv2 import agent
import common.envs
import shutil
import pandas as pd
import matplotlib.pyplot as plt

from moviepy.editor import ImageSequenceClip
import utils.loading_utils as lu

examples = {
  # 0: {'env_name': 'admc_sphero_mazemultiagentInteract15_example_8', 'action': [(-1, 0)]},
  1: {'env_name': 'admc_sphero_mazemultiagentInteract19_7_novel_example_b3', 'action': [(-1, 0)]},
  2: {'env_name': 'admc_sphero_mazemultiagentInteract19_7_novel_example_b4', 'action': [(-1, 0)]},
  3: {'env_name': 'admc_sphero_mazemultiagentInteract19_7_novel_example_b3_2', 'action': [(-1, 0)]},
  4: {'env_name': 'admc_sphero_mazemultiagentInteract19_7_novel_example_b4_2', 'action': [(-1, 0)]},
}

examples = {
  # 0: {'env_name': 'admc_sphero_mazemultiagentInteract15_example_8', 'action': [(-1, 0)]},
  1: {'env_name': 'admc_sphero_mazemultiagentInteract19_7_novel_example_b3_black', 'action': [(-1, 0)]},
  2: {'env_name': 'admc_sphero_mazemultiagentInteract19_7_novel_example_b4_black', 'action': [(-1, 0)]},
  3: {'env_name': 'admc_sphero_mazemultiagentInteract19_7_novel_example_b3_2_black', 'action': [(-1, 0)]},
  4: {'env_name': 'admc_sphero_mazemultiagentInteract19_7_novel_example_b4_2_black', 'action': [(-1, 0)]},
}

examples = {
  # 0: {'env_name': 'admc_sphero_mazemultiagentInteract15_example_8', 'action': [(-1, 0)]},
  1: {'env_name': 'admc_sphero_object_example1_magenta', 'action': [(-1, 0)]},
  2: {'env_name': 'admc_sphero_object_example1_yellow', 'action': [(-1, 0)]},
  3: {'env_name': 'admc_sphero_object_example1_white', 'action': [(-1, 0)]},
  4: {'env_name': 'admc_sphero_object_example1_empty', 'action': [(-1, 0)]},
  5: {'env_name': 'admc_sphero_object_example1_magenta', 'action': [(-.5, 0)]},
  6: {'env_name': 'admc_sphero_object_example1_yellow', 'action': [(-.5, 0)]},
  7: {'env_name': 'admc_sphero_object_example1_white', 'action': [(-.5, 0)]},
  8: {'env_name': 'admc_sphero_object_example1_empty', 'action': [(-.5, 0)]},
  9: {'env_name': 'admc_sphero_object_example1_magenta', 'action': [(0, 0.5)]},
  10: {'env_name': 'admc_sphero_object_example1_yellow', 'action': [(0, 0.5)]},
  11: {'env_name': 'admc_sphero_object_example1_white', 'action': [(0, 0.5)]},
  12: {'env_name': 'admc_sphero_object_example1_empty', 'action': [(0, 0.5)]},
  # 13: {'env_name': 'admc_sphero_object_example1_magenta', 'action': {'basedir': '/home/saal2/logdir/admc_sphero_novel_object_2ball/p2e/9',
  #                                                                    'ckpt': 'variables_000101000.pkl'}},
  # 14: {'env_name': 'admc_sphero_object_example1_yellow', 'action': {'basedir': '/home/saal2/logdir/admc_sphero_novel_object_2ball/p2e/9',
  #                                                                    'ckpt': 'variables_001001000.pkl'}},
  20: {'env_name': 'admc_sphero_object_example1_magenta', 'action': 'explore_1'},
  21: {'env_name': 'admc_sphero_object_example1_yellow', 'action': 'explore_1'},
  22: {'env_name': 'admc_sphero_object_example1_magenta', 'action': 'explore_2'},
  23: {'env_name': 'admc_sphero_object_example1_yellow', 'action': 'explore_2'},
  24: {'env_name': 'admc_sphero_object_example6_magenta', 'action': 'explore_2'},
  25: {'env_name': 'admc_sphero_object_example6_yellow', 'action': 'explore_2'},
  26: {'env_name': 'admc_sphero_object_example4_magenta', 'action': 'turn_1'},
  27: {'env_name': 'admc_sphero_object_example4_yellow', 'action': 'turn_1'},
  28: {'env_name': 'admc_sphero_object_example1_magenta', 'action': 'forward_back_1'},
  29: {'env_name': 'admc_sphero_object_example1_yellow', 'action': 'forward_back_1'},
  30: {'env_name': 'admc_sphero_object_example6_magenta', 'action': 'explore_3'},
  31: {'env_name': 'admc_sphero_object_example6_yellow', 'action': 'explore_3'},
  32: {'env_name': 'admc_sphero_object_example4_magenta', 'action': 'explore_3'},
  33: {'env_name': 'admc_sphero_object_example4_yellow', 'action': 'explore_3'},
  34: {'env_name': 'admc_sphero_object_example6_magenta', 'action': 'explore_4'},
  35: {'env_name': 'admc_sphero_object_example6_yellow', 'action': 'explore_4'},
  36: {'env_name': 'admc_sphero_object_example4_magenta', 'action': 'explore_4'},
  37: {'env_name': 'admc_sphero_object_example4_yellow', 'action': 'explore_4'},
  38: {'env_name': 'admc_sphero_object_example6_magenta', 'action': 'explore_5'},
  39: {'env_name': 'admc_sphero_object_example6_yellow', 'action': 'explore_5'},
  40: {'env_name': 'admc_sphero_object_example1_magenta', 'action': 'explore_1'},  # train set
  41: {'env_name': 'admc_sphero_object_example6_magenta', 'action': 'explore_2'},  # test set
  42: {'env_name': 'admc_sphero_object_example1_magenta', 'action': [(-1, 0)]},
  43: {'env_name': 'admc_sphero_object_example4_magenta', 'action': 'turn_2'},
  44: {'env_name': 'admc_sphero_object_example4_yellow', 'action': 'turn_2'},
  45: {'env_name': 'admc_sphero_object_example4_magenta', 'action': 'turn_3'},
  46: {'env_name': 'admc_sphero_object_example4_yellow', 'action': 'turn_3'},
  47: {'env_name': 'admc_sphero_object_example8_magenta', 'action': 'forward_back_2'},
  48: {'env_name': 'admc_sphero_object_example8_yellow', 'action': 'forward_back_2'},
  49: {'env_name': 'admc_sphero_object_example8_magenta', 'action': 'forward_back_3'},
  50: {'env_name': 'admc_sphero_object_example8_yellow', 'action': 'forward_back_3'},

  100: {'env_name': 'admc_sphero_labyrinth_black_example', 'action': 'maze_right_right_right'},

  # 200: {'env_name': 'admc_sphero_distal', 'action': 'explore_1'},  # train set
  # 201: {'env_name': 'admc_sphero_distal_column1', 'action': 'explore_1'},  # train set
  # 202: {'env_name': 'admc_sphero_distal_column2', 'action': 'explore_1'},  # train set
  # 203: {'env_name': 'admc_sphero_distal', 'action': 'explore_2'},  # train set
  # 204: {'env_name': 'admc_sphero_distal_column1', 'action': 'explore_2'},  # train set
  # 205: {'env_name': 'admc_sphero_distal_column2', 'action': 'explore_2'},  # train set

  300: {'env_name': 'admc_sphero_distal', 'action': 'explore_1'},  # train set
  301: {'env_name': 'admc_sphero_distal_column1', 'action': 'explore_1'},  # train set
  302: {'env_name': 'admc_sphero_distal_column2', 'action': 'explore_1'},  # train set
  303: {'env_name': 'admc_sphero_distal', 'action': 'explore_2'},  # train set
  304: {'env_name': 'admc_sphero_distal_column1', 'action': 'explore_2'},  # train set
  305: {'env_name': 'admc_sphero_distal_column2', 'action': 'explore_2'},  # train set

  306: {'env_name': 'admc_sphero_distal', 'action': [(0, 0.5)]},  # train set
  307: {'env_name': 'admc_sphero_distal', 'action': [(0, 0.4)]},  # train set
  308: {'env_name': 'admc_sphero_distal', 'action': [(0, 0.3)]},  # train set
  309: {'env_name': 'admc_sphero_distal', 'action': 'move_spin1'},  # train set
  310: {'env_name': 'admc_sphero_distal', 'action': 'move_spin2'},  # train set
  311: {'env_name': 'admc_sphero_distal', 'action': 'move_spin3'},  # train set
}

def make_env(mode, config, logdir):
  suite, task = config.task.split('_', 1)
  if suite == 'dmc':
    env = common.DMC(
      task, config.action_repeat, config.render_size, config.dmc_camera)
    env = common.NormalizeAction(env)
  elif suite == 'atari':
    env = common.Atari(
      task, config.action_repeat, config.render_size,
      config.atari_grayscale)
    env = common.OneHotAction(env)
  elif suite == 'crafter':
    assert config.action_repeat == 1
    outdir = logdir / 'crafter' if mode == 'train' else None
    reward = bool(['noreward', 'reward'].index(task)) or mode == 'eval'
    env = common.Crafter(outdir, reward)
    env = common.OneHotAction(env)
  elif suite == 'admc':
    from adaptgym.wrapped import ADMC
    env = ADMC(task, action_repeat=config.action_repeat, size=config.render_size, logdir=logdir, mode=mode,
               record_every_k_timesteps=1,
               flush_logger_every=5,
               spoof_done_every=1000,
               # wide_fov=True,
               )
    # from adaptgym.wrapped import AdaptDMC_nonepisodic
    # env = AdaptDMC_nonepisodic(task, action_repeat=config.action_repeat, size=config.render_size, logdir=logdir, mode=mode)
    env = common.NormalizeAction(env)
  else:
    raise NotImplementedError(suite)
  env = common.TimeLimit(env, config.time_limit)
  return env


if __name__ == "__main__":
  # for which_example in examples.keys():
  # for which_example in [24, 25]:
  # for which_example in [26, 27]:
  # for which_example in [309, 310, 311]:
  for which_example in [310]:

    print(which_example)
    home = os.path.expanduser("~")
    logdir = pathlib.PosixPath(f'{home}/logdir/EXAMPLE_EPS/{which_example}/')
    if os.path.exists(logdir):
      shutil.rmtree(logdir)
    # logdir = pathlib.PosixPath(f'{home}/logs/GEN-EXAMPLE_EPS_black/')
    os.makedirs(logdir, exist_ok=True)
    gifdir = logdir / 'gifs'
    os.makedirs(gifdir, exist_ok=True)


    env_name = examples[which_example]['env_name']
    action = examples[which_example]['action']

    nepisodes = 1
    time_limit = 120
    if 'explore' in action:
      nepisodes = 200
      time_limit = 1e8

    config = {
      'task': env_name,
      'action_repeat': 2,
      'render_size': [64, 64],
      'aesthetic': 'default',
      'egocentric_camera': True,
      'num_envs': 1,
      'time_limit': time_limit, #120, #3000, #120,
      'replay_size': 5e5,
      'control_timestep': 0.03,
      'physics_timestep': 0.005,
      'reset_position_freq': 0,
    }
    config = elements.Config(config)

    replay = common.Replay(logdir / f'train_episodes', config.replay_size)

    # Define actions
    envs = [make_env('train', config, logdir) for _ in range(config.num_envs)]
    action_space = envs[0].action_space['action']
    driver = common.Driver(envs)

    step = elements.Counter(0)
    driver.on_step(lambda x, worker: step.increment())
    driver.on_episode(lambda ep: replay.add_episode(ep))
    driver.reset()
    do_load_agent = isinstance(action, dict)
    if do_load_agent:
      agnt = lu.load_agent(action['basedir'],
                           checkpoint_name=action['ckpt'],
                           batch_size=5)
      policy = lambda *args: agnt.policy(*args, mode='explore')
    else:
      policy = lambda x, y:  ({'action': action}, y)

    # do_forward_bias = 'forward' in action
    do_action_seq = isinstance(action, str)
    if do_action_seq:
      def create_stateless_policy():
        if action == 'explore_1':
          action_sequence = [np.array([(-1.0, 0.0)])]*20
          action_sequence += [np.array([(0.0, -0.5)])]*10
          action_sequence += [np.array([(-0.8, 0.0)])]*50
          action_sequence += [np.array([(0.0, -0.5)])]*10
          action_sequence += [np.array([(1.0, 0.0)])]*10
          action_sequence += [np.array([(0.0, -0.5)])]*10
          action_sequence *= 2
          action_sequence += [np.array([(0.0, 0.5)])]*40
        elif action == 'explore_2':
          action_sequence = [np.array([(-1.0, 0.0)])]*23
          action_sequence += [np.array([(0.0, 0.5)])]*10
          action_sequence += [np.array([(-0.8, 0.0)])]*49
          action_sequence += [np.array([(0.0, 0.5)])]*10
          action_sequence += [np.array([(1.0, 0.0)])]*9
          action_sequence += [np.array([(0.0, 0.5)])]*10
          action_sequence *= 3
          action_sequence += [np.array([(0.0, -0.5)])]*39
        elif action == 'explore_3':
          action_sequence = [np.array([(-1.0, 0.0)])]*23
          action_sequence += [np.array([(0.0, 0.5)])]*10
          action_sequence += [np.array([(-0.8, 0.0)])]*49
          action_sequence += [np.array([(0.0, 0.5)])]*10
          action_sequence += [np.array([(0.0, -0.5)])]*40
          action_sequence += [np.array([(1.0, 0.0)])]*9
          action_sequence += [np.array([(0.0, 0.5)])]*5
          action_sequence += [np.array([(1.0, 0.0)])]*9
          action_sequence += [np.array([(0.0, 0.5)])]*10
          action_sequence *= 3
          action_sequence += [np.array([(0.0, -0.5)])]*39
        elif action == 'explore_4':
          action_sequence = [np.array([(-1.0, 0.0)])]*13
          action_sequence += [np.array([(0.0, 0.5)])] * 5
          action_sequence += [np.array([(-1.0, 0.0)])]*5
          action_sequence += [np.array([(0.0, -0.5)])] * 5
          action_sequence += [np.array([(-1.0, 0.0)])]*5
          action_sequence += [np.array([(0.0, 0.5)])]*10
          action_sequence += [np.array([(-0.8, 0.0)])]*40
          action_sequence += [np.array([(0.0, 0.5)])] * 5
          action_sequence += [np.array([(-1.0, 0.0)])]*1
          action_sequence += [np.array([(0.0, -0.5)])] * 5
          action_sequence += [np.array([(-1.0, 0.0)])]*9
          action_sequence += [np.array([(0.0, 0.5)])]*10
          action_sequence += [np.array([(0.0, -0.5)])]*40
          action_sequence += [np.array([(1.0, 0.0)])]*9
          action_sequence += [np.array([(0.0, 0.5)])]*5
          action_sequence += [np.array([(1.0, 0.0)])]*9
          action_sequence += [np.array([(0.0, 0.5)])]*10
          action_sequence *= 3
          action_sequence += [np.array([(0.0, -0.5)])]*39
        elif action == 'explore_5':
          action_sequence = [np.array([(-1.0, 0.0)])]*23
          action_sequence += [np.array([(0.0, 0.5)])]*13
          action_sequence += [np.array([(-0.8, 0.0)])]*49
          action_sequence += [np.array([(0.0, 0.5)])]*10
          action_sequence += [np.array([(0.0, -0.5)])]*37
          action_sequence += [np.array([(1.0, 0.0)])]*9
          action_sequence += [np.array([(0.0, 0.5)])]*7
          action_sequence += [np.array([(1.0, 0.0)])]*9
          action_sequence += [np.array([(0.0, 0.5)])]*10
          action_sequence *= 3
          action_sequence += [np.array([(0.0, -0.5)])]*45
        elif action == 'turn_1':
          action_sequence = [np.array([(0.0, 0.5)])]*6
          action_sequence += [np.array([(0.0, -0.5)])]*12
          action_sequence += [np.array([(0.0, 0.5)])]*6
        elif action == 'forward_back_1':
          action_sequence = [np.array([(-1.0, 0.0)])]*45
          action_sequence += [np.array([(1.0, 0.0)])]*60
          action_sequence += [np.array([(-1.0, 0.0)])]*10
        elif action == 'turn_2':
          action_sequence = [np.array([(0.0, 0.5)])]*16
          action_sequence += [np.array([(0.0, -0.5)])]*32
          action_sequence += [np.array([(0.0, 0.5)])]*16
        elif action == 'turn_3':
          action_sequence = [np.array([(0.0, 0.5)])]*24
          action_sequence += [np.array([(0.0, -0.5)])]*48
          action_sequence += [np.array([(0.0, 0.5)])]*24
        elif action == 'forward_back_2':
          action_sequence = [np.array([(-1.0, 0.0)])]*50
          action_sequence += [np.array([(1.0, 0.0)])]*75
          action_sequence += [np.array([(-1.0, 0.0)])]*20
        elif action == 'forward_back_3':
          action_sequence = [np.array([(-1.0, 0.0)])]*50
          action_sequence += [np.array([(1.0, 0.0)])]*55
          action_sequence += [np.array([(-1.0, 0.0)])]*40
          action_sequence += [np.array([(1.0, 0.0)])]*40
          action_sequence += [np.array([(-1.0, 0.0)])]*40
          action_sequence += [np.array([(1.0, 0.0)])]*50

        elif action == 'maze_right_right_right':
          action_sequence = [np.array([(-1.0, 0.0)])]*85
          action_sequence += [np.array([(0.0, 0.0)])]*10
          action_sequence += [np.array([(-.9, 0.0)])]*1
          action_sequence += [np.array([(0.0, 0.0)])]*10
          action_sequence += [np.array([(0.0, -0.5)])]*15
          action_sequence += [np.array([(0.0, 0.0)])]*30
          action_sequence += [np.array([(-1.0, 0.0)])]*27
          action_sequence += [np.array([(0.0, 0.0)])]*15
          action_sequence += [np.array([(1.0, 0.0)])]*4
          action_sequence += [np.array([(0.0, 0.0)])]*20
          action_sequence += [np.array([(0.0, -0.5)])]*17
          action_sequence += [np.array([(0.0, 0.0)])]*10
          action_sequence += [np.array([(-1.0, 0.0)])]*75

          # action_sequence = [np.array([(-1.0, 0.0)])]*100
          # action_sequence += [np.array([(1.0, 0.0)])]*2
          # action_sequence += [np.array([(0.0, -0.5)])]*15
          # action_sequence += [np.array([(-1.0, 0.0)])]*160
          # action_sequence += [np.array([(0.0, -0.5)])]*15
          # action_sequence += [np.array([(-1.0, 0.0)])]*80
        elif action == 'move_spin1':
          action_sequence = [np.array([(-1.0, 0.0)])]*25
          action_sequence += [np.array([(0.0, 0.5)])]*110
        elif action == 'move_spin2':
          action_sequence = [np.array([(0.0, -0.5)])]*10
          action_sequence += [np.array([(-1.0, 0.0)])]*35
          action_sequence += [np.array([(0.0, 0.5)])]*110
        elif action == 'move_spin3':
          action_sequence = [np.array([(0.0, 0.5)])]*20
          action_sequence += [np.array([(-1.0, 0.0)])]*25
          action_sequence += [np.array([(0.0, 0.5)])]*110

        action_index = 0  # Index to track the current action
        print(len(action_sequence))

        def stateless_policy(x, y):
          nonlocal action_index
          action = action_sequence[action_index]
          action_index = (action_index + 1) % len(action_sequence)  # Cycle through actions
          # print(action)
          return ({'action': action}, y)

        return stateless_policy


      policy = create_stateless_policy()

    # def policy(x, y):
      #   r = np.random.rand()
      #   if r > 0.7:
      #     action = [(-1, 0)]
      #   elif r > 0.2 and r <= 0.5:
      #     action = [(0, np.random.rand())]
      #   else:
      #     action = [(0, -np.random.rand())]
      #   print(action, r)
      #   return ({'action': action}, y)

    driver(policy, episodes=nepisodes)


    # Make a gif
    # eps = list(replay._episodes.keys())
    eps = list(replay._complete_eps.keys())
    eps.sort()
    path = eps[-1]
    ep = np.load(path)
    imgs = ep['image']
    print(len(imgs))

    clip = ImageSequenceClip(list(imgs), fps=20)
    clip.write_gif(f'{gifdir}/{which_example}.gif', fps=20)

    # Plot trajectory
    # logdir = '/home/saal2/logs/EXAMPLE_EPS/20'
    df = pd.read_csv(f'{str(logdir)}/log_train_env0.csv')
    plt.figure()
    plt.plot(df['agent0_xloc'], df['agent0_yloc'], 'k')
    if 'yellow' in env_name:
      color = 'y'
    elif 'magenta' in env_name:
      color = 'm'
    else:
      color = 'g'
    try:
      plt.plot(df['object1_xloc'], df['object1_yloc'], color)
    except:
      pass
    import matplotlib.patches as patches
    rectangle = patches.Rectangle((-12.78-0.5, -14.18-0.5), 25.5+1, 25.5+1, linewidth=1, edgecolor='k', facecolor='none')
    plt.gca().add_patch(rectangle)
    plt.savefig(f'{str(logdir)}/trajectory.png')

  print('done')