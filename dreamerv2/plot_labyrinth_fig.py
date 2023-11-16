"""Summarize logs across multiple runs with novel objects."""

import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


from dreamerv2.utils import labyrinth_utils as lu

CONTROL_TIMESTEP = 0.03 * 20  # This is based on the config value when the expt was run, multiplied by saveout frequency


def set_interactive_plot(interactive):
  import matplotlib as mpl
  mpl.use('TkAgg') if interactive else mpl.use('module://backend_interagg')


set_interactive_plot(False)


def main():
  user = 'ikauvar'
  if user == 'ikauvar':
    user_info = {'local_user': 'saal2',
                 'gcloud_path': '/home/saal2/google-cloud-sdk/bin/gcloud',
                 'user': 'ikauvar',
                 'gcp_zone': 'us-central1-a'}
  elif user == 'cd':
    user_info = {'local_user': 'cd',
                 'gcloud_path': '/snap/bin/gcloud',
                 'user': 'cd',
                 'gcp_zone': 'us-central1-a'}

  # ids = {  # labyrinth
  #   'DRE-357': 't4-tf-6',
  #   'DRE-358': 't4-tf-7',
  #   'DRE-359': 't4-tf-8',
  #   'DRE-360': 't4-tf-9',
  # }
  ids = {  # labyrinth black
    'DRE-362': 't4-tf-10',  # freerange
    # 'DRE-366': 't4-tf-9', #freerange # something wrong with this?
    # 'DRE-363': 't4-tf-6', #10k
    'DRE-364': 't4-tf-7',  # 10k
    # 'DRE-365': 't4-tf-8', #1k
    'DRE-380': 't4-tf-5',  # 100k
  }
  ids = {  # labyrinth black
    # 'DRE-381': 't4-tf-1', #10k
    # 'DRE-382': 't4-tf-2',  # 10k
    # 'DRE-390': 't4-tf-5',  # 100k
    # 'DRE-384': 't4-tf-6',  # 100k
    # 'DRE-385': 't4-tf-7',  # freerange
    # 'DRE-386': 't4-tf-8',  # freerange
    # 'DRE-387': 't4-tf-9',  # random
    # 'DRE-388': 't4-tf-10',  # random
    # 'DRE-389': 't4-tf-3',  # random
    'DRE-391': 't4-tf-1',  # 100k
    'DRE-392': 't4-tf-2',  # 100k
    'DRE-394': 't4-tf-3',  # freerange
    'DRE-395': 't4-tf-5',  # freerange
  }

  ids = {  # labyrinth black
    # 'DRE-442': 't4-tf-1',  # 100k
    # 'DRE-443': 't4-tf-2',  # 100k
    # 'DRE-444': 't4-tf-3',  # 100k
    # 'DRE-445': 't4-tf-4',  # 100k
    # 'DRE-446': 't4-tf-5',  # 200k
    # 'DRE-447': 't4-tf-6',  # 200k
    # 'DRE-448': 't4-tf-7',  # 200k
    # 'DRE-449': 't4-tf-8',  # 200k
    # # 'DRE-450': 't4-tf-9',  # 300k
    # 'DRE-451': 't4-tf-10',  # 300k
    # 'DRE-478': 't4-tf-2f',  # 50k
    # 'DRE-479': 't4-tf-3f',  # 50k
    # 'DRE-480': 't4-tf-4f',  # 50k
    # 'DRE-389': 't4-tf-3',  # random

    'DRA-20': 't4-tf-7',  # freerange
    'DRA-21': 't4-tf-8',  # freerange
    'DRA-22': 't4-tf-9',  # freerange
    'DRA-23': 't4-tf-10',  # freerange
    'DRA-24': 't4-tf-11',  # freerange
    'DRA-25': 't4-tf-12',  # freerange

    'DRE-394': 't4-tf-3',  # freerange
    'DRE-395': 't4-tf-5',  # freerange
    'DRE-385': 't4-tf-7',  # freerange
    'DRE-386': 't4-tf-8',  # freerange
  }

  ## Revisiting this (20230803) and aggregating expts
  all_ids = {}
  all_ids['Plan2Explore'] = {
    'DRA-20': 't4-tf-7',  # freerange
    'DRA-21': 't4-tf-8',  # freerange
    'DRA-22': 't4-tf-9',  # freerange
    'DRA-23': 't4-tf-10',  # freerange
    'DRA-24': 't4-tf-11',  # freerange
    'DRA-25': 't4-tf-12',  # freerange

    'DRE-394': 't4-tf-3',  # freerange
    'DRE-395': 't4-tf-5',  # freerange
    'DRE-385': 't4-tf-7',  # freerange
    'DRE-386': 't4-tf-8',  # freerange
  }

  all_ids['reset100k'] = {
    'DRE-442': 't4-tf-1',  # 100k
    'DRE-443': 't4-tf-2',  # 100k
    'DRE-444': 't4-tf-3',  # 100k
    'DRE-445': 't4-tf-4',  # 100k
    'DRE-390': 't4-tf-5',  # 100k
    'DRE-384': 't4-tf-6',  # 100k
    'DRE-391': 't4-tf-1',  # 100k
    'DRE-392': 't4-tf-2',  # 100k
  }

  labels = {
    'ball3': 'familiar',
    'ball4': 'novel',
  }
  colors = {
    'ball3': 'orange',
    'ball4': 'magenta',
  }

  plot_path = os.path.join(os.path.expanduser('~'), 'Dropbox/_gendreamer/plots/LABYRINTH/')
  os.makedirs(plot_path, exist_ok=True)

  do_plot_positions = True
  do_end_node_efficiency = True
  do_show_efficiency_from_start = False
  do_plot_node_sequence = True

  do_turn_bias = True
  do_time_in_maze = True
  do_simulate_random_turns = False

  force_download = 0

  env_nums = [0]

  saverate = 20
  xlim = int(15000e3 / saverate)

  # xlim = np.arange(0.5e6, 5.5e6, 0.5e6)/saverate
  # xlim = np.array([0.5e6, 1e6, 1.5e6, 2e6, 2.5e6, 3e6])/saverate
  # xlim = np.array([2e6, 5e6, 8e6, 11e6])/saverate
  # xlim = xlim.astype(int)

  do_plot_trajs = False

  ball_ids = list(colors.keys())

  expt_set = 'Plan2Explore'
  # expt_set = 'reset100k'
  ids = all_ids[expt_set]

  all_df = None
  changepoints = []
  for env_num in env_nums:

    efficiency_infos = {}
    maze_occupancies = {}
    turn_bias_infos = {}
    locs = {}

    for id, remote in ids.items():
      # df = au.load_log(id, env_num, remote, user_info, force_download=force_download)
      fn = f'/home/{user_info["local_user"]}/logs/csv/log_{id}_train_env{env_num}.csv'
      df = pd.read_csv(fn)
      df['id'] = id
      df['env_num'] = env_num

      save_freq = np.diff(df['total_step'].to_numpy())[1]

      if all_df is None:
        all_df = df
      else:
        all_df = pd.concat((all_df, df))

    if do_plot_trajs:
      if type(xlim) == int:
        xlim_list = [xlim]
      else:
        xlim_list = xlim
      for id in all_df['id'].unique():
        for xl in xlim_list:
          plot_df = all_df[all_df['id'] == id]

          plt.figure(figsize=(3, 3))
          plt.plot(plot_df['agent0_xloc'][:xl], plot_df['agent0_yloc'][:xl], 'k.',
                   markersize=1, alpha=0.2)
          plt.xlim(-30, 30)
          plt.ylim(-30, 30)
          plt.axis('off')
          plt.title(f'{id}: {xl * saverate / 1e6} M')
        plt.show()

    for id in all_df['id'].unique():
      plot_df = all_df[all_df['id'] == id]
      x = plot_df['agent0_xloc'][:xlim]
      y = plot_df['agent0_yloc'][:xlim]

      locs[id] = {'x': x, 'y': y}

      nt = len(x)
      (node_visits, intersection_visits,
       homecage_visits, stem_visits,
       right_visits, left_visits) = lu.get_roi_visits(x, y, do_plot=True, do_plot_rect=True,
                                                      title_str=f'{id}: total steps: {saverate * len(x):.2E}')

      if do_time_in_maze:
        bin_seconds = 500
        maze_occupancy = lu.get_time_in_maze(homecage_visits[0], nt, bin_seconds, CONTROL_TIMESTEP, do_plot=True,
                                             title_str=f'{id} maze occupancy')
        maze_occupancies[id] = maze_occupancy

      if do_turn_bias:
        turn_bias_info = lu.get_turn_bias(stem_visits, intersection_visits, right_visits, left_visits,
                                          nt, saverate, do_plot=True)
        turn_bias_infos[id] = turn_bias_info

      ### Count num unique nodes, from the start
      if do_end_node_efficiency:
        efficiency_info = lu.get_end_node_efficiency(node_visits, nt)
        lu.plot_end_node_efficiency(efficiency_info, saverate, nt)
        lu.do_plot_node_sequence(efficiency_info, saverate, nt)

        efficiency_infos[id] = efficiency_info

    for id in all_df['id'].unique():
      plot_df = all_df[all_df['id'] == id]

      plt.figure(figsize=(3, 3))
      plt.plot(plot_df['agent0_xloc'][:xlim], plot_df['agent0_yloc'][:xlim], 'k.',
               markersize=1, alpha=0.2)
      for b in ball_ids:
        try:
          plt.plot(plot_df[f'{b}_xloc'][:xlim], plot_df[f'{b}_yloc'][:xlim], '.',
                   markersize=2, color=colors[b], alpha=0.5)
        except:
          pass
      plt.title(f'Position {id}')

      plt.tight_layout()
      plt.show()

  print('done')

  import importlib
  importlib.reload(lu)
  lu.plot_end_node_efficiency(efficiency_infos, saverate, nt, multiple_expts=True, name_str=expt_set,
                              plot_path=plot_path, n_steps=xlim * saverate, fs=20)

  ## TODO: Overlay 100k and freerange onto a single plot
  print('done')

  # Plot turn biases
  importlib.reload(lu)
  lu.plot_choice_biases(turn_bias_infos, name_str=expt_set, plot_path=plot_path, draw_std=True, fs=12)

  importlib.reload(lu)
  lu.plot_maze_occupancies(maze_occupancies, name_str=expt_set,
                           bin_seconds=bin_seconds,
                           plot_path=plot_path, fs=10)

  for key in locs.keys():
    plt.figure(figsize=(3, 3))
    plt.plot(locs[key]['x'][:xlim], locs[key]['y'][:xlim], 'k.',
             markersize=1, alpha=0.2)
    plt.xlim(-30, 30)
    plt.ylim(-30, 30)
    plt.axis('off')
    # plt.title(f'{key}: {xlim * saverate / 1e6} M')
    plt.savefig(f'{plot_path}/traj_{expt_set}_{key}.png')
    plt.show()

  print('done')


if __name__ == "__main__":
  main()