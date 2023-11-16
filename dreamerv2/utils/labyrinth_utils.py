import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches


def set_interactive_plot(interactive):
  import matplotlib as mpl
  mpl.use('TkAgg') if interactive else mpl.use('module://backend_interagg')


def simple_plot(ax):
  ax.spines.right.set_visible(False)
  ax.spines.top.set_visible(False)
  ax.yaxis.set_ticks_position('left')
  ax.xaxis.set_ticks_position('bottom')


# node_mat = []
# for node, inds in node_visits.items():
#   visits = np.zeros(nt)
#   visits[inds] = 1
#   node_mat.append(visits)
# node_mat = np.array(node_mat)
# node_id_mat = node_mat * np.arange(1, node_mat.shape[0] + 1)[:, np.newaxis]
# node_vec = np.sum(node_id_mat, axis=0)


def organize_visits(which_visits, nt):
  """Organize roi visits into a vector, where entry represents which roi."""
  mat = []
  for node, inds in which_visits.items():
    visits = np.zeros(nt)
    visits[inds] = 1
    mat.append(visits)
  mat = np.vstack(mat)
  id_mat = mat * np.arange(1, mat.shape[0] + 1)[:, np.newaxis]
  vec = np.sum(id_mat, axis=0)
  return vec


def get_end_node_efficiency(node_visits, nt):
  """Compute efficiency of reaching different end nodes in the labyrinth"""

  node_vec = organize_visits(node_visits, nt)
  node_vec = node_vec[np.where(node_vec > 0)[0]]  # Get rid of all timepoints where not in any roi.

  node_seq = []
  last_node = 0
  for i in range(len(node_vec)):
    n = node_vec[i]
    if n != 0 and n != last_node:
      node_seq.append(n)
      last_node = n

  # Count how many unique nodes have been visited, since the start.
  n_unique_0 = []
  for i in range(len(node_seq)):
    n_unique_0.append(len(np.unique(node_seq[:i])))

  # Count how many unique nodes have been visited in different sized windows.
  # In a string of n nodes, how many of these are distinct.
  # Slide a window of size n across the sequence, count d distinct nodes
  # in each window. Then average over d over all windows in all clips.
  mean_n_unique = {}
  ns = np.array([2, 4, 8, 16, 32, 48, 64, 100, 128, 156, 200, 250, 300, 350, 400, 450, 500, 512])
  for n in ns:
    n_unique = []
    for i in range(len(node_seq) - n):
      seq = node_seq[i:i + n]
      n_unique.append(len(np.unique(seq)))
    mean_n_unique[n] = np.mean(np.array(n_unique))

  N32 = np.inf
  for i in range(len(node_seq)):
    if len(np.unique(node_seq[:i])) == 32:
      N32 = i
  efficiency = 32 / N32

  efficiency_info = {'node_seq': node_seq,
                     'n_unique_0': n_unique_0,
                     'mean_n_unique': mean_n_unique,
                     'efficiency': efficiency}
  return efficiency_info


def do_plot_node_sequence(efficiency_info, saverate, nt):
  node_seq = efficiency_info['node_seq']
  plt.plot(node_seq)
  plt.ylabel('Endnode id')
  plt.xlabel('Endnode visit')
  # simple_plot(plt.gca())
  plt.title(f'{id}: total steps: {saverate * nt:.2E}')
  plt.show()


def plot_maze_occupancies(maze_occupancies, name_str, bin_seconds,
                          plot_path, fs=12):
  min_t = np.min([len(x) for x in maze_occupancies.values()])
  occupancy = np.vstack([x[:min_t] for x in maze_occupancies.values()])
  occupancy_fraction = np.mean(occupancy, axis=1)

  import matplotlib.gridspec as gridspec
  x = np.arange(occupancy.shape[1]) * bin_seconds
  y = occupancy.mean(0)
  fig = plt.figure(figsize=(4, 2.5))
  gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

  # Plot on the first subplot with wider width
  ax1 = plt.subplot(gs[0])
  plt.fill_between(x, y - np.std(occupancy, axis=0), y + np.std(occupancy, axis=0),
                   color='m', alpha=0.2)
  plt.plot(x, y, color='m')
  plt.xlabel('Absolute time (s)', fontsize=fs)
  plt.ylabel('Fraction of time in maze', fontsize=fs)
  plt.xticks([0, 1e5, 2e5], ['0', '100K', '200K'], fontsize=fs)
  plt.yticks(fontsize=fs)
  plt.tight_layout()
  # plt.show()

  # plt.figure(figsize=(2, 3))
  ax1 = plt.subplot(gs[1])
  plt.bar(['Mouse', name_str], [0.44, occupancy_fraction.mean()], yerr=[0, np.std(occupancy_fraction)],
          color=['r', 'm'])
  plt.ylabel('Fraction of time in maze', fontsize=fs)
  # plt.ylim([0, 1])
  plt.xticks(rotation=30)
  plt.xticks(fontsize=fs)
  plt.yticks(fontsize=fs)
  plt.tight_layout()
  plt.savefig(f'{plot_path}/time_in_maze.pdf')
  plt.show()


def plot_choice_biases(turn_bias_infos, name_str, plot_path=None, draw_std=False, fs=8):
  plt.figure(figsize=(3, 6))
  plt.subplot(2, 1, 1)
  Psfs = np.array([turn_bias_infos[x]['Psf'] for x in turn_bias_infos.keys()])
  Pbfs = np.array([turn_bias_infos[x]['Pbf'] for x in turn_bias_infos.keys()])
  mouse_Psf = (0.78, 0.02)
  mouse_Pbf = (0.81, 0.03)
  if draw_std:
    p1, = plt.plot(np.mean(Psfs), np.mean(Pbfs), 'm.')
    ellipse = patches.Ellipse((np.mean(Psfs), np.mean(Pbfs)),
                              2 * np.std(Psfs), 2 * np.std(Pbfs), color='m', alpha=0.3)
    plt.gca().add_patch(ellipse)
  else:
    p1, = plt.plot(Psfs, Pbfs, 'g.')
  p2, = plt.plot(2 / 3, 2 / 3, 'b.')
  p3, = plt.plot(mouse_Psf[0], mouse_Pbf[0], 'r.')

  # Create a red circle patch
  ellipse = patches.Ellipse((mouse_Psf[0], mouse_Pbf[0]),
                            2 * mouse_Psf[1], 2 * mouse_Pbf[1], color='red', alpha=0.3)
  plt.gca().add_patch(ellipse)

  plt.xlim([0, 1]), plt.ylim([0, 1])
  plt.xticks(fontsize=fs)
  plt.yticks(fontsize=fs)
  plt.xlabel(r'$P_{SF}$ (forward from stem)', fontsize=fs)
  plt.ylabel(r'$P_{BF}$ (forward from branch)', fontsize=fs)
  plt.title('Biases to move forward', fontsize=fs)
  plt.legend([p1, p2, p3], [name_str, 'Random Choice', 'Mouse'], frameon=True)
  # plt.show()

  plt.subplot(2, 1, 2)
  Psas = np.array([turn_bias_infos[x]['Psa'] for x in turn_bias_infos.keys()])
  Pbss = np.array([turn_bias_infos[x]['Pbs'] for x in turn_bias_infos.keys()])

  mouse_Psa = (0.71, 0.02)
  mouse_Pbs = (0.63, 0.02)

  if draw_std:
    p1, = plt.plot(np.mean(Psas), np.mean(Pbss), 'm.')
    ellipse = patches.Ellipse((np.mean(Psas), np.mean(Pbss)),
                              2 * np.std(Psas), 2 * np.std(Pbss), color='m', alpha=0.3)
    plt.gca().add_patch(ellipse)
  else:
    p1, = plt.plot(Psas, Pbss, 'm.')
  p2, = plt.plot(1 / 2, 1 / 2, 'b.')
  p3, = plt.plot(0.71, 0.63, 'r.')
  p3, = plt.plot(mouse_Psa[0], mouse_Pbs[0], 'r.')
  ellipse = patches.Ellipse((mouse_Psa[0], mouse_Pbs[0]),
                            2 * mouse_Psa[1], 2 * mouse_Pbs[1], color='red', alpha=0.3)
  plt.gca().add_patch(ellipse)
  plt.xticks(fontsize=fs)
  plt.yticks(fontsize=fs)
  plt.xlim([0, 1]), plt.ylim([0, 1])
  plt.xlabel(r'$P_{SA}$ (alternate from last turn)', fontsize=fs)
  plt.ylabel(r'$P_{BS}$ (turn, not straight)', fontsize=fs)
  plt.title('Turn biases', fontsize=fs)
  plt.suptitle('Choice biases at intersections', fontsize=fs + 2)
  plt.tight_layout()
  if plot_path is not None:
    plt.savefig(f'{plot_path}/choice_biases.pdf')
  plt.show()


def plot_end_node_efficiency(efficiency_infos, saverate, nt, multiple_expts=False,
                             do_show_efficiency_from_start=False,
                             name_str='',
                             plot_path=None,
                             n_steps=0, fs=12):
  n_expts = 1
  if multiple_expts:
    expt_ids = list(efficiency_infos.keys())
    n_expts = len(expt_ids)

  plt.figure()
  random_x = [2, 10, 20, 30, 60, 100, 200, 500, 1000, 2000]
  random_y = [2, 4.2, 6.4, 8.4, 13.7, 19.67, 31.6, 50, 62.6, 64]
  mousec1_x = [2, 10, 20, 30, 60, 100, 200, 500, 1000, 2000]
  mousec1_y = [2, 6.4, 11.9, 16.5, 28.1, 39.2, 53.3, 62.6, 63.5, 64]
  p_rand, = plt.semilogx(random_x, random_y, 'b')
  p_mouse, = plt.semilogx(mousec1_x, mousec1_y, 'r')

  efficiencies = []
  end_nodes_visited = []
  for i in range(n_expts):
    if multiple_expts:
      efficiency_info = efficiency_infos[expt_ids[i]]
    else:
      efficiency_info = efficiency_infos
    node_seq = efficiency_info['node_seq']
    n_unique_0 = efficiency_info['n_unique_0']
    mean_n_unique = efficiency_info['mean_n_unique']
    efficiency = efficiency_info['efficiency']
    efficiencies.append(efficiency)
    end_nodes_visited.append(len(np.unique(node_seq)))

    ns = np.sort(np.array(list(mean_n_unique.keys())))
    ### Numbers based on Fig 8 in https://elifesciences.org/articles/66175#equ1

    optimal = ns
    p_opt, = plt.semilogx(ns, optimal, 'k')
    if multiple_expts:
      alpha = 0.3
    else:
      alpha = 1
    p_ai, = plt.semilogx(ns, [mean_n_unique[x] for x in ns], 'm', alpha=alpha)
    if do_show_efficiency_from_start:
      p5, = plt.semilogx(np.arange(len(n_unique_0)), n_unique_0, 'g')
    plt.ylim([0, 64])
    plt.xlim([2, 2000])
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.axhline(32, color='k', linestyle='--')
    plt.ylabel('New end nodes found', fontsize=fs)
    plt.xlabel('End nodes visited', fontsize=fs)
    plt.xlim([4, 350])
    plt.title(
      f'Mean efficiency: {np.mean(np.array(efficiencies)):0.2f}, Mean ends visited: '
      f'{np.mean(np.array(end_nodes_visited)):0.2f}, Steps: {n_steps:1.1e} '
      # f', total steps: {saverate * nt:.2E}'
    )

  # Overlay mean
  if multiple_expts:
    mean_effic = {}
    for y in efficiency_infos[expt_ids[0]]['mean_n_unique'].keys():
      mean_effic[y] = np.nanmean(np.array([efficiency_infos[x]['mean_n_unique'][y] for x in expt_ids]))

    p_ai, = plt.semilogx(ns, [mean_effic[x] for x in ns], 'm')

  if do_show_efficiency_from_start:
    plt.legend([p_opt, p_rand, p_mouse, p_ai, p5],
               ['Optimal (1.0)', 'Random Choice (0.23)', 'Mouse (0.42)',
                f'{name_str} ({np.mean(np.array(efficiencies)):0.2f})',
                f'{name_str} from start'], fontsize=16, handlelength=0.75, handletextpad=0.5)
  else:
    plt.legend([p_opt, p_rand, p_mouse, p_ai],
               ['Optimal (1.0)', 'Random Choice (0.23)', 'Mouse (0.42)',
                f'{name_str} ({np.mean(np.array(efficiencies)):0.2f})'],
               frameon=False, fontsize=16, handlelength=0.75, handletextpad=0.5)

  plt.tight_layout()
  if plot_path is not None:
    plt.savefig(f'{plot_path}/end_node_efficiency_{name_str}.pdf')

  plt.show()


def get_turn_bias(stem_visits, intersection_visits, right_visits, left_visits,
                  nt, saverate, do_plot=True):
  tt = saverate * np.arange(nt)
  stem_vec = organize_visits(stem_visits, nt)
  intersection_vec = organize_visits(intersection_visits, nt)
  left_vec = organize_visits(left_visits, nt)
  right_vec = organize_visits(right_visits, nt)

  STEM = 0
  LEFT = 1
  RIGHT = 2
  INTX = 3
  vec = np.vstack([stem_vec, left_vec, right_vec, intersection_vec])
  vec = vec[:, np.where(np.sum(vec, axis=0) > 0)[0]]  # Get rid of all timepoints where not in any roi.
  seq = []
  last_v = vec[:, 0]
  for k in range(vec.shape[1]):
    v = vec[:, k]
    if np.all(v == 0):
      continue
    if np.all(v == last_v):
      continue
    seq.append(v)
    last_v = v
  seq = np.vstack(seq).T

  # Probability of moving forward from a stem instead of reversing
  stem_entries = np.where(np.logical_and(seq[STEM, :-1] > 0, seq[INTX, 1:] > 0))[0]
  stem_returns = np.where(np.logical_and(np.logical_and(seq[STEM, :-2] > 0, seq[INTX, 1:-1] > 0), seq[STEM, 2:] > 0))[0]
  Psf = 1 - (len(stem_returns) / len(stem_entries))

  # Probability of an alternating turn from the preceding one
  left_turns = np.logical_and(np.logical_and(seq[STEM, :-2] > 0, seq[INTX, 1:-1] > 0), seq[LEFT, 2:] > 0).astype(int)
  right_turns = np.logical_and(np.logical_and(seq[STEM, :-2] > 0, seq[INTX, 1:-1] > 0), seq[RIGHT, 2:] > 0).astype(int)
  turns = 1 * left_turns + 2 * right_turns
  turns = turns[np.where(turns > 0)[0]]
  alternating_turns = np.where(np.abs(np.diff(turns)) > 0)[0]
  non_alternating_turns = np.where(np.abs(np.diff(turns)) == 0)[0]
  Psa = len(alternating_turns) / (len(non_alternating_turns) + len(alternating_turns))

  # Probability of moving forward from a branch instead of reversing
  l_to_r = np.logical_and(np.logical_and(seq[LEFT, :-2] > 0, seq[INTX, 1:-1] > 0), seq[RIGHT, 2:] > 0).astype(int)
  l_to_s = np.logical_and(np.logical_and(seq[LEFT, :-2] > 0, seq[INTX, 1:-1] > 0), seq[STEM, 2:] > 0).astype(int)
  l_to_l = np.logical_and(np.logical_and(seq[LEFT, :-2] > 0, seq[INTX, 1:-1] > 0), seq[LEFT, 2:] > 0).astype(int)
  r_to_l = np.logical_and(np.logical_and(seq[RIGHT, :-2] > 0, seq[INTX, 1:-1] > 0), seq[LEFT, 2:] > 0).astype(int)
  r_to_s = np.logical_and(np.logical_and(seq[RIGHT, :-2] > 0, seq[INTX, 1:-1] > 0), seq[STEM, 2:] > 0).astype(int)
  r_to_r = np.logical_and(np.logical_and(seq[RIGHT, :-2] > 0, seq[INTX, 1:-1] > 0), seq[RIGHT, 2:] > 0).astype(int)
  nbranch_forward = len(np.where(l_to_r)[0]) + len(np.where(r_to_l)[0]) + \
                    len(np.where(l_to_s)[0]) + len(np.where(r_to_s)[0])
  nbranch_reverse = len(np.where(l_to_l > 0)[0]) + len(np.where(r_to_r > 0)[0])
  Pbf = nbranch_forward / (nbranch_forward + nbranch_reverse)

  # Probability of moving from branch to stem
  l_to_s = np.logical_and(np.logical_and(seq[LEFT, :-2] > 0, seq[INTX, 1:-1] > 0), seq[STEM, 2:] > 0).astype(int)
  r_to_s = np.logical_and(np.logical_and(seq[RIGHT, :-2] > 0, seq[INTX, 1:-1] > 0), seq[STEM, 2:] > 0).astype(int)
  l_to_r = np.logical_and(np.logical_and(seq[LEFT, :-2] > 0, seq[INTX, 1:-1] > 0), seq[RIGHT, 2:] > 0).astype(int)
  r_to_l = np.logical_and(np.logical_and(seq[RIGHT, :-2] > 0, seq[INTX, 1:-1] > 0), seq[LEFT, 2:] > 0).astype(int)
  nto_stem = len(np.where(l_to_s)[0]) + len(np.where(r_to_s)[0])
  nto_branch = len(np.where(l_to_r)[0]) + len(np.where(r_to_l)[0])
  Pbs = nto_stem / (nto_stem + nto_branch)

  print(f'Psf: {Psf:0.2f}, Psa: {Psa:0.2f}, Pbf: {Pbf:0.2f}, Pbs: {Pbs:0.2f}')

  if do_plot:
    plt.plot(Psf, Pbf, 'g+'), plt.plot(2 / 3, 2 / 3, 'b+')
    plt.xlim([0, 1]), plt.ylim([0, 1])
    plt.xlabel('Psf'), plt.ylabel('Pbf')
    plt.show()

    plt.plot(Psa, Pbs, 'g+'), plt.plot(1 / 2, 1 / 2, 'b+')
    plt.xlim([0, 1]), plt.ylim([0, 1])
    plt.xlabel('Psa'), plt.ylabel('Pbs')
    plt.show()

  turn_bias_info = {'Psf': Psf,
                    'Pbf': Pbf,
                    'Psa': Psa,
                    'Pbs': Pbs}
  return turn_bias_info
  # return (Psf, Pbf, Psa, Pbs)


def pointInRect(xs, ys, rect):
  x1, y1, w, h = rect
  x2, y2 = x1 + w, y1 + h
  # in_rect = np.logical_and(np.logical_and(xs > x1, xs < x2), np.logical_and(ys > y1, ys < y2))
  in_rect = np.all((xs > x1, xs < x2, ys > y1, ys < y2), axis=0)
  return in_rect


def roi_visit(coord, x, y, w, h, color, do_plot, do_plot_rect, do_plot_dots=True):
  r = (coord[0] - w / 2, coord[1] - h / 2, w, h)
  inds = np.where(pointInRect(x.to_numpy(), y.to_numpy(), r))[0]
  if do_plot:
    if do_plot_dots:
      plt.plot(x[inds], y[inds], '.', color=color, markersize=1, alpha=1, linewidth=0)
    if do_plot_rect:
      rect = Rectangle((coord[0] - w / 2, coord[1] - h / 2), w, h, fill=False, edgecolor=color)
      plt.gca().add_patch(rect)
  return inds


def get_roi_visits(x, y, do_plot=True, do_plot_rect=True, title_str=''):
  """Get timestamps of each visit to various regions of interest."""
  (stem_coords, left_coords,
   right_coords, intersections1_coords,
   endnode_coords, nonmaze_coords) = get_labyrinth_coords(stem_d=0.9)

  # Set up regions of interest
  w_stem = 0.8
  h_stem = 0.8
  w_intersections = 0.9
  h_intersections = 0.9
  w_end = 3
  h_end = 1.5
  w_nonmaze = 12.5
  h_nonmaze = 26.5

  if do_plot:
    plt.plot(x, y, 'k.', markersize=1, alpha=0.2)
    ax = plt.gca()

  node_visits = {}
  intersection_visits = {}
  homecage_visits = {}
  stem_visits = {}
  right_visits = {}
  left_visits = {}
  for i, coord in enumerate(endnode_coords):
    node_visits[i] = roi_visit(coord, x, y, w_end, h_end, 'r', do_plot, do_plot_rect)
  for i, coord in enumerate(intersections1_coords):
    intersection_visits[i] = roi_visit(coord, x, y, w_intersections, h_intersections, 'g', do_plot, do_plot_rect)
  for i, coord in enumerate(stem_coords):
    stem_visits[i] = roi_visit(coord, x, y, w_stem, h_stem, 'm', do_plot, do_plot_rect)
  for i, coord in enumerate(right_coords):
    right_visits[i] = roi_visit(coord, x, y, w_stem, h_stem, 'c', do_plot, do_plot_rect)
  for i, coord in enumerate(left_coords):
    left_visits[i] = roi_visit(coord, x, y, w_stem, h_stem, 'orange', do_plot, do_plot_rect)
  for i, coord in enumerate(nonmaze_coords):
    coord = np.array(coord) + np.array([w_nonmaze / 2, h_nonmaze / 2])
    homecage_visits[i] = roi_visit(coord, x, y, w_nonmaze, h_nonmaze, 'b', do_plot, do_plot_rect, do_plot_dots=False)

  if do_plot:
    nt = len(x)
    plt.title(title_str)
    plt.show()

  return (node_visits, intersection_visits,
          homecage_visits, stem_visits,
          right_visits, left_visits)


def get_time_in_maze(homecage_visits, nt, bin_seconds, CONTROL_TIMESTEP, do_plot=True, title_str=''):
  """Compute time spent in the maze (as opposed to homecage)."""
  homecage_occupancy = np.zeros(nt)
  homecage_occupancy[homecage_visits] = 1
  maze_occupancy = 1 - homecage_occupancy

  binsize = int(bin_seconds / CONTROL_TIMESTEP)
  nbins = int(np.floor(len(maze_occupancy) / binsize))
  binned_maze_occupancy = np.zeros(nbins)
  for i in np.arange(nbins):
    binned_maze_occupancy[i] = np.sum(maze_occupancy[binsize * i:binsize * (i + 1)]) / binsize

  if do_plot:
    plt.plot(np.arange(nbins) * bin_seconds, binned_maze_occupancy)
    plt.ylim([0, 1])
    plt.xlim([0, nbins * bin_seconds])
    plt.ylabel('% time spent in maze')
    plt.xlabel('Absolute time (s)')
    simple_plot(plt.gca())
    plt.title(title_str)
    plt.tight_layout()
    plt.show()

  return binned_maze_occupancy


def get_labyrinth_coords(stem_d=0.9):
  y_stems1 = np.array([19.6, 8.4, -2.8, -14]) - stem_d
  x_stems1 = np.array([-9.775, 1.35, 12.55, 23.775])
  y_left1 = np.array([19.6, 8.4, -2.8, -14])
  x_left1 = np.array([-9.775, 1.35, 12.55, 23.775]) - stem_d
  y_right1 = np.array([19.6, 8.4, -2.8, -14])
  x_right1 = np.array([-9.775, 1.35, 12.55, 23.775]) + stem_d
  stem_coords = list(itertools.product(x_stems1, y_stems1))
  left_coords = list(itertools.product(x_left1, y_left1))
  right_coords = list(itertools.product(x_right1, y_right1))

  y_stems1 = np.array([14, 2.8, -8.4, -19.6]) + stem_d
  x_stems1 = np.array([-9.775, 1.35, 12.55, 23.775])
  y_left1 = np.array([14, 2.8, -8.4, -19.6])
  x_left1 = np.array([-9.775, 1.35, 12.55, 23.775]) + stem_d
  y_right1 = np.array([14, 2.8, -8.4, -19.6])
  x_right1 = np.array([-9.775, 1.35, 12.55, 23.775]) - stem_d
  stem_coords.extend(list(itertools.product(x_stems1, y_stems1)))
  left_coords.extend(list(itertools.product(x_left1, y_left1)))
  right_coords.extend(list(itertools.product(x_right1, y_right1)))

  y_stems1 = np.array([-16.8, -5.6, 5.6, 16.8])
  x_stems1 = np.array([-9.775, 12.55]) + stem_d
  y_left1 = np.array([-16.8, -5.6, 5.6, 16.8]) - stem_d
  x_left1 = np.array([-9.775, 12.55])
  y_right1 = np.array([-16.8, -5.6, 5.6, 16.8]) + stem_d
  x_right1 = np.array([-9.775, 12.55])
  stem_coords.extend(list(itertools.product(x_stems1, y_stems1)))
  left_coords.extend(list(itertools.product(x_left1, y_left1)))
  right_coords.extend(list(itertools.product(x_right1, y_right1)))

  y_stems1 = np.array([-16.8, -5.6, 5.6, 16.8])
  x_stems1 = np.array([1.35, 23.775]) - stem_d
  y_left1 = np.array([-16.8, -5.6, 5.6, 16.8]) + stem_d
  x_left1 = np.array([1.35, 23.775])
  y_right1 = np.array([-16.8, -5.6, 5.6, 16.8]) - stem_d
  x_right1 = np.array([1.35, 23.775])
  stem_coords.extend(list(itertools.product(x_stems1, y_stems1)))
  left_coords.extend(list(itertools.product(x_left1, y_left1)))
  right_coords.extend(list(itertools.product(x_right1, y_right1)))

  y_stems1 = np.array([16.8, -5.6]) - stem_d
  x_stems1 = np.array([-4.2125, 18.1625])
  y_left1 = np.array([16.8, -5.6])
  x_left1 = np.array([-4.2125, 18.1625]) - stem_d
  y_right1 = np.array([16.8, -5.6])
  x_right1 = np.array([-4.2125, 18.1625]) + stem_d
  stem_coords.extend(list(itertools.product(x_stems1, y_stems1)))
  left_coords.extend(list(itertools.product(x_left1, y_left1)))
  right_coords.extend(list(itertools.product(x_right1, y_right1)))

  y_stems1 = np.array([-16.8, 5.6]) + stem_d
  x_stems1 = np.array([-4.2125, 18.1625])
  y_left1 = np.array([-16.8, 5.6])
  x_left1 = np.array([-4.2125, 18.1625]) + stem_d
  y_right1 = np.array([-16.8, 5.6])
  x_right1 = np.array([-4.2125, 18.1625]) - stem_d
  stem_coords.extend(list(itertools.product(x_stems1, y_stems1)))
  left_coords.extend(list(itertools.product(x_left1, y_left1)))
  right_coords.extend(list(itertools.product(x_right1, y_right1)))

  y_stems1 = np.array([-11.2, 11.2])
  x_stems1 = np.array([-4.2125]) + stem_d
  y_left1 = np.array([-11.2, 11.2]) - stem_d
  x_left1 = np.array([-4.2125])
  y_right1 = np.array([-11.2, 11.2]) + stem_d
  x_right1 = np.array([-4.2125])
  stem_coords.extend(list(itertools.product(x_stems1, y_stems1)))
  left_coords.extend(list(itertools.product(x_left1, y_left1)))
  right_coords.extend(list(itertools.product(x_right1, y_right1)))

  y_stems1 = np.array([-11.2, 11.2])
  x_stems1 = np.array([18.1625]) - stem_d
  y_left1 = np.array([-11.2, 11.2]) + stem_d
  x_left1 = np.array([18.1625])
  y_right1 = np.array([-11.2, 11.2]) - stem_d
  x_right1 = np.array([18.1625])
  stem_coords.extend(list(itertools.product(x_stems1, y_stems1)))
  left_coords.extend(list(itertools.product(x_left1, y_left1)))
  right_coords.extend(list(itertools.product(x_right1, y_right1)))

  y_stems1 = np.array([11.2]) - stem_d
  x_stems1 = np.array([6.95])
  y_left1 = np.array([11.2])
  x_left1 = np.array([6.95]) - stem_d
  y_right1 = np.array([11.2])
  x_right1 = np.array([6.95]) + stem_d
  stem_coords.extend(list(itertools.product(x_stems1, y_stems1)))
  left_coords.extend(list(itertools.product(x_left1, y_left1)))
  right_coords.extend(list(itertools.product(x_right1, y_right1)))

  y_stems1 = np.array([-11.2]) + stem_d
  x_stems1 = np.array([6.95])
  y_left1 = np.array([-11.2])
  x_left1 = np.array([6.95]) + stem_d
  y_right1 = np.array([-11.2])
  x_right1 = np.array([6.95]) - stem_d
  stem_coords.extend(list(itertools.product(x_stems1, y_stems1)))
  left_coords.extend(list(itertools.product(x_left1, y_left1)))
  right_coords.extend(list(itertools.product(x_right1, y_right1)))

  y_stems1 = np.array([0])
  x_stems1 = np.array([6.95]) - stem_d
  y_left1 = np.array([0]) + stem_d
  x_left1 = np.array([6.95])
  y_right1 = np.array([0]) - stem_d
  x_right1 = np.array([6.95])
  stem_coords.extend(list(itertools.product(x_stems1, y_stems1)))
  left_coords.extend(list(itertools.product(x_left1, y_left1)))
  right_coords.extend(list(itertools.product(x_right1, y_right1)))

  y_intersections1 = [-19.6, -14, -8.4, -2.8, 2.8, 8.4, 14, 19.6]
  x_intersections1 = [-9.775, 1.35, 12.55, 23.775]
  intersections1_coords = list(itertools.product(x_intersections1, y_intersections1))
  y_intersections2 = [-16.8, -5.6, 5.6, 16.8]
  x_intersections2 = [-9.775, -4.2125, 1.35, 12.55, 18.1625, 23.775]
  intersections2_coords = list(itertools.product(x_intersections2, y_intersections2))
  y_intersections3 = [-11.2, 11.2]
  x_intersections3 = [-4.2125, 6.95, 18.1625]
  intersections3_coords = list(itertools.product(x_intersections3, y_intersections3))
  y_intersections4 = [0]
  x_intersections4 = [6.95]
  intersections4_coords = list(itertools.product(x_intersections4, y_intersections4))
  intersections1_coords.extend(intersections2_coords)
  intersections1_coords.extend(intersections3_coords)
  intersections1_coords.extend(intersections4_coords)

  y_endnodes = [-19.6, -14, -8.4, -2.8, 2.8, 8.4, 14, 19.6]
  x_endnodes = [-12.75, -6.8, -1.6, 4.3, 9.6, 15.5, 20.8, 26.75]
  endnode_coords = list(itertools.product(x_endnodes, y_endnodes))

  x_nonmaze = -27.25
  y_nonmaze = -13.25
  nonmaze_coords = [(x_nonmaze, y_nonmaze)]

  return (stem_coords, left_coords, right_coords,
          intersections1_coords, endnode_coords, nonmaze_coords)