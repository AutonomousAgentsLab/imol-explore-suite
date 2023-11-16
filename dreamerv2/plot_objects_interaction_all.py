"""Plot summary of object interaction log."""
import os
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
try:
  matplotlib.use('module://backend_interagg')
except:
  pass

# Defaults
# SAVERATE = 20  # record_every_k_timesteps param in logging_params in adaptgym.wrapped.ADMC
SAVERATE = 1
PREFILL = 1e3  # prefill param in config.yaml
PREFILL = 0

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, help='Path to the log directory',
                        default='/home/saal2/logdir/admc_sphero_novel_object_2ball/mloss/')
    parser.add_argument('--outdir', type=str, help='Path to the output directory for plots',
                        # default='/home/saal2/logdir/admc_sphero_novel_object_2ball/p2e/')
                        default='/home/saal2/Dropbox/_gendreamer/plots/2BALL/mloss/')
    parser.add_argument('--runs', type=str, help='Run ids',
                        # default='8,9,10,11')
                        default='2,3,4,5')

    return parser.parse_args()

CONTROL_TIMESTEP = 0.03 # seconds per action
def main():
  args = parse_arguments()
  logdir = args.logdir
  outdir = args.outdir
  runs = args.runs
  os.makedirs(outdir, exist_ok=True)

  expid = '/'.join(logdir.split('/')[-2:]) + runs
  dfs = {}
  for r in runs.split(','):
    fn = f'{logdir}/{r}/log_train_env0.csv'
    dfs[r] = pd.read_csv(fn)

  colors = {'object2': 'magenta',
            'object1': 'orange'}  # Object name defined by admc_sphero_novel_object_unchanging environment.
  labels = {'object2': 'Old',
            'object1': 'New'}
  ball_ids = list(colors.keys())

  # Set timestep of object introduction into environment.
  if 'unchanging' in fn:
    tstart = int((0 + PREFILL) / SAVERATE)
    tend = int(1e6 / SAVERATE)
  elif '2ball' in fn:
    tstart = int((0 + PREFILL) / SAVERATE)
    tend = int(1e6 / SAVERATE)
  else:
    tstart = int((5e5 + PREFILL) / SAVERATE)
    tend = int(1.5e6 / SAVERATE)

  # #  Plot overall exploration trajectory and interactions across runs.
  do_trace_summaries = True
  if do_trace_summaries:
    plot_playground_object_expts(dfs, tstart, tend, ball_ids, colors,
                                 labels, expid, SAVERATE,
                                 savepath=outdir)

    plot_playground_object_expts(dfs, int(tstart + 5e5/SAVERATE), tend, ball_ids, colors,
                                 labels, expid, SAVERATE,
                                 savepath=outdir,
                                 use_seconds=True)

    plot_playground_object_expts(dfs, int(tstart + 5e5/SAVERATE), tend, ball_ids, colors,
                                 labels, expid, SAVERATE,
                                 savepath=outdir,
                                 use_seconds=True,
                                 do_cumsum=False,
                                 metric='attention',
                                 do_legend=False)

  do_vid = False
  if do_vid:
    df = dfs['8']
    t0 = np.where(df[f'collisions_object1/shell'])[0][20]
    trajectory_vid(df, t0, 20, colors, savepath='/home/saal2/Dropbox/_gendreamer/plots/2BALL/p2e/gifs', do_gif=True)



  do_plot_deep_interactions = True
  if do_plot_deep_interactions:
    plot_deep_shallow(dfs, int(tstart + 5e5/SAVERATE), tend, colors, labels, savepath=outdir)


  do_plot_approach = True
  if do_plot_approach:
    """Events when cross from outer to inner circle within two timesteps"""
    df = dfs['8']
    x0 = df['agent0_xloc']
    y0 = df['agent0_yloc']
    r0 = df['agent0_orientation']
    c1 = df[f'collisions_object1/shell']
    x1 = df[f'object1_xloc']
    y1 = df[f'object1_yloc']
    a1 = df[f'attention_object1/shell']
    c2 = df[f'collisions_object2/shell']
    x2 = df[f'object2_xloc']
    y2 = df[f'object2_yloc']
    a2 = df[f'attention_object2/shell']
    w0 = df[f'collisions_wall']
    d1 = np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)

    plot_approaches(dfs, int(tstart + 5e5 / SAVERATE), tend, colors, labels, savepath=outdir)

    near_radius = 2
    far_radius = 2.1
    approaches1 = []
    approaches2 = []
    for df in dfs.values():
      approach1 = get_approaches(df, 'object1', int(tstart + 5e5 / SAVERATE), tend, near_radius, far_radius)
      approach2 = get_approaches(df, 'object2', int(tstart + 5e5 / SAVERATE), tend, near_radius, far_radius)
      approaches1.append(approach1)
      approaches2.append(approach2)
    approaches1 = np.vstack(approaches1)
    approaches2 = np.vstack(approaches2)

    plt.figure(figsize=(2, 3))
    plt.plot(1*np.ones(approaches1.shape[0]), approaches1.sum(axis=1), 'k.')
    plt.plot(2*np.ones(approaches2.shape[0]), approaches2.sum(axis=1), 'k.')
    plt.xticks([1,2], [labels['object1'], labels['object2']], fontsize=16)
    plt.bar(1, approaches1.sum(axis=1).mean(), alpha=0.5, color=colors['object1'])
    plt.bar(2, approaches2.sum(axis=1).mean(), alpha=0.5, color=colors['object2'])
    plt.ylabel('Approach Events', fontsize=16)
    plt.yticks(fontsize=16)
    simple_axis(plt.gca())
    plt.tight_layout()
    plt.show()

  print('done')

def plot_approaches(dfs, t0, tend, colors, labels, savepath=None, fontsize=16):
  near_radius = 2
  far_radius = 2.1
  approaches1 = []
  approaches2 = []
  for df in dfs.values():
    approach1 = get_approaches(df, 'object1', t0, tend, near_radius, far_radius)
    approach2 = get_approaches(df, 'object2', t0, tend, near_radius, far_radius)
    approaches1.append(approach1)
    approaches2.append(approach2)
  approaches1 = np.vstack(approaches1)
  approaches2 = np.vstack(approaches2)

  plt.figure(figsize=(2, 3))
  plt.plot(1 * np.ones(approaches1.shape[0]), approaches1.sum(axis=1), 'k.')
  plt.plot(2 * np.ones(approaches2.shape[0]), approaches2.sum(axis=1), 'k.')
  plt.xticks([1, 2], [labels['object1'], labels['object2']], fontsize=fontsize)
  plt.bar(1, approaches1.sum(axis=1).mean(), alpha=0.5, color=colors['object1'])
  plt.bar(2, approaches2.sum(axis=1).mean(), alpha=0.5, color=colors['object2'])
  plt.ylabel('Approach Events', fontsize=fontsize)
  plt.yticks(fontsize=fontsize)
  simple_axis(plt.gca())
  plt.tight_layout()
  plt.savefig(f'{savepath}/approaches.png')
  plt.savefig(f'{savepath}/approaches.pdf')
  plt.show()

def get_approaches(df, object, t0, tend, near_radius, far_radius):
  """From ahamdlou et al. : approaching (turning of the head towards the object
  accompanied by a body movement decreasing the distance between the mouse and the
  object; approaches were only counted when the mouse started more than 0.5 cm from the
  object and ended when the mouse was at 0.5 cm of the object; The body point for
  distance thresholding was the nose tip. The head direction was considered to be towards
  the object if the imaginary line between head center and nose tip was aligned with
  object.)"""
  x0 = df['agent0_xloc'][t0:tend]
  y0 = df['agent0_yloc'][t0:tend]
  x1 = df[f'{object}_xloc'][t0:tend]
  y1 = df[f'{object}_yloc'][t0:tend]
  c1 = df[f'collisions_{object}/shell'][t0:tend]
  a1 = df[f'attention_{object}/shell'][t0:tend]
  d1 = np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)

  turn_to_object = np.logical_and(a1[1:].to_numpy() < 1, a1[:-1].to_numpy() > 0)
  go_to_object = np.logical_and(d1[:-2].to_numpy() <= near_radius,
                                np.logical_or(d1[1:-1].to_numpy() > far_radius, d1[2:].to_numpy() > far_radius))
  approach = np.logical_and(turn_to_object[1:], go_to_object)
  return approach


def plot_deep_shallow(dfs, t0, tend, colors, labels, savepath=None):
  """
  From Ahmadlou et al. "Therefore, we categorized the object investigation sequences into shallow investigation
  (in which sniffing is not followed by biting) and deep investigation
  (in which sniffing is followed by biting).
  In both cases, the investigatory event starts with sniff and ends when no
  investigatory action (sniff, bite, grab, and carry) is taken anymore for at least 100 ms (fig. S1C).
  We introduced the deep versus shallow investigation preference (DSP) using the relative time a mouse carries out
  deep investigation compared with the shallow investigation.
  DSP varies between –π/2 and π/2, where –π/2 and π/2 indicate the absolute preference for shallow and deep investigation,
  respectively, and 0 indicates equal preference for deep and shallow investigation.
  This depth of investigation was much higher for novel objects than it was for familiar objects"

  To apply this to our setting, we consider shallow investigation to be when the agent is close
  to the object, attending to it for at least 1/20 timesteps, and not colliding with it.
  We consider deep investigation to be when the agent is colliding with an object at least 10/20
  timesteps. """

  DSP1s = []
  DSP2s = []
  for df in dfs.values():
    DSP1, deep1, shallow1 = get_deep_investigation(df, 'object1', t0, tend)
    DSP2, deep2, shallow2 = get_deep_investigation(df, 'object2', t0, tend)
    DSP1s.append(DSP1)
    DSP2s.append(DSP2)


  fig = plt.figure(figsize=(3,3))
  ax = plt.axes(polar=True)

  theta = DSP1s
  width = np.radians(1)
  p1 = ax.bar(DSP1s, np.ones_like(DSP1s), width=width,
              facecolor=colors['object1'], edgecolor=colors['object1'], alpha=0.5, align='edge')
  p2 = ax.bar(DSP2s, np.ones_like(DSP2s), width=width,
              facecolor=colors['object2'], edgecolor=colors['object2'], alpha=0.5, align='edge')
  ax.set_thetalim(-np.pi / 2, np.pi / 2)
  pi = np.pi
  ax.set_xticks([-pi / 2, -pi / 4, 0, pi / 4, pi / 2])
  ax.set_xticklabels([r'$-\pi/2$: Shallow', r'$-\pi/4$', '0', r'$\pi/4$', r'$\pi/2$: Deep'])

  ax.set_rticks([0, 1])
  ax.set_rlim([0, 1])
  plt.legend([p2, p1], [labels['object2'], labels['object1']], frameon=False, loc='upper left', bbox_to_anchor=[0.7, 1])
  plt.savefig(f'{savepath}/deep_shallow.png')
  plt.savefig(f'{savepath}/deep_shallow.pdf')
  plt.show()

def get_deep_investigation(df, object, t0, tend, close_radius=2, deep_thresh=10):
  """
  DSP is deep vs. shallow investigation preference.
  pi/2 means exclusively deep investigation, 0 means equal deep and shallow investigation
  """
  x0 = df['agent0_xloc'][t0:tend]
  y0 = df['agent0_yloc'][t0:tend]
  c1 = df[f'collisions_{object}/shell'][t0:tend]
  x1 = df[f'{object}_xloc'][t0:tend]
  y1 = df[f'{object}_yloc'][t0:tend]
  a1 = df[f'attention_{object}/shell'][t0:tend]

  d1 = np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
  shallow1 = np.logical_and(np.logical_and(d1 < close_radius, c1 < 1), a1 > 1)
  deep1 = c1 > deep_thresh
  DSP1 = np.arcsin((deep1.sum() - shallow1.sum()) / (deep1.sum() + shallow1.sum()))

  return DSP1, deep1, shallow1


def trajectory_vid(df, start, nframes, colors, savepath, do_gif=True):
  from matplotlib.animation import FuncAnimation

  xo = df['agent0_xloc']
  yo = df['agent0_yloc']
  ro = df['agent0_orientation']
  c1 = df[f'collisions_object1/shell']
  x1 = df[f'object1_xloc']
  y1 = df[f'object1_yloc']
  a1 = df[f'attention_object1/shell']
  c2 = df[f'collisions_object2/shell']
  x2 = df[f'object2_xloc']
  y2 = df[f'object2_yloc']
  a2 = df[f'attention_object2/shell']
  w0 = df[f'collisions_wall']

  fig, ax = plt.subplots(figsize=(5, 5))

  # Define a function to update the plot for each frame
  def update_frame(tc):
    update_frame_w_ax(tc, ax)

  def update_frame_w_ax(tc, ax):
    ax.clear()  # Clear the previous frame

    ax.plot(xo[tc], yo[tc], 'ko', markersize=11.5)
    ax.plot(x1[tc], y1[tc], 'o', markersize=11.5, color=colors['object1'])
    ax.plot(x2[tc], y2[tc], 'o', markersize=11.5, color=colors['object2'])

    angled_line((xo[tc], yo[tc]), ro[tc], ax)

    w = 15
    rectangle = patches.Rectangle((-12.78-0.5, -14.18-0.5), 25.5+1, 25.5+1, linewidth=1, edgecolor='k', facecolor='none')
    ax.add_patch(rectangle)

    ax.set_title(f't={tc*SAVERATE:>6}, Wall={w0[tc]:>2}\n'
                 f'Attention: orange={a1[tc]:>2}, magenta={a2[tc]:>2}\n'
                 f'Collision: orange={c1[tc]:>2}, magenta={c2[tc]:>2}', font='monospace')

    ax.axis('equal')
    ax.set_xlim([-15, 15])
    ax.set_ylim([-15, 15])
    ax.axis('off')

  if do_gif:
    # Create the animation
    ani = FuncAnimation(fig, update_frame, frames=range(start, start + nframes), interval=200)

    # Save the animation as a GIF
    fname = f'{savepath}/animation_{start}_{nframes}.gif'
    ani.save(fname, writer='pillow')  # 'pillow' is a common writer for GIFs
    print(fname)
    # Display the animation (optional)
    plt.show()
  else:
    for tc in range(start, start+nframes):
      print(tc)
      fig, ax = plt.subplots(figsize=(5, 5))
      update_frame_w_ax(tc, ax)
      plt.show()

def angled_line(p, r, ax, line_length=2.0):
  x = [p[0], p[0] + line_length * np.cos(r + 1 * np.pi / 2)]  # This seems to be right?
  y = [p[1], p[1] + line_length * np.sin(r + 1 * np.pi / 2)]
  ax.plot(x, y, '-')



  ## Make video over time showing orientation (agent0_orientation, agent0_xloc, agent0_yloc)
def plot_playground_object_expts(plot_dfs, tstart, tend,
                                ball_ids, colors, labels,
                                expid, saverate,
                                savepath=None,
                                use_seconds=False,
                                metric='collisions',
                                do_cumsum=True,
                                do_legend=True):
  """Plot map of agent and object locations over the session,
  and cumulative collisions vs time."""
  plt.figure(figsize=(4, 2))
  plt.suptitle(f'{expid}')

  legend_str = []
  legend_plots = []
  for b in ball_ids:
    ys = []
    for plot_df in plot_dfs.values():
      if do_cumsum:
        ys.append(np.cumsum(plot_df[f'{metric}_{b}/shell'][tstart:tend]))
      else:
        N = 500
        ys.append(np.convolve(plot_df[f'{metric}_{b}/shell'][tstart:tend],
                              np.ones(N) / N, mode='same'))

    ys = np.vstack(ys)
    print(ys.shape)

    mean = np.median(ys, axis=0)
    sd = np.std(ys, axis=0)
    tt = saverate * np.arange(ys[0].shape[0])
    if use_seconds:
      tt = tt*CONTROL_TIMESTEP
    plt.fill_between(tt, np.maximum(0, mean-sd), mean+sd, color=colors[b], alpha=0.5)
    p1, = plt.plot(tt, mean, color=colors[b])
    legend_plots.append(p1)
    legend_str.append(labels[f'{b}'].capitalize())
    plt.axvline(0.5e6-tstart*saverate, color='k', linestyle='--')
    simple_axis(plt.gca())
    # legend_str.append(f'{b}:' + labels[f'{b}'])
    if not use_seconds:
      plt.xlabel('Steps in env')
    else:
      plt.xlabel('Time from new object introduction (s)')
    if do_cumsum:
      plt.ylabel(f'Cumulative {metric}')
    else:
      plt.ylabel(f'Smoothed {metric}')
  if do_legend:
    plt.legend(legend_plots, legend_str, frameon=False, loc='upper left',
               handlelength=0.5, handletextpad=0.5, bbox_to_anchor=[0.05, 1.1])
  plt.tight_layout()
  if savepath is not None:
    if use_seconds:
      plt.savefig(f'{savepath}/{metric}_summary_s.png')
      plt.savefig(f'{savepath}/{metric}_summary_s.pdf')
    else:
      plt.savefig(f'{savepath}/{metric}_summary.png')
      plt.savefig(f'{savepath}/{metric}_summary.pdf')
    print(f'Saving to {savepath}')
  plt.show()


def plot_indiv_playground_object_expt(plot_df, tstart, tend,
                                      ball_ids, colors, labels,
                                      expid, saverate,
                                      savepath=None):
  """Plot map of agent and object locations over the session,
  and cumulative collisions vs time."""
  plt.figure(figsize=(8, 3))
  ax = plt.subplot(1, 2, 1)
  print(tstart)
  print(tend)
  print(plot_df['agent0_xloc'][tstart:tend])
  print(plot_df['agent0_yloc'][tstart:tend])

  plt.plot(plot_df['agent0_xloc'][tstart:tend],
           plot_df['agent0_yloc'][tstart:tend], 'k.',
           markersize=1, alpha=0.2)
  for b in ball_ids:
    plt.plot(plot_df[f'{b}_xloc'][tstart:tend],
             plot_df[f'{b}_yloc'][tstart:tend], '.',
             markersize=2, color=colors[b], alpha=0.5)
  ax.axis('equal')
  ax.set_xlim([-15, 15])
  ax.set_ylim([-15, 15])
  plt.axis('off')
  plt.suptitle(f'{expid}')

  plt.subplot(1, 2, 2)
  legend_str = []
  for b in ball_ids:
    y = np.cumsum(plot_df[f'collisions_{b}/shell'][tstart:tend])
    tt = saverate * np.arange(y.shape[0])
    plt.plot(tt, y,
             color=colors[b])
    legend_str.append(f'{b}:' + labels[f'{b}'])
    plt.xlabel('Steps in env')
    plt.ylabel('Cumulative collisions')
  plt.legend(legend_str, bbox_to_anchor=[1.05, 1])
  plt.tight_layout()
  if savepath is not None:
    plt.savefig(f'{savepath}/interaction_summary.png')
    print(f'Saving to {savepath}')
  plt.show()


def plot_indiv_trajectories(plot_df, tstart, tends,
                            ball_ids, colors,  titlestrs,
                            savepath=None):
  """Plot map of agent and object locations accumulated
  to different timepoints."""
  fig, axs = plt.subplots(1, len(tends), figsize=(9, 3))
  for i in range(len(tends)):
    ax = axs[i]
    tend = tends[i]
    ax.plot(plot_df['agent0_xloc'][tstart:tend],
            plot_df['agent0_yloc'][tstart:tend], 'k.',
            markersize=1, alpha=0.2)
    for b in ball_ids:
      ax.plot(plot_df[f'{b}_xloc'][tstart:tend],
              plot_df[f'{b}_yloc'][tstart:tend], '.',
              markersize=2, color=colors[b], alpha=0.5)
    rect = patches.Rectangle((-12.76, -14.1713),
                             2 * 12.76, 2 * 12.76,
                             linewidth=1, edgecolor='None',
                             facecolor='#eeeeee')
    ax.add_patch(rect)
    ax.annotate(f'{titlestrs[i]}', (-4, 13), fontsize=12)
    ax.axis('equal')
    ax.set_xlim([-15, 15])
    ax.set_ylim([-14.5, 12])
    ax.axis('off')
  plt.tight_layout()
  if savepath is not None:
    plt.savefig(f'{savepath}/interaction_snapshots.png')
    print(f'Saving to {savepath}')
  plt.show()


def simple_axis(ax):
  ax.spines.right.set_visible(False)
  ax.spines.top.set_visible(False)
  ax.yaxis.set_ticks_position('left')
  ax.xaxis.set_ticks_position('bottom')


if __name__ == "__main__":
  main()
