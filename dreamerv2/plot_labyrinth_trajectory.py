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
SAVERATE = 20  # record_every_k_timesteps param in logging_params in adaptgym.wrapped.ADMC
PREFILL = 1e3  # prefill param in config.yaml


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, help='Path to the log directory',
                        default='~/logdir/admc_sphero_novel_object_2ball/p2e/11')
    parser.add_argument('--outdir', type=str, help='Path to the output directory for plots',
                        default='~/logdir/admc_sphero_novel_object_2ball/p2e/11/plots')
    return parser.parse_args()

def main():
  args = parse_arguments()
  logdir = args.logdir
  outdir = args.outdir
  os.makedirs(outdir, exist_ok=True)
  expid = '/'.join(logdir.split('/')[-3:])
  fn = f'{logdir}/log_train_env0.csv'
  df = pd.read_csv(fn)

  tstart = int(0)
  tend = int(15e6)

  plot_indiv_labyrinth_expt(df, tstart, tend, expid, SAVERATE,
                            savepath=outdir)

def plot_indiv_labyrinth_expt(plot_df, tstart, tend,
                              expid, saverate,
                              savepath=None):
  """Plot map of agent and object locations over the session,
  and cumulative collisions vs time."""
  plt.figure(figsize=(8, 3))
  ax = plt.subplot(1, 2, 1)

  plt.plot(plot_df['agent0_xloc'][tstart:tend],
           plot_df['agent0_yloc'][tstart:tend], 'k.',
           markersize=1, alpha=0.2)

  ax.axis('equal')
  plt.xlim(-30, 30)
  plt.ylim(-30, 30)
  plt.axis('off')
  plt.suptitle(f'{expid} ({len(plot_df["agent0_xloc"][tstart:tend])*SAVERATE:.1e} steps)')

  if savepath is not None:
    plt.savefig(f'{savepath}/interaction_summary.png')
    print(f'Saving to {savepath}')
  plt.show()


if __name__ == "__main__":
  main()
