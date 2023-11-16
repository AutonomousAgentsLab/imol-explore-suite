"""Fiddle around with environments."""
from adaptgym import wrapped
from adaptgym.fiddle_env import display, interactive, gif
## May need to do: pip install pyopengl==3.1.6

if __name__ == "__main__":

  # name = 'admc_sphero_distal_column1'
  name = 'admc_sphero_labyrinth_black'
  # name = 'admc_rodent_multiagent_novel_objects_step2_single_magenta'
  envname, taskname = name.split('_', 1)


  if envname == 'cdmc':
    env = wrapped.CDMC(taskname)
  elif envname == 'ddmc':
    env = wrapped.DDMC(taskname)
  elif envname == 'admc':
    env = wrapped.ADMC(taskname)

  mode = 'interactive'
  if mode == 'display':
    display(env)
  elif mode == 'gif':
    gif(env)
  elif mode == 'interactive':
    interactive(env, envname)