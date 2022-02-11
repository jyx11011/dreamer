from dm_control import suite
import numpy as np

domain="cartpole"
task="swingup"

env = suite.load(domain_name=domain, task_name=task)
camera = dict(quadruped=2).get(domain, 0)
f=domain+"_"+task
obs=env.physics.render(64,64,camera_id=camera)

np.save(f,obs)
#export MUJOCO_GL=egl
