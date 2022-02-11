from dm_control import suite

domain="cartpole"
task="swingup"

env = suite.load(domain_name=domain, task_name=task)
camera = dict(quadruped=2).get(domain, 0)

f = open(domain+"_"+task+".txt", "a")
f.write(env.physics.render(64,64,camera_id=camera))

