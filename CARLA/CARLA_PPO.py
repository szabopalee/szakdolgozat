import glob
import os
import sys
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

import random
import time
import numpy as np
import weakref

import pygame

import gym
from gym import spaces

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2

IM_WIDTH = 160
IM_HEIGHT = 160
N_CHANNELS = 3

class World(object):
    def __init__(self, carla_world):
        self.actor_list = []
        self.world = carla_world
        self.restart()
    def restart(self):
        self.destroy_actors()

        blueprint_library = self.world.get_blueprint_library()

        blueprint = blueprint_library.filter('model3')[0]   # Tesla Model3
        print(blueprint)

        spawn_point = random.choice(self.world.get_map().get_spawn_points())   # random spawn point

        vehicle = self.world.spawn_actor(blueprint, spawn_point)
        #vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))

        self.actor_list.append(vehicle)

        # kamera 
        blueprint2 = blueprint_library.find('sensor.camera.rgb')
        blueprint2.set_attribute('image_size_x', str(IM_WIDTH))
        blueprint2.set_attribute('image_size_y', str(IM_HEIGHT))
        blueprint2.set_attribute('fov', '110')
        spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(blueprint2, spawn_point, attach_to=vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: World.process_img(data))
        
        #utkozes szenzor
        blueprint3 = blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(blueprint3, carla.Transform(), attach_to=vehicle)
        self.actor_list.append(self.collision_sensor)
        weak_self = weakref.ref(self)
        self.collision_sensor.listen(lambda event: World.on_collission(weak_self, event))

    def destroy_actors(self):
        print('destroying actors')
        for actor in self.actor_list:
            actor.destroy()
        self.actor_list = []
        print('done.')

    def process_img(image):
        i = np.array(image.raw_data)
        i2 = i.reshape((IM_WIDTH, IM_HEIGHT, 4))
        i3 = i2[:, :, :3]
        i3 = i3[:,:,::-1]

        # Pygame megjelenites
        #surface = pygame.surfarray.make_surface(i3.swapaxes(0,1))
        #display.blit(surface, (0,0))
        #pygame.display.flip()

        return i3/255

    def on_collission(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.restart()

class CARLAEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    super(CARLAEnv, self).__init__()
    self.client = carla.Client('localhost', 2000)
    self.client.set_timeout(2.0)
    self.world = World(client.get_world())

    self.action_space = spaces.Box(np.array([0,-1,0]), np.array([+1,+1,+1]), dtype=np.float32)     # Legyen fékezés vagy ne?
    self.observation_space = spaces.Box(low=0, high=255, shape=(IM_HEIGHT, IM_WIDTH, N_CHANNELS), dtype=np.uint8)

  def step(self, action):
    # TODO
    return observation, reward, done, info
  def reset(self):
    # TODO
    return observation
  def render(self, mode='human'):
    # TODO
  def close (self):
    # TODO


# Megjeleniteshez pygame
#pygame.init()

#display = pygame.display.set_mode(
#    (IM_WIDTH, IM_HEIGHT),
#    pygame.HWSURFACE | pygame.DOUBLEBUF)

env = CARLAEnv()

model = PPO2(CnnPolicy, env, verbose=1, )
model.learn(total_timesteps=25000)

# Saving
model.save("ppo2_cartpole")

del model # remove to demonstrate saving and loading

# Loading
model = PPO2.load("ppo2_cartpole")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()