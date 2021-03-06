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
        self.collision_happened = False

        blueprint_library = self.world.get_blueprint_library()

        blueprint = blueprint_library.filter('model3')[0]   # Tesla Model3
        print(blueprint)

        spawn_point = random.choice(self.world.get_map().get_spawn_points())   # random spawn point

        self.vehicle = self.world.spawn_actor(blueprint, spawn_point)
        self.actor_list.append(vehicle)

        # kamera 
        blueprint2 = blueprint_library.find('sensor.camera.rgb')
        blueprint2.set_attribute('image_size_x', str(IM_WIDTH))
        blueprint2.set_attribute('image_size_y', str(IM_HEIGHT))
        blueprint2.set_attribute('fov', '110')
        spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(blueprint2, spawn_point, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: World.process_img(data))
        
        #utkozes szenzor
        blueprint3 = blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(blueprint3, carla.Transform(), attach_to=self.vehicle)
        self.actor_list.append(self.collision_sensor)
        weak_self = weakref.ref(self)
        self.collision_sensor.listen(lambda event: World.on_collission(weak_self, event))

        self.time_step = 0



    def destroy_actors(self):
        print('destroying actors')
        for actor in self.actor_list:
            actor.destroy()
        self.actor_list = []
        print('done.')

    def process_img(image):
        self.img = np.array(image.raw_data)
        self.img = self.img.reshape((IM_WIDTH, IM_HEIGHT, 4))
        self.img = self.img[:, :, :3]
        self.img = self.img[:,:,::-1]
        return self.img

    def on_collission(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.collision_happened = True

    def get_reward(self):         # TODO
        v = self.vehicle.get_velocity()
        speed = np.sqrt(v.x**2 + v.y**2)
        r_speed = -abs(speed - 13.9)     # 50 km/h ~ 13.9 m/s

    def terminal(self):           # TODO


class CARLAEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    super(CARLAEnv, self).__init__()

    self.client = carla.Client('localhost', 2000)
    self.client.set_timeout(2.0)
    self.world = World(client.get_world())
    self.settings = self.world.get_settings()
	self.settings.synchronous_mode = True
    self.settings.fixed_delta_seconds = 0.05            # Mennyi legyen?
	self.world.apply_settings(settings)

    self.action_space = spaces.Box(np.array([0,-1]), np.array([+1,+1]), dtype=np.float32)
    self.observation_space = spaces.Box(low=0, high=255, shape=(IM_HEIGHT, IM_WIDTH, N_CHANNELS), dtype=np.uint8)
	
    pygame.init()
    self.display = pygame.display.set_mode((IM_WIDTH, IM_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)

  def step(self, action):
    acc = action[0]
    steer = action[1]

    self.world.vehicle.apply_control(carla.VehicleControl(throttle=acc, steer=steer))
    self.world.tick()

    self.time_step = self.time_step + 1

    return self.world.img, self.world.get_reward(), self.world.terminal(), info

  def reset(self):
    self.world.destroy_actors()
    self.world.restart()
    return observation
  def render(self, mode='human'):
    # Pygame megjelenites
    surface = pygame.surfarray.make_surface(self.world.img.swapaxes(0,1))
    self.display.blit(surface, (0,0))
    pygame.display.flip()

  def close (self):
    # TODO

env = CARLAEnv()

model = PPO2(CnnPolicy, env, verbose=1, )
model.learn(total_timesteps=25000)

# Saving
model.save("ppo2_CARLA")

del model # remove to demonstrate saving and loading

# Loading
model = PPO2.load("ppo2_CARLA")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()