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

#IM_WIDTH = 640
#IM_HEIGHT = 480

# Global functions

pygame.init()

display = pygame.display.set_mode(
    (640, 480),
    pygame.HWSURFACE | pygame.DOUBLEBUF)

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
        vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))   # arra utasítjuk egyelőre, hogy csak menjen előre

        self.actor_list.append(vehicle)

        # kamera 
        blueprint2 = blueprint_library.find('sensor.camera.rgb')
        blueprint2.set_attribute('image_size_x', '640')
        blueprint2.set_attribute('image_size_y', '480')
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
        self.collision_sensor.listen(lambda event: World.on_collission(weak_self, event)) # weak ref??

    def destroy_actors(self):
        print('destroying actors')
        for actor in self.actor_list:
            actor.destroy()
        self.actor_list = []              # ????
        print('done.')


    def process_img(image):
        #image.save_to_disk('output/%06d.png' % image.frame)   # fileok kimentese png-be
        #egy lehetseges feldolgozas: 
        i = np.array(image.raw_data)
        i2 = i.reshape((480, 640, 4))		#  160 160
        i3 = i2[:, :, :3]
        i3 = i3[:,:,::-1]

        surface = pygame.surfarray.make_surface(i3.swapaxes(0,1))
        display.blit(surface, (0,0))
        pygame.display.flip()

        return i3/255

    def on_collission(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.restart()

client = carla.Client('localhost', 2000)
client.set_timeout(2.0)

world = World(client.get_world())

while 1:
    time.sleep(1)