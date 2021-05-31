#######################################################
# Copyright (c) 2021 Qihao Shan, All Rights Reserved #
#######################################################

import random
import numpy as np
import math
import matplotlib.pyplot as plt
import dm_objects


TIME_STEP = 0.01
TSPF = 20
SAMPLE_NUM = 10


class epuck:
    def __init__(self):
        # constants
        self.vl = 0.16
        self.va = 0.75
        self.r = 0.035
        self.comm_dist = 0.5  # 0.5
        # movement
        self.dir = random.random() * 2 * math.pi
        self.dir_v = np.array([math.sin(self.dir), math.cos(self.dir)])
        self.turn_dir = int(random.random() * 2) * 2 - 1
        self.walk_state = int(random.random() * 2)
        self.walk_timer = 0


class arena:
    def __init__(self, fill_ratio, pattern, block_mean_width, block_std_width, hypotheses, dm_object, N=20, dim=np.array([2, 2]), axis=None):
        # initialise arena
        self.length = int(dim[0]/0.1)
        self.width = int(dim[1]/0.1)
        self.tile_array = self.generate_pattern(self.length, self.width, fill_ratio, pattern, block_mean_width, block_std_width)
        # initialise agents
        self.robots = []
        self.coo_array = np.array([]).reshape([0, 2])
        self.n = float(N)
        self.dim = dim
        for i in range(N):
            coo = np.array([random.random(), random.random()] * self.dim)
            self.robots.append(epuck())
            while self.collision_detect(self.coo_array, coo):
                coo = np.array([random.random(), random.random()] * self.dim)
                #print('new position', i, coo)
            self.coo_array = np.vstack((self.coo_array, coo))
        self.axis = axis
        self.hypotheses = hypotheses
        self.dm_object = dm_object
        self.dm_object.tile_array = self.tile_array
        self.dm_object.comm_dist = self.robots[0].comm_dist

    def generate_pattern(self, length, width, fill_ratio, pattern, block_mean_width, block_std_width):
        tiles = np.zeros(width * length)
        if pattern == 'Block' and block_mean_width > 0:
            tiles = tiles.reshape((length, width))
            target_black_tiles = round(width * length * fill_ratio)
            # put blocks
            while target_black_tiles-np.sum(tiles) > 0:
                block_width = abs(round(np.random.normal(block_mean_width, block_std_width)))
                if target_black_tiles-np.sum(tiles) - block_width**2 > -20 or \
                        target_black_tiles-np.sum(tiles) - block_width**2 > -(block_mean_width - 3 * block_std_width)**2:
                    pos = [int(random.random()*(length-block_width)), int(random.random()*(width-block_width))]
                    pos_end = [min(pos[0]+block_width, length), min(pos[1]+block_width, width)]
                    tiles[pos[0]:pos_end[0], pos[1]:pos_end[1]] = 1
            # randomly distributing remaining black tiles
            if target_black_tiles > np.sum(tiles):
                while target_black_tiles > np.sum(tiles):
                    pos = [int(random.random()*length), int(random.random()*width)]
                    if self.check_neighbouring_tiles(tiles, pos, 1):
                        tiles[pos[0], pos[1]] = 1
            elif target_black_tiles < np.sum(tiles):
                while target_black_tiles < np.sum(tiles):
                    pos = [int(random.random() * length), int(random.random() * width)]
                    if self.check_neighbouring_tiles(tiles, pos, 0):
                        tiles[pos[0], pos[1]] = 0
        else:
            # random
            tiles[:int(tiles.size * fill_ratio)] = 1
            tiles = np.random.permutation(tiles)
            tiles = tiles.reshape((length, width))
        print('num of black tiles ', np.sum(tiles))
        return tiles

    def check_neighbouring_tiles(self, tiles, coo, value):
        appended_tile_array = np.ones((self.length + 2, self.width + 2)) * (1 - value)
        appended_tile_array[1:self.length + 1, 1:self.width + 1] = tiles
        #vicinity_block = appended_tile_array[coo[0]:coo[0]+3, coo[1]:coo[1]+3]
        vicinity_block = np.array([appended_tile_array[coo[0]+1, coo[1]+1], appended_tile_array[coo[0], coo[1]+1],
                                   appended_tile_array[coo[0]+1, coo[1]], appended_tile_array[coo[0]+1, coo[1]+2],
                                   appended_tile_array[coo[0]+2, coo[1]+1]])
        if np.any(vicinity_block == value):
            return True
        else:
            return False

    def oob(self, coo):
        # out of bound
        if self.robots[0].r < coo[0] < self.dim[0] - self.robots[0].r \
                and self.robots[0].r < coo[1] < self.dim[1] - self.robots[0].r:
            return False
        else:
            #print('oob ', coo)
            return True

    def collision_detect(self, coo_array, new_coo):
        # check if new_coo clip with any old coo, or oob
        if self.oob(new_coo):
            return True
        elif len(self.robots) == 1:
            return False
        else:
            dist_array = np.sqrt(np.sum((coo_array - new_coo) ** 2, axis=1))
            if np.min(dist_array) < 2 * self.robots[0].r:
                #print(dist_array)
                #print('collision ')
                return True
            else:
                return False

    def random_walk(self):
        for i in range(len(self.robots)):
            self.robots[i].walk_timer -= 1
            new_coo = self.coo_array[i, :] + self.robots[i].dir_v * self.robots[
                i].vl * TIME_STEP * 10  # check collision in next 10 time steps
            coo_array_ = np.delete(self.coo_array, i, 0)
            if self.robots[i].walk_state == 0:
                # going straight
                if (not self.collision_detect(coo_array_, new_coo)) and self.robots[i].walk_timer > 0:
                    self.coo_array[i, :] += self.robots[i].dir_v * self.robots[i].vl * TIME_STEP
                else:
                    # start turning
                    self.robots[i].walk_state = 1
                    self.robots[i].walk_timer = random.random() * 4.5 / TIME_STEP
                    self.robots[i].turn_dir = int(random.random() * 2) * 2 - 1
            else:
                # turning
                if self.robots[i].walk_timer > 0:
                    self.robots[i].dir += self.robots[i].turn_dir * self.robots[i].va * TIME_STEP
                    self.robots[i].dir_v = np.array([math.sin(self.robots[i].dir), math.cos(self.robots[i].dir)])
                elif self.collision_detect(coo_array_, new_coo):
                    self.robots[i].walk_timer = random.random() * 4.5 / TIME_STEP
                    self.robots[i].turn_dir = int(random.random() * 2) * 2 - 1
                else:
                    # start going straight
                    self.robots[i].walk_state = 0
                    self.robots[i].walk_timer = np.random.exponential(scale=40) / TIME_STEP
                    self.robots[i].dir_v = np.array([math.sin(self.robots[i].dir), math.cos(self.robots[i].dir)])

    def plot_arena(self, t_step):
        if t_step % TSPF == 0:
            self.axis[0, 0].cla()
            self.axis[0, 1].cla()
            self.axis[1, 0].cla()
            self.axis[1, 1].cla()
            self.axis[0, 1].set_ylim([-1, len(self.hypotheses)])

            self.axis[0, 0].set_title('timestep '+str(t_step))
            for i in range(self.width):
                for j in range(self.length):
                    if self.tile_array[i, j] == 1:
                        self.axis[0, 0].fill_between([i*0.1, (i+1)*0.1], [j*0.1, j*0.1], [(j+1)*0.1, (j+1)*0.1], facecolor='k')

            for i in range(len(self.robots)):
                circle = plt.Circle((self.coo_array[i, 0], self.coo_array[i, 1]), self.robots[0].r, color='r', fill=False)
                self.axis[0, 0].add_artist(circle)
                self.axis[0, 0].plot(np.array([self.coo_array[i, 0], self.coo_array[i, 0]+self.robots[i].dir_v[0]*0.05]), np.array([self.coo_array[i, 1], self.coo_array[i, 1]+self.robots[i].dir_v[1]*0.05]),'b')
            self.axis[0, 0].plot(self.coo_array[self.dm_object.diss_state_array == 0, 0],
                              self.coo_array[self.dm_object.diss_state_array == 0, 1], 'ro', markersize=3)
            self.axis[0, 0].plot(self.coo_array[self.dm_object.diss_state_array == 1, 0],
                              self.coo_array[self.dm_object.diss_state_array == 1, 1], 'bo', markersize=3)
            self.axis[0, 1].plot(self.dm_object.decision_array, 'r*')
            self.axis[0, 0].set(xlim=(0, self.dim[0]), ylim=(0, self.dim[1]))
            self.axis[0, 0].set_aspect('equal', adjustable='box')

            # strategy specific monitoring, comment out if error
            if self.dm_object.dm_type == 'dc':
                self.axis[1, 0].plot(self.hypotheses[self.dm_object.decision_array], self.dm_object.quality_array, 'o')
                self.axis[1, 0].set(xlim=(0, 1))
            elif self.dm_object.dm_type == 'os':
                self.axis[1, 0].plot(self.dm_object.quality_mat_self[0, :], 'r')
                self.axis[1, 0].plot(self.dm_object.quality_mat_neigh[0, :], 'b')
                self.axis[1, 0].plot(dm_objects.normalise(self.dm_object.quality_mat_self[0, :] * self.dm_object.quality_mat_neigh[0, :]), 'm')
                self.axis[1, 1].plot(self.dm_object.neighbour_tag_record[0, :], 'o')
                self.axis[1, 1].set(ylim=(0, 20))

            plt.draw()
            plt.pause(0.001)
        else:
            pass
