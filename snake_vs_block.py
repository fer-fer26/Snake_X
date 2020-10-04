import pygame
import sys
from pygame.locals import *
from copy import deepcopy
import math
import numpy as np
from collections import deque
import random
from PIL import Image
from color import tile_color
from collections import deque




class Snake_vs_block:
    def __init__(self, arg):
        self.play = arg['play']
        self.enable_img = arg['enable_img']
        self.draw_screen = arg['draw_screen']

        self.finish = False
        self.score = 0

        #initilize
        pygame.init()
        
        self.width = 450
        self.height = 800
        if self.draw_screen:
            self.screen = pygame.display.set_mode((self.width, self.height))

        #for ai
        self.field = [[[0 for i in range(5)] for i in range(9)] for j in range(4)]

        #param
        self.body_rad = 10
        self.body_pos = (int(self.width / 2), 450)
        self.cont_point = [[np.cos(math.pi/6 * i) * self.body_rad, np.sin(math.pi/6 * i) * self.body_rad] for i in range(5,14)]
        self.body_length = 4
        if self.play:
            self.body_orbit = [self.body_pos[0] for i in range(1000)]

        if self.play:
            self.body_color = (181, 11, 210)
            self.background_color = (0, 0, 0)
            self.wall_color = (255, 255, 255)

        else:
            self.body_color = (0, 0, 0)
            self.background_color = (255, 255, 255)
            self.wall_color = (0, 0, 0)

        self.line_num = 5
        self.speed = 0.15
        self.move = True
        self.poss_dire = [True, True]

        self.x = self.body_pos[0]
        self.real_x = self.body_pos[0]
        self.last_x = 0
        self.num = 0

        self.tile_list = deque(maxlen=25)
        self.tile_sepa = 100
        self.tile_sepa_count = 100
        self.tile_size = int(self.width / self.line_num)
        self.tile_pos = [i * self.tile_size for i in range(self.line_num)]
        self.dec_num = [0, -1, False]
        self.dec_sepa = 150
        self.tile_count = 0
        self.tile_sum_sepa_count = 0

        self.wall_list = deque(maxlen=25)
        self.wall_width = 5

        self.bait_list = deque(maxlen=25)
        self.bait_rad = 8
        self.bait_color = self.body_color

        self.action_list = [i * 0.2 for i in range(-5,6)]
        self.action_list_2 = [i * self.tile_size + self.tile_size/2 for i in range(5)]

        #font
        if self.play:
            self.fonts_tile = []
            self.fonts_tile_size = 100
            for i in range(1, 51):
                self.fonts_tile.append(pygame.font.Font(None, self.fonts_tile_size).render(str(i), True, (0, 0, 0)))

            self.fonts_body = []
            self.fonts_body_size = 20
            for i in range(1000):
                self.fonts_body.append(pygame.font.Font(None, self.fonts_body_size).render(str(i), True, (255, 255, 255)))

        #tile color
        global tile_color
        self.tile_color = []
        if self.play:
            self.tile_color = tile_color

        else:
            for i in range(1, 51):
                self.tile_color.append((255-i*5, 255-i*5, 255-i*5))

    def control_img(self, action):
        self.x += self.action_list[action]

        if not self.poss_dire[0] and self.x - self.last_x > 0 or not self.poss_dire[1] and self.last_x - self.x > 0:
            self.x = self.last_x

        if self.x < 1:
            self.x = 1

        elif self.x > self.width:
            self.x = self.width

        self.last_x = self.x

    def control(self, action):
        if self.x > self.action_list_2[action]:
            self.x -= 1

        elif self.x < self.action_list_2[action]:
            self.x += 1

        if self.x < 1:
            self.x = 1

        elif self.x > self.width:
            self.x = self.width

        self.last_x = self.x

    def draw_player(self):
        #contact determinatioin with tiles
        y_range = (self.body_pos[1] - self.body_rad - self.tile_size, self.body_pos[1] + self.body_rad)
        for i in range(len(self.tile_list)):
            if self.tile_list[i][0] != 0 and y_range[0] <= self.tile_list[i][2] <= y_range[1]:
                tile = [self.tile_list[i][1], self.tile_list[i][2], self.tile_list[i][1] + self.tile_size, self.tile_list[i][2] + self.tile_size]
                for point in range(len(self.cont_point)):
                    if tile[1] <= self.body_pos[1] + self.cont_point[point][1] <= tile[3] and tile[0] <= self.x + self.cont_point[point][0] <= tile[2]:
                        if point == 4:
                            self.move = False

                            if self.dec_num[0] % self.dec_sepa == 0 or (self.dec_num[0] == 0 and self.dec_num[1] == self.tile_list[i][3]):
                                self.tile_list[i][0] -= 1
                                self.dec_num[1] = self.tile_list[i][3]
                                self.dec_num[2] = True

                                self.score += 1
                                self.body_length -= 1
                                if self.body_length < 0:
                                    self.finish = True

                                if self.tile_list[i][0] == 0:
                                    self.move = True
                                    self.dec_num = [0, -1, False]

                            self.poss_dire[0] = True
                            self.poss_dire[1] = True

                        elif point > 4:
                            self.move = True
                            self.dec_num = [0, -1, False]

                            self.x = tile[0] - self.body_rad

                        elif point < 4:
                            self.move = True
                            self.dec_num = [0, -1, False]

                            self.x = tile[2] + self.body_rad

                    elif point == 4:
                        if self.dec_num[1] == self.tile_list[i][3]:
                            self.move = True
                            self.dec_num = [0, -1, False]

                    else:
                        self.poss_dire[0] = True
                        self.poss_dire[1] = True

        #contact determinatioin with walls
        y_range = (self.body_pos[1] - self.body_rad - self.tile_size * 2, self.body_pos[1] + self.body_rad)
        for i in range(len(self.wall_list)):
            if y_range[0] <= self.wall_list[i][1] <= y_range[1]:
                wall = [self.wall_list[i][0], self.wall_list[i][1], self.wall_list[i][0] + self.wall_width, self.wall_list[i][1] + self.wall_list[i][2]]
                for point in range(len(self.cont_point)):
                    if wall[1] <= self.body_pos[1] + self.cont_point[point][1] <= wall[3] and wall[0] <= self.x + self.cont_point[point][0] <= wall[2]:
                        if point == 4:
                            self.poss_dire[0] = True
                            self.poss_dire[1] = True

                        elif point > 4:
                            self.x = int(wall[0] - self.body_rad)

                        elif point < 4:
                            self.x = int(wall[2] + self.body_rad)

        #draw first body 
        if self.draw_screen:
            pygame.draw.circle(self.screen, self.body_color, (int(self.x), self.body_pos[1]), self.body_rad)

        if self.play:
            self.screen.blit(self.fonts_body[self.body_length], (int(self.x - self.fonts_body_size / 2), int(self.body_pos[1] - 30)))

            if self.move:
                self.body_orbit.append(self.x)

    def draw_body(self):
        #draw bodys except first one
        dist = 0
        ind = -1
        for i in range(1, self.body_length):
            while dist <= i * self.body_rad * 2:
                ind -= 1
                dist += math.sqrt((self.body_orbit[ind + 1] - self.body_orbit[ind]) ** 2 + self.speed ** 2)

            pygame.draw.circle(self.screen, self.body_color, (int(self.body_orbit[ind]), int(self.body_pos[1] + self.speed * -ind)), self.body_rad)

    def draw_tile(self):
        for i in range(len(self.tile_list)):
            if self.tile_list[i][0] != 0:
                if self.tile_list[i][2] >= 0:
                    if self.draw_screen:
                        self.screen.fill(self.tile_color[self.tile_list[i][0] - 1], Rect(self.tile_list[i][1], self.tile_list[i][2], self.tile_size, self.tile_size))

                else:
                    if self.draw_screen:
                        self.screen.fill(self.tile_color[self.tile_list[i][0] - 1], Rect(self.tile_list[i][1], 0, self.tile_size, self.tile_size + self.tile_list[i][2]))

                if self.play:
                    if self.draw_screen:
                        if self.tile_list[i][0] <= 9:
                            self.screen.blit(self.fonts_tile[self.tile_list[i][0] - 1], (self.tile_list[i][1] + self.tile_size / 2 - 20, self.tile_list[i][2] + self.tile_size / 2 - 30))

                        else:
                            self.screen.blit(self.fonts_tile[self.tile_list[i][0] - 1], (self.tile_list[i][1] + self.tile_size / 2 - 40, self.tile_list[i][2] + self.tile_size / 2 - 30))

                if self.move:
                    self.tile_list[i][2] += self.speed

        if self.move:
            self.tile_sepa_count += self.speed
            self.tile_sum_sepa_count += self.speed

    first = False
    rand_sepa = random.choice(range(2,4))
    def tile_appear(self):
        if (self.tile_sepa_count / self.tile_size) >= self.tile_sepa or (not self.first and self.count > 100):
            if not self.first:
                for i in range(self.line_num):
                    num = random.choice([1, 2])
                    self.tile_list.append([num, self.tile_pos[i], -self.tile_size, self.tile_count])
                    self.tile_count += 1

                    self.field[0][6][i] = num / 50

            else:
                for i in range(self.line_num):
                    num = random.choice(range(1, 51))
                    self.tile_list.append([num, self.tile_pos[i], -self.tile_size, self.tile_count])
                    self.tile_count += 1

                    self.field[0][6][i] = num / 50

            self.tile_sepa = random.choice(range(9,15))
            self.tile_sepa_count = 0
            self.rand_sepa = random.choice(range(2,4))
            self.first = True

            #wall
            self.wall_appear(self.rand_sepa)

        elif (self.tile_sepa_count / self.tile_size) >= self.rand_sepa and self.first:
            ran = [i for i in range(self.line_num)]
            pos_list = []
            for i in range(random.choice(range(1,3))):
                pos = random.choice(ran)
                num = random.choice(range(1, 51))
                self.tile_list.append([num, self.tile_pos[pos], -self.tile_size, self.tile_count])
                self.tile_count += 1

                self.field[0][6][pos] = num / 50

                for j in range(-1,2):
                    if 0 <= pos-j < len(ran):
                        ran.pop(pos-j)

                pos_list.append(pos)

            rand = random.choice(range(2,4))

            cover = 0
            if rand + self.rand_sepa >= self.tile_sepa:
                cover = (rand + self.rand_sepa) - self.tile_sepa

            #wall
            self.wall_appear(rand)

            #bait
            self.bait_appear(pos_list, rand - cover)

            self.rand_sepa = rand + self.rand_sepa
            if self.rand_sepa == self.tile_sepa - 1:
                self.rand_sepa += 1

    def draw_wall(self):
        for i in range(len(self.wall_list)):
            if self.wall_list[i][1] >= 0:
                if self.draw_screen:
                    self.screen.fill(self.wall_color, Rect(self.wall_list[i][0], self.wall_list[i][1], self.wall_width, self.wall_list[i][2]))

            else:
                if self.draw_screen:
                    self.screen.fill(self.wall_color, Rect(self.wall_list[i][0], 0, self.wall_width, self.wall_list[i][1] + self.wall_list[i][2])) 

            if self.move:
                self.wall_list[i][1] += self.speed

    def draw_bait(self):
        for i in range(len(self.bait_list)):
            if self.bait_list[i][3] != 0:
                if self.body_pos[1] - (self.body_rad + self.bait_rad) <= self.bait_list[i][2] <= self.body_pos[1] + (self.body_rad + self.bait_rad):
                    dist = (self.x - self.bait_list[i][1]) ** 2 + (self.body_pos[1] - self.bait_list[i][2]) ** 2
                    if dist <= (self.body_rad + self.bait_rad) ** 2:
                        self.bait_list[i][3] = 0

                        self.body_length += self.bait_list[i][0]

                    else:
                        if self.draw_screen:
                            pygame.draw.circle(self.screen, self.bait_color, (int(self.bait_list[i][1]), int(self.bait_list[i][2])), self.bait_rad)

                else:
                    if self.draw_screen:
                        pygame.draw.circle(self.screen, self.bait_color, (int(self.bait_list[i][1]), int(self.bait_list[i][2])), self.bait_rad)

                if self.play:
                    self.screen.blit(self.fonts_body[self.bait_list[i][0]], ((int(self.bait_list[i][1]) - self.fonts_body_size / 2, int(self.bait_list[i][2]) - 30)))

                if self.move:
                    self.bait_list[i][2] += self.speed

    def bait_appear(self, pos_list, rand):
        poss_pos = [i for i in range(self.line_num) if i not in pos_list]
        if 4 >= random.choice(range(10)):
            if 2 >= random.choice(range(10)):
                for i in range(2):
                    pos = random.choice(range(len(poss_pos)))
                    num = random.choice(range(1, 6))
                    self.bait_list.append([num, self.tile_pos[poss_pos[pos]] + self.tile_size / 2, -self.tile_size / 2, 1])

                    self.field[1][6][poss_pos[pos]] = num / 5

                    poss_pos.pop(pos)

            else:
                pos = random.choice(poss_pos)
                num = random.choice(range(1, 6))
                self.bait_list.append([num, self.tile_pos[pos] + self.tile_size / 2, -self.tile_size / 2, 1])

                self.field[1][6][pos] = num / 5

        for i in range(rand - 2):
            poss_pos = [i for i in range(self.line_num)]
            if 4 >= random.choice(range(10)):
                if 2 >= random.choice(range(10)):
                    for i in range(2):
                        pos = random.choice(range(len(poss_pos)))
                        num = random.choice(range(1, 6))
                        self.bait_list.append([num, self.tile_pos[poss_pos[pos]] + self.tile_size / 2, -self.tile_size / 2 - self.tile_size * (i + 1), 1])

                        self.field[1][6+i+1][poss_pos[pos]] = num / 5

                        poss_pos.pop(pos)

                else:
                    pos = random.choice(poss_pos)
                    num = random.choice(range(1, 6))
                    self.bait_list.append([num, self.tile_pos[pos] + self.tile_size / 2, -self.tile_size / 2 - self.tile_size * (i + 2), 1])

                    self.field[1][6+i+2][pos] = num / 5

    def wall_appear(self, rand):
        ran = [i for i in range(1, len(self.tile_pos))]
        for i in range(random.choice([0,1,1,2,2,2,2,2,2])):
            length = random.choice(range(1, min(rand, 3)))
            pos = random.choice(ran)
            if i != 1:
                ran.pop(pos-1)

            self.wall_list.append([self.tile_pos[pos] - self.wall_width / 2, -self.tile_size * (length + 1), length * self.tile_size])

            if length == 2:
                self.field[2][8][pos] = 1
                self.field[2][7][pos] = 1

            else:
                self.field[2][7][pos] = 1

    def init_bait(self):
        rand = random.choice([1, 2])
        poss_pos = [i for i in range(self.line_num)]
        for i in range(rand):
            pos = random.choice(range(len(poss_pos)))
            num = random.choice(range(1, 6))
            self.bait_list.append([num, self.tile_pos[poss_pos[pos]] + self.tile_size / 2, 30, 1])

            self.field[1][5][poss_pos[pos]] = num / 5

            poss_pos.pop(pos)

        poss_pos = [i for i in range(self.line_num)]
        for i in range(3 - rand):
            pos = random.choice(range(len(poss_pos)))
            num = random.choice(range(1, 6))
            self.bait_list.append([num, self.tile_pos[poss_pos[pos]] + self.tile_size / 2, 130, 1])

            self.field[1][4][poss_pos[pos]] = num / 5

            poss_pos.pop(pos)

    def screenshot(self):
        img = pygame.surfarray.array3d(self.screen)
        img = img[::4, ::4]

        return img

    def get_state(self):
        return np.array(self.field)[:, :6, :]

    def draw(self):
        wall = True
        tile = True

        #player
        self.draw_player()
        if self.play:
            self.draw_body()

        #tile
        self.tile_appear()
        if not tile:
            self.tile_list = deque(maxlen=25)
        self.draw_tile()

        if self.dec_num[2]:
            self.dec_num[0] += 1

        #wall
        if not wall:
            self.wall_list = deque(maxlen=25)
        self.draw_wall()

        #bait
        if self.count == 0:
            self.init_bait()
            
        self.draw_bait()
        
    def reset(self):
        self.finish = False
        self.score = 0
        self.last_score = 0

        #param
        self.body_pos = (int(self.width / 2), 450)
        self.body_length = 4
        if self.play:
            self.body_orbit = [self.body_pos[0] for i in range(1000)]

        self.move = True
        self.poss_dire = [True, True]

        self.x = self.body_pos[0]
        self.field = [[[0 for i in range(5)] for i in range(9)] for j in range(4)]

        self.tile_list = deque(maxlen=25)
        self.dec_num = [0, -1, False]
        self.tile_count = 0

        self.wall_list = deque(maxlen=25)

        self.bait_list = deque(maxlen=25)

        self.first = False
        self.rand_sepa = random.choice(range(2,4))

        self.count = 0

        #background
        if self.draw_screen:
            self.screen.fill(self.background_color)

        #draw
        self.draw()

        self.count += 1

        #update display
        if self.draw_screen:
            pygame.display.update()

        #self.field[0][4] = 3
        self.field[3][0][2] = 1

        if self.enable_img:
            return self.screenshot()

        else:
            return self.get_state()

    last_score = 0
    last_length = 4
    def reward(self):
        if self.finish:
            reward = -1

        elif self.body_length > self.last_length:
            reward = 1

        else:
            reward = 0

        return reward

    def step(self, action):
        self.last_score = self.score
        self.last_length = self.body_length

        while self.tile_sum_sepa_count <= self.tile_size:
            if self.draw_screen:
                for event in pygame.event.get():
                    if event.type == QUIT:
                        pygame.quit()
                        sys.exit()

            #action
            self.control(action)

            #background
            if self.draw_screen:
                self.screen.fill(self.background_color)

            if self.finish:
                break

            #draw
            self.draw()

            #update display
            if self.draw_screen:
                pygame.display.update()

            self.count += 1

        self.tile_sum_sepa_count = 0

        if self.enable_img:
            return self.screenshot(), self.reward(), self.finish, 1

        else:
            for i in range(4):
                self.field[i].pop(0)
                self.field[i].append([0 for j in range(5)])

            self.field[3][0][int(self.x / self.tile_size)] = 3

            return self.get_state(), self.reward(), self.finish, 1