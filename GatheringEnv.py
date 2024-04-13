import numpy as np
import numpy.random as rnd
import random

#todo:
#add tag penalty
class GatheringEnv:
    def __init__(self, grid_size=(10, 15), apples=20, N_apple=10, N_tagged=2, max_time_steps=1000, tag_hits=2):
        self.grid_size = grid_size

        self.grid = np.zeros(grid_size)

        self.beam_count=0
        self.both_play=0
        self.beam_rate = 0
        self.max_time_steps = max_time_steps
        self.time_step = 0

        self.apples = apples
        self.apple_positions = []
        self.N_apple = N_apple

        self.N_tagged = N_tagged

        self.tag_hits = tag_hits

        self.players = [
            {'position': (0, 0), 'orientation': 0, 'tagged': 0, 'tag_timer': 0},
            {'position': (0, 0), 'orientation': 0, 'tagged': 0, 'tag_timer': 0}
        ]
        self.reset()

    def reset(self):
        self.grid.fill(0)
        self._place_apples()
        self.players[0]['position'] = (self.grid_size[0]//2, self.grid_size[1]//2)
        self.players[1]['position'] = (self.grid_size[0]//2, self.grid_size[1]//2 +1)
        for player in self.players:
            player['tagged'] = 0
            player['tag_timer'] = 0
        self._update_grid()
        self.beam_count = 0
        self.both_play = 0
        self.beam_rate = 0
        self.time_step = 0
        return self._get_observations()

    def _place_apples(self):
        self.apple_positions = []
        self.apple_grid = np.ones(self.grid_size, dtype=int)*-1
        for _ in range(self.apples):
            position = (rnd.randint(0, self.grid_size[0]), rnd.randint(0, self.grid_size[1]))
            self.apple_grid[position] = 0
            self.apple_positions.append(position)

    def step(self, actions):

        self.time_step += 1
        rewards = [0]*len(self.players)

        self._respawn_apples()

        for i in range(len(self.players)):
          new_position = self._update_player_position(i, actions[i])
          #collect apple
          if self.grid[new_position] == 3 and self.players[i]['tagged'] == 0: #there is an apple and not tagged
              self.grid[new_position] == 0
              self.apple_grid[new_position] = self.N_apple
              rewards[i] += 1  # Reward for collecting an apple

        #todo update for more agents
        if self.players[0]['tagged'] < self.tag_hits and self.players[1]['tagged'] < self.tag_hits:
            self.both_play += 1


        # respawn a player if tagged for enough time
        for i, player in enumerate(self.players):
            if player['tag_timer'] > 0:
                player['tag_timer'] -= 1
                if player['tag_timer'] == 0:
                    player['tagged'] = 0
                    self.respawn_agent(i)

        # TODO could also add other goals per agent or global collected_apples >= target_apples
        if self.time_step >= self.max_time_steps:
            dones = [True] * len(self.players)
            if self.both_play == 0:
                print(self.beam_count)
            else:
                self.beam_rate = self.beam_count/self.both_play
        else:
            dones = [False] * len(self.players)

        return self._get_observations(), rewards, dones

    def _update_grid(self):
      for i, player in enumerate(self.players):
         player = self.players[i]
         if player['tagged'] == 0:
             self.grid[player['position']] = i+1  # position player i

      for position in self.apple_positions:
         if self.apple_grid[position] == 0: #check if not removed
             self.grid[position] = 3  # place apples
         else:
             self.grid[position] = 0

    def _get_observations(self):
        return np.copy(self.grid), np.copy(self.grid)


    def _update_player_position(self, player_index, action):

        # possible actions
        # step forward:  0
        # step backward: 1
        # step left:     2
        # step right:    3
        # rotate left:   4
        # rotate right:  5
        # use beam:      6
        # stand still    7

        # possible orrientations
        # orientation: up:0, right:1, down:2, left:3

        player = self.players[player_index]
        position = player['position']

        if player['tagged'] == self.tag_hits: #tagged then skip turn
            return position

        orientation = player['orientation']
        x=player['position'][0]
        y=player['position'][1]

        if action == 6:
            self.beam_count += 1
            self._apply_beam(player_index)
            return position

        #stand still
        if action == 7:
            return position
        next_position = (-1,-1)
        if action == 0:  # forward
            if orientation == 0:  # up
               next_position = (x,y-1)
            elif  orientation == 1: #right
                next_position = (x+1, y)
            elif  orientation == 2: #down
                next_position = (x, y+1)
            else: #left
                next_position = (x-1, y)

        elif action == 1:  # backward
            if orientation == 0:  # up
               next_position = (x,y+1)
            elif  orientation == 1: #right
                next_position = (x-1, y)
            elif  orientation == 2:  #down
                next_position = (x, y-1)
            else: #left
                next_position = (x+1, y)

        elif action == 2:  # stepleft
            if orientation == 0:  # up
               next_position = (x-1,y)
            elif orientation == 1: #right
                next_position = (x, y-1)
            elif  orientation == 2:  #down
                next_position = (x+1, y)
            else: #left
                next_position = (x, y+1)

        elif action == 3:  # stepright
            if orientation == 0:  # up
               next_position = (x+1,y)
            elif orientation == 1: #right
                next_position = (x, y+1)
            elif  orientation == 2:  #down
                next_position = (x-1, y)
            else: #left
                next_position = (x, y-1)

        # orientation: up:0, right:1, down:2, left:3
        elif action == 4:  # rotateleft
            if orientation == 0:  # up
               player['orientation'] = 3
            elif orientation == 1: #right
                player['orientation'] = 0
            elif  orientation == 2:  #down
                player['orientation'] = 1
            else: #left
                player['orientation'] = 2

        elif action == 5:  # rotateright
            if orientation == 0:  # up
               player['orientation'] = 1
            elif orientation == 1: #right
                player['orientation'] = 2
            elif  orientation == 2:  #down
                player['orientation'] = 3
            else: #left
                player['orientation'] = 0

        other_indexes = [i for i in range(1, len(self.players)+1) if i != player_index]

        # account for out of grid cases or stepping on the other player
        if not(0 <= next_position[0] < self.grid_size[0] and
            0 <= next_position[1] < self.grid_size[1] and
            self.grid[next_position] not in other_indexes):
            next_position = position

        self.grid[next_position] = 1 + player_index

        return next_position

    def _apply_beam(self, shooter_index):
        shooter = self.players[shooter_index]
        target_index = 1 - shooter_index
        target = self.players[target_index]

        hit = False

        if shooter['orientation'] == 0:  # Up
            if (shooter['position'][0] == target['position'][0] and
                shooter['position'][1] > target['position'][1]):
                hit = True
        elif shooter['orientation'] == 1:  # Right
            if (shooter['position'][1] == target['position'][1] and
                shooter['position'][0] < target['position'][0]):
                hit = True
        elif shooter['orientation'] == 2:  # Down
            if (shooter['position'][0] == target['position'][0] and
                shooter['position'][1] < target['position'][1]):
                hit = True
        elif shooter['orientation'] == 3:  # Left
            if (shooter['position'][1] == target['position'][1] and
                shooter['position'][0] > target['position'][0]):
                hit = True

        if hit:
            target['tagged'] += 1
            if target['tagged'] == self.tag_hits:
                target['tag_timer'] = self.N_tagged
                self.grid[target['position']] = 0 #remove player
    def _respawn_apples(self):
        for position in self.apple_positions:
            if self.apple_grid[position] > 0:
                self.apple_grid[position] -= 1
                if self.apple_grid[position] == 0:
                    self.grid[position] = 3 # respawn
    def respawn_agent(self, tagged_player_index):

        new_position = (random.randint(0, self.grid_size[0]-1), random.randint(0, self.grid_size[1])-1)
        while self.grid[new_position] != 0:
            new_position = (random.randint(0, self.grid_size[0]-1), random.randint(0, self.grid_size[1])-1)

        self.players[tagged_player_index]['position'] = new_position
        self.players[tagged_player_index]['tagged'] = 0
        self.grid[new_position] = tagged_player_index
