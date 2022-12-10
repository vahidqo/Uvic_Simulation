import gym
import numpy as np
import ma_gym
from ma_gym.envs.utils.action_space import MultiAgentActionSpace
from ma_gym.envs.utils.observation_space import MultiAgentObservationSpace
from pygame.event import pump
import pygame
import math

class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self):

        
        #super(MultiAgentEnv, self).__init__()

        self.fps = 5
        self.fps_clock = pygame.time.Clock()
        
        self.screen = pygame.display.set_mode((500, 500))

        self.ag = [pygame.image.load(r'car.png'),
                   pygame.image.load(r'car1.png'),
                   pygame.image.load(r'scar.png')]

        self.img1 = pygame.image.load("store.png")
        self.img2 = pygame.image.load("store.png")
        self.img3 = pygame.image.load("placeholder.png")
        self.img4 = pygame.image.load("village.png")
        self.img5 = pygame.image.load("village.png")
        self.img6 = pygame.image.load("village.png")

        self.screen.blit(self.img2, (0,436))
        self.screen.blit(self.img3, (200,200))
        self.screen.blit(self.img4, (436,0))
        self.screen.blit(self.img5, (436,186))
        self.screen.blit(self.img6, (436,436))
        
        #display.set_caption('RUN')

        self.total_time = 48
        self.n_supplier = 2
        self.n_hub = 1
        self.n_customer = 3
        
        self.customer_or = np.zeros([self.n_supplier + 1, self.total_time+1, 2])

        self.supplier_o = np.zeros([self.n_supplier + 1, 2])

        self.capacity = 10

        self.order_r = 0
        self.order_d = 0

       

        self.n_agents = 3
        self.agents = ["agent_" + str(r) for r in range(self.n_agents)]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.n_agents))))

        self.xy = np.zeros([self.n_agents + 1, 2])

        self.x = [0,0,0,200,436,436,436]
        self.y = [0,0,436,200,0,186,436]

        self.m_speed = 10
        self.dis = np.array([[0,0,0,0,0,0,0],
                             [0,0,15,30,0,0,0],
                             [0,15,0,20,0,0,0],
                             [0,30,20,0,25,40,15],
                             [0,0,0,25,0,15,0],
                             [0,0,0,40,15,0,10],
                             [0,0,0,15,0,10,0],])


        self.obs_dim = (self.n_supplier * 2) + 8
        self.state_dim = (self.n_supplier * 2) + 8 * self.n_agents

        self.agent_v = np.zeros(self.n_agents)
        self.agent_pr = np.zeros(self.n_agents)
        self.agent_oo = np.zeros(self.n_agents)
        self.agent_p = np.zeros(self.n_agents)
        self.agent_dd = np.zeros(self.n_agents)

        box_lows = np.zeros(self.state_dim)
        box_highs = np.hstack([
            np.repeat(self.n_customer + self.n_hub + self.n_supplier, 1),
            np.repeat(10, 1),
            np.repeat(self.n_customer + self.n_hub + self.n_supplier, 1),
            np.repeat(10, 1),
            np.repeat(self.m_speed, 1),
            np.repeat(self.n_customer + self.n_hub + self.n_supplier, 1),
            np.repeat(self.n_customer + self.n_hub + self.n_supplier, 1),
            np.repeat(10, 1),
            np.repeat(10, 1),
            np.repeat(self.n_customer + self.n_hub + self.n_supplier, 1),
            np.repeat(10, 1),
            np.repeat(self.n_customer + self.n_hub + self.n_supplier, 1),
            np.repeat(self.m_speed, 1),
            np.repeat(self.n_customer + self.n_hub + self.n_supplier, 1),
            np.repeat(self.n_customer + self.n_hub + self.n_supplier, 1),
            np.repeat(10, 1),
            np.repeat(10, 1),
            np.repeat(self.n_customer + self.n_hub + self.n_supplier, 1),
            np.repeat(10, 1),
            np.repeat(self.n_customer + self.n_hub + self.n_supplier, 1),
            np.repeat(self.m_speed, 1),
            np.repeat(self.n_customer + self.n_hub + self.n_supplier, 1),
            np.repeat(self.n_customer + self.n_hub + self.n_supplier, 1),
            np.repeat(10, 1),
            np.repeat(10, 1),
            np.repeat(self.n_customer + self.n_hub + self.n_supplier, 1),
            np.repeat(10, 1),
            np.repeat(self.n_customer + self.n_hub + self.n_supplier, 1),
        ])

        box_low = np.zeros(self.obs_dim)
        box_high = np.hstack([
            np.repeat(self.n_customer + self.n_hub + self.n_supplier, 1),
            np.repeat(10, 1),
            np.repeat(self.n_customer + self.n_hub + self.n_supplier, 1),
            np.repeat(10, 1),
            np.repeat(self.m_speed, 1),
            np.repeat(self.n_customer + self.n_hub + self.n_supplier, 1),
            np.repeat(self.n_customer + self.n_hub + self.n_supplier, 1),
            np.repeat(10, 1),
            np.repeat(10, 1),
            np.repeat(self.n_customer + self.n_hub + self.n_supplier, 1),
            np.repeat(10, 1),
            np.repeat(self.n_customer + self.n_hub + self.n_supplier, 1),
        ])

        self.observation_space = MultiAgentObservationSpace([gym.spaces.Box(low=box_low, high=box_high, dtype=np.float64,)
                                                             for _ in range(self.n_agents)])
        
#        self.observation_spaces = dict(
 #           zip(
  #              self.agents,
   #             [
    #                gym.spaces.Box(
     #                   low=box_low,
      #                  high=box_high,
       #                 dtype=np.float,
        #            )
         #       ] * self.n_agents
          #  )
        #)

        #self.action_space = MultiAgentActionSpace([gym.spaces.MultiDiscrete([self.n_customer + self.n_hub + self.n_supplier+1, 2]) for _ in range(self.n_agents)])
        self.action_space = MultiAgentActionSpace([gym.spaces.Discrete((self.n_customer + self.n_hub + self.n_supplier+1)* 2) for _ in range(self.n_agents)])

        #self.action_spaces = dict(
         #       zip(
          #          self.agents,
           #         [gym.spaces.MultiDiscrete([self.n_customer + self.n_hub + self.n_supplier, 1])] * self.n_agents,
            #    )
            #)
        
        self.state_space = gym.spaces.Box(
                        low=box_lows,
                        high=box_highs,
                        dtype=np.float64,
                    )
    
    def action_space_sample(self):
        return [agent_action_space.sample() for agent_action_space in self.action_space]

    #def observation_space(self, agent):
    #    return self.observation_spaces[agent]

    #def action_space(self, agent):
    #    return self.action_spaces[agent]

    #def seed(self, seed=None):
    #    self.np_random, seed = gym.utils.seeding.np_random(seed)

    def step(self, action):
      #pump()

      assert len(action) == self.n_agents

      obs_n = []
      reward_n = []
      done_n = []
      info_n = {'n': []}
      
      for i, agent in enumerate(self.agents):
        rew = 0
        t=0
        act = action[i]
        d = 0
        
        if act%7 == 0:
          if self.agent_v[i] > 0:
            t = self.traffic[self.agent_o[i]][self.agent_d[i]]
            self.agent_v[i] = self.m_speed * (1 - (t/self.capacity))
            self.agent_pr[i] += self.agent_v[i]/self.dis[self.agent_o[i]][self.agent_d[i]]
            d = self.agent_v[i]/self.dis[self.agent_o[i]][self.agent_d[i]]
            if self.agent_pr[i] >=1:
              self.agent_v[i] = 0
              self.agent_o[i] = self.agent_d[i]
              t = 0
              self.agent_pr[i] = 1
              rew = -1
              if act//7 == 1:
                if self.agent_oo[i] == self.agent_dd[i] == self.agent_p[i] == 0 and ((self.agent_d[i] == 1 and self.supplier_o[1][1]>0) or (self.agent_d[i] == 2 and self.supplier_o[2][1]>0)):
                  self.agent_oo[i] = self.agent_d[i]
                  self.agent_dd[i] = self.supplier_o[self.agent_d[i]][0] 
                  self.agent_p[i] = self.supplier_o[self.agent_d[i]][1]
                  self.supplier_o[self.agent_d[i]][0] = 0
                  self.supplier_o[self.agent_d[i]][1] = 0
                  self.order_r += 1
                  rew += 5
                else:
                  rew = -1
              elif act//7 == 0:
                if self.agent_oo[i] == self.agent_dd[i] == self.agent_p[i] > 0 and self.agent_d[i] == self.agent_dd[i]:
                  self.agent_oo[i] = 0
                  self.agent_dd[i] = 0
                  self.order_d += 1
                  rew += self.agent_p[i]
                  self.agent_p[i] = 0
                else:
                  rew += -1
            else:
              rew += -1

        elif act%7 == 1:
          if self.agent_o[i] == self.agent_d[i] == 2 or self.agent_o[i] == self.agent_d[i] == 3:
            self.agent_d[i] = 1
            t = self.traffic[self.agent_o[i]][self.agent_d[i]]
            self.agent_v[i] = self.m_speed * (1 - (t/self.capacity))
            self.agent_pr[i] = self.agent_v[i]/self.dis[self.agent_o[i]][self.agent_d[i]]
            d = self.agent_v[i]/self.dis[self.agent_o[i]][self.agent_d[i]]
            rew += -1
          elif self.agent_o[i] != self.agent_d[i]:
            t = self.traffic[self.agent_o[i]][self.agent_d[i]]
            self.agent_v[i] = self.m_speed * (1 - (t/self.capacity))
            self.agent_pr[i] += self.agent_v[i]/self.dis[self.agent_o[i]][self.agent_d[i]]
            d = self.agent_v[i]/self.dis[self.agent_o[i]][self.agent_d[i]]
            if self.agent_pr[i] >=1:
              self.agent_v[i] = 0
              self.agent_o[i] = self.agent_d[i]
              t = 0
              self.agent_pr[i] = 1
              rew = -1
              if act//7 == 1:
                if self.agent_oo[i] == self.agent_dd[i] == self.agent_p[i] == 0 and ((self.agent_d[i] == 1 and self.supplier_o[1][1]>0) or (self.agent_d[i] == 2 and self.supplier_o[2][1]>0)):
                  self.agent_oo[i] = self.agent_d[i]
                  self.agent_dd[i] = self.supplier_o[self.agent_d[i]][0] 
                  self.agent_p[i] = self.supplier_o[self.agent_d[i]][1]
                  self.supplier_o[self.agent_d[i]][0] = 0
                  self.supplier_o[self.agent_d[i]][1] = 0
                  self.order_r += 1
                  rew += 5
                else:
                  rew += -20
              elif act//7 == 0:
                if self.agent_oo[i] == self.agent_dd[i] == self.agent_p[i] > 0 and self.agent_d[i] == self.agent_dd[i]:
                  self.agent_oo[i] = 0
                  self.agent_dd[i] = 0
                  self.order_d += 1
                  rew += self.agent_p[i]
                  self.agent_p[i] = 0
                else:
                  rew += -20
            else:
                rew += -20
        elif act%7 == 2:
          if self.agent_o[i] == self.agent_d[i] == 1 or self.agent_o[i] == self.agent_d[i] == 3:
            self.agent_d[i] = 2
            t = self.traffic[self.agent_o[i]][self.agent_d[i]]
            self.agent_v[i] = self.m_speed * (1 - (t/self.capacity))
            self.agent_pr[i] = self.agent_v[i]/self.dis[self.agent_o[i]][self.agent_d[i]]
            d = self.agent_v[i]/self.dis[self.agent_o[i]][self.agent_d[i]]
            rew += -1
          elif self.agent_o[i] != self.agent_d[i]:
            t = self.traffic[self.agent_o[i]][self.agent_d[i]]
            self.agent_v[i] = self.m_speed * (1 - (t/self.capacity)) 
            self.agent_pr[i] += self.agent_v[i]/self.dis[self.agent_o[i]][self.agent_d[i]]
            d = self.agent_v[i]/self.dis[self.agent_o[i]][self.agent_d[i]]
            if self.agent_pr[i] >=1:
              self.agent_v[i] = 0
              self.agent_o[i] = self.agent_d[i]
              t = 0
              self.agent_pr[i] = 1
              rew = -1
              if act//7 == 1:
                if self.agent_oo[i] == self.agent_dd[i] == self.agent_p[i] == 0 and ((self.agent_d[i] == 1 and self.supplier_o[1][1]>0) or (self.agent_d[i] == 2 and self.supplier_o[2][1]>0)):
                  self.agent_oo[i] = self.agent_d[i]
                  self.agent_dd[i] = self.supplier_o[self.agent_d[i]][0] 
                  self.agent_p[i] = self.supplier_o[self.agent_d[i]][1]
                  self.supplier_o[self.agent_d[i]][0] = 0
                  self.supplier_o[self.agent_d[i]][1] = 0
                  self.order_r += 1
                  rew += 5
                else:
                  rew += -20
              elif act//7 == 0:
                if self.agent_oo[i] == self.agent_dd[i] == self.agent_p[i] > 0 and self.agent_d[i] == self.agent_dd[i]:
                  self.agent_oo[i] = 0
                  self.agent_dd[i] = 0
                  self.order_d += 1
                  rew += self.agent_p[i]
                  self.agent_p[i] = 0
                else:
                  rew += -20
            else:
                rew += -20
        
        elif act%7 == 3:
          if self.agent_o[i] == self.agent_d[i] == 1 or self.agent_o[i] == self.agent_d[i] == 2 or self.agent_o[i] == self.agent_d[i] == 4 or self.agent_o[i] == self.agent_d[i] == 5 or self.agent_o[i] == self.agent_d[i] == 6:
            self.agent_d[i] = 3
            t = self.traffic[self.agent_o[i]][self.agent_d[i]]
            self.agent_v[i] = self.m_speed * (1 - (t/self.capacity))
            self.agent_pr[i] = self.agent_v[i]/self.dis[self.agent_o[i]][self.agent_d[i]]
            d = self.agent_v[i]/self.dis[self.agent_o[i]][self.agent_d[i]]
            rew += -1
          elif self.agent_o[i] != self.agent_d[i]:
            t = self.traffic[self.agent_o[i]][self.agent_d[i]]
            self.agent_v[i] = self.m_speed * (1 - (t/self.capacity)) 
            self.agent_pr[i] += self.agent_v[i]/self.dis[self.agent_o[i]][self.agent_d[i]]
            d = self.agent_v[i]/self.dis[self.agent_o[i]][self.agent_d[i]]
            if self.agent_pr[i] >=1:
              self.agent_v[i] = 0
              self.agent_o[i] = self.agent_d[i]
              t = 0
              self.agent_pr[i] = 1
              rew = -1
              if act//7 == 1:
                if self.agent_oo[i] == self.agent_dd[i] == self.agent_p[i] == 0 and ((self.agent_d[i] == 1 and self.supplier_o[1][1]>0) or (self.agent_d[i] == 2 and self.supplier_o[2][1]>0)):
                  self.agent_oo[i] = self.agent_d[i]
                  self.agent_dd[i] = self.supplier_o[self.agent_d[i]][0] 
                  self.agent_p[i] = self.supplier_o[self.agent_d[i]][1]
                  self.supplier_o[self.agent_d[i]][0] = 0
                  self.supplier_o[self.agent_d[i]][1] = 0
                  self.order_r += 1
                  rew += 5
                else:
                  rew += -20
              elif act//7 == 0:
                if self.agent_oo[i] == self.agent_dd[i] == self.agent_p[i] > 0 and self.agent_d[i] == self.agent_dd[i]:
                  self.agent_oo[i] = 0
                  self.agent_dd[i] = 0
                  self.order_d += 1
                  rew += self.agent_p[i]
                  self.agent_p[i] = 0
                else:
                  rew += -20
            else:
                rew += -20

        elif act%7 == 4:
          if self.agent_o[i] == self.agent_d[i] == 5 or self.agent_o[i] == self.agent_d[i] == 3:
            self.agent_d[i] = 4
            t = self.traffic[self.agent_o[i]][self.agent_d[i]]
            self.agent_v[i] = self.m_speed * (1 - (t/self.capacity))
            self.agent_pr[i] = self.agent_v[i]/self.dis[self.agent_o[i]][self.agent_d[i]]
            d = self.agent_v[i]/self.dis[self.agent_o[i]][self.agent_d[i]]
            rew += -1
          elif self.agent_o[i] != self.agent_d[i]:
            t = self.traffic[self.agent_o[i]][self.agent_d[i]]
            self.agent_v[i] = self.m_speed * (1 - (t/self.capacity))
            self.agent_pr[i] += self.agent_v[i]/self.dis[self.agent_o[i]][self.agent_d[i]]
            d = self.agent_v[i]/self.dis[self.agent_o[i]][self.agent_d[i]]
            if self.agent_pr[i] >=1:
              self.agent_v[i] = 0
              self.agent_o[i] = self.agent_d[i]
              t = 0
              self.agent_pr[i] = 1
              rew = -1
              if act//7 == 1:
                if self.agent_oo[i] == self.agent_dd[i] == self.agent_p[i] == 0 and ((self.agent_d[i] == 1 and self.supplier_o[1][1]>0) or (self.agent_d[i] == 2 and self.supplier_o[2][1]>0)):
                  self.agent_oo[i] = self.agent_d[i]
                  self.agent_dd[i] = self.supplier_o[self.agent_d[i]][0] 
                  self.agent_p[i] = self.supplier_o[self.agent_d[i]][1]
                  self.supplier_o[self.agent_d[i]][0] = 0
                  self.supplier_o[self.agent_d[i]][1] = 0
                  self.order_r += 1
                  rew += 5
                else:
                  rew += -20
              elif act//7 == 0:
                if self.agent_oo[i] == self.agent_dd[i] == self.agent_p[i] > 0 and self.agent_d[i] == self.agent_dd[i]:
                  self.agent_oo[i] = 0
                  self.agent_dd[i] = 0
                  self.order_d += 1
                  rew += self.agent_p[i]
                  self.agent_p[i] = 0
                else:
                  rew += -20
            else:
                rew += -20
        
        elif act%7 == 5:
          if self.agent_o[i] == self.agent_d[i] == 4 or self.agent_o[i] == self.agent_d[i] == 3 or self.agent_o[i] == self.agent_d[i] == 6:
            self.agent_d[i] = 5
            t = self.traffic[self.agent_o[i]][self.agent_d[i]]
            self.agent_v[i] = self.m_speed * (1 - (t/self.capacity))
            self.agent_pr[i] = self.agent_v[i]/self.dis[self.agent_o[i]][self.agent_d[i]]
            d = self.agent_v[i]/self.dis[self.agent_o[i]][self.agent_d[i]]
            rew += -1
          elif self.agent_o[i] != self.agent_d[i]:
            t = self.traffic[self.agent_o[i]][self.agent_d[i]]
            self.agent_v[i] = self.m_speed * (1 - (t/self.capacity))
            self.agent_pr[i] += self.agent_v[i]/self.dis[self.agent_o[i]][self.agent_d[i]]
            d = self.agent_v[i]/self.dis[self.agent_o[i]][self.agent_d[i]]
            if self.agent_pr[i] >=1:
              self.agent_v[i] = 0
              self.agent_o[i] = self.agent_d[i]
              t = 0
              self.agent_pr[i] = 1
              rew = -1
              if act//7 == 1:
                if self.agent_oo[i] == self.agent_dd[i] == self.agent_p[i] == 0 and ((self.agent_d[i] == 1 and self.supplier_o[1][1]>0) or (self.agent_d[i] == 2 and self.supplier_o[2][1]>0)):
                  self.agent_oo[i] = self.agent_d[i]
                  self.agent_dd[i] = self.supplier_o[self.agent_d[i]][0] 
                  self.agent_p[i] = self.supplier_o[self.agent_d[i]][1]
                  self.supplier_o[self.agent_d[i]][0] = 0
                  self.supplier_o[self.agent_d[i]][1] = 0
                  self.order_r += 1
                  rew += 5
                else:
                  rew += -20
              elif act//7 == 0:
                if self.agent_oo[i] == self.agent_dd[i] == self.agent_p[i] > 0 and self.agent_d[i] == self.agent_dd[i]:
                  self.agent_oo[i] = 0
                  self.agent_dd[i] = 0
                  self.order_d += 1
                  rew += self.agent_p[i]
                  self.agent_p[i] = 0
                else:
                  rew += -20
            else:
                rew += -20
        
        elif act%7 == 6:
          if self.agent_o[i] == self.agent_d[i] == 5 or self.agent_o[i] == self.agent_d[i] == 3:
            self.agent_d[i] = 6
            t = self.traffic[self.agent_o[i]][self.agent_d[i]]
            self.agent_v[i] = self.m_speed * (1 - (t/self.capacity))
            self.agent_pr[i] = self.agent_v[i]/self.dis[self.agent_o[i]][self.agent_d[i]]
            d = self.agent_v[i]/self.dis[self.agent_o[i]][self.agent_d[i]]
            rew += -1
          elif self.agent_o[i] != self.agent_d[i]:
            t = self.traffic[self.agent_o[i]][self.agent_d[i]]
            self.agent_v[i] = self.m_speed * (1 - (t/self.capacity))
            self.agent_pr[i] += self.agent_v[i]/self.dis[self.agent_o[i]][self.agent_d[i]]
            d = self.agent_v[i]/self.dis[self.agent_o[i]][self.agent_d[i]]
            if self.agent_pr[i] >=1:
              self.agent_v[i] = 0
              self.agent_o[i] = self.agent_d[i]
              t = 0
              self.agent_pr[i] = 1
              rew = -1
              if act//7 == 1:
                if self.agent_oo[i] == self.agent_dd[i] == self.agent_p[i] == 0 and ((self.agent_d[i] == 1 and self.supplier_o[1][1]>0) or (self.agent_d[i] == 2 and self.supplier_o[2][1]>0)):
                  self.agent_oo[i] = self.agent_d[i]
                  self.agent_dd[i] = self.supplier_o[self.agent_d[i]][0] 
                  self.agent_p[i] = self.supplier_o[self.agent_d[i]][1]
                  self.supplier_o[self.agent_d[i]][0] = 0
                  self.supplier_o[self.agent_d[i]][1] = 0
                  self.order_r += 1
                  rew += 5
                else:
                  rew += -20
              elif act//7 == 0:
                if self.agent_oo[i] == self.agent_dd[i] == self.agent_p[i] > 0 and self.agent_d[i] == self.agent_dd[i]:
                  self.agent_oo[i] = 0
                  self.agent_dd[i] = 0
                  self.order_d += 1
                  rew += self.agent_p[i]
                  self.agent_p[i] = 0
                else:
                  rew += -20
            else:
                rew += -20
        
        ob = np.hstack([
            self.supplier_o[1][0],
            self.supplier_o[1][1],
            self.supplier_o[2][0],
            self.supplier_o[2][1],
            self.agent_v[i],
            self.agent_o[i],
            self.agent_d[i],
            self.agent_pr[i],
            t,
            self.agent_oo[i],
            self.agent_p[i],
            self.agent_dd[i]
        ])

        do = False

        if self.agent_o[i] == self.agent_d[i]:
          self.xy[i+1][0] += self.x[self.agent_o[i]] - self.xy[i+1][0]
          self.xy[i+1][1] += self.y[self.agent_o[i]] - self.xy[i+1][1]
        else:
          
          self.xy[i+1][0] += -(self.xy[i+1][0]-self.x[int(self.agent_d[i])])*d
          self.xy[i+1][1] += -(self.xy[i+1][1]-self.y[int(self.agent_d[i])])*d

        obs_n.append(ob)
        reward_n.append(rew)
        done_n.append(do)
      
      for i in range(self.n_supplier):
        if self.supplier_o[i+1][1] == 0:
          self.supplier_o[i+1][0] = self.customer_or[i+1][self.time][0]
          self.supplier_o[i+1][1] = self.customer_or[i+1][self.time][1]
      
      self.time += 1

      if self.time >= self.total_time:
        done_n = [True] * self.n_agents

      self.fps_clock.tick(self.fps)
      
      
      #pygame.display.update()
      
      #obs_n = dict(zip(self.agents, obs_n))
      #reward_n = dict(zip(self.agents, reward_n))
      #done_n = dict(zip(self.agents, done_n))
      return obs_n, reward_n, done_n, info_n
    
    def render(self, mode: str = 'human'):

        pygame.init()
        pygame.display.init()

        self.screen.fill((200, 200, 200))
        #pygame.draw.polygon(self.screen, (0, 0, 0), ((100, 0), (200, 0), (200, 200), (300, 200), (150, 300), (0, 200), (100, 200)))
        #pygame.draw.polygon(self.screen, (0, 0, 0), ((40,32), (56,32), (40, 200), (56, 200), (40, 400), (56, 400), (48,468)))
        def draw_arrow(screen, colour, start, end):
            pygame.draw.line(screen,colour,start,end,15)
            rotation = math.degrees(math.atan2(start[1]-end[1], end[0]-start[0]))+90
            pygame.draw.polygon(screen, (0, 0, 0), ((end[0]+20*math.sin(math.radians(rotation)), end[1]+20*math.cos(math.radians(rotation))), (end[0]+20*math.sin(math.radians(rotation-120)), end[1]+20*math.cos(math.radians(rotation-120))), (end[0]+20*math.sin(math.radians(rotation+120)), end[1]+20*math.cos(math.radians(rotation+120)))))

        draw_arrow(self.screen, (0, 0, 0),[48,48],[48,400])
        draw_arrow(self.screen, (0, 0, 0),[16,468],[16,80])

        draw_arrow(self.screen, (0, 0, 0),[48,32],[216,200])
        draw_arrow(self.screen, (0, 0, 0),[212,228],[80,94])

        draw_arrow(self.screen, (0, 0, 0),[48,468],[218,260])
        draw_arrow(self.screen, (0, 0, 0),[212,228],[64,420])

        draw_arrow(self.screen, (0, 0, 0),[244,228],[452,64])
        draw_arrow(self.screen, (0, 0, 0),[452,32],[244,196])

        draw_arrow(self.screen, (0, 0, 0),[244,228],[452,436])
        draw_arrow(self.screen, (0, 0, 0),[452,468],[244,260])

        draw_arrow(self.screen, (0, 0, 0),[244,244],[420,244])
        draw_arrow(self.screen, (0, 0, 0),[468,212],[280,212])

        draw_arrow(self.screen, (0, 0, 0),[484,228],[484,80])
        draw_arrow(self.screen, (0, 0, 0),[452,80],[452,180])

        draw_arrow(self.screen, (0, 0, 0),[452,228],[452,400])
        draw_arrow(self.screen, (0, 0, 0),[484,468],[484,280])

        
        pygame.draw.line(self.screen,(self.traffic[1][2]*35,255 - self.traffic[1][2]*35,0),(48,32),(48,468),5)
        pygame.draw.line(self.screen,(self.traffic[2][1]*35,255 - self.traffic[2][1]*35,0),(16,32),(16,468),5)
        
        pygame.draw.line(self.screen,(self.traffic[1][3]*35,255 - self.traffic[1][3]*35,0),(48,32),(244,228),5)
        pygame.draw.line(self.screen,(self.traffic[3][1]*35,255 - self.traffic[3][1]*35,0),(16,32),(212,228),5)
        
        pygame.draw.line(self.screen,(self.traffic[2][3]*35,255 - self.traffic[2][3]*35,0),(48,468),(244,228),5)
        pygame.draw.line(self.screen,(self.traffic[3][2]*35,255 - self.traffic[3][2]*35,0),(16,468),(212,228),5)
        
        pygame.draw.line(self.screen,(self.traffic[3][4]*35,255 - self.traffic[3][4]*35,0),(244,228),(484,32),5)
        pygame.draw.line(self.screen,(self.traffic[4][3]*35,255 - self.traffic[4][3]*35,0),(212,228),(452,32),5)
        
        pygame.draw.line(self.screen,(self.traffic[3][6]*35,255 - self.traffic[3][6]*35,0),(244,228),(484,468),5)
        pygame.draw.line(self.screen,(self.traffic[3][6]*35,255 - self.traffic[3][6]*35,0),(212,228),(452,468),5)
        
        pygame.draw.line(self.screen,(self.traffic[3][5]*35,255 - self.traffic[3][5]*35,0),(244,244),(468,244),5)
        pygame.draw.line(self.screen,(self.traffic[5][3]*35,255 - self.traffic[5][3]*35,0),(235,212),(468,212),5)
        
        pygame.draw.line(self.screen,(self.traffic[5][4]*35,255 - self.traffic[5][4]*35,0),(484,228),(484,32),5)
        pygame.draw.line(self.screen,(self.traffic[4][5]*35,255 - self.traffic[4][5]*35,0),(452,228),(452,32),5)
        
        pygame.draw.line(self.screen,(self.traffic[5][6]*35,255 - self.traffic[5][6]*35,0),(452,228),(452,468),5)
        pygame.draw.line(self.screen,(self.traffic[6][5]*35,255 - self.traffic[6][5]*35,0),(484,228),(484,468),5)
        
        
        self.screen.blit(self.img1, (0,0))
        self.screen.blit(self.img2, (0,436))
        self.screen.blit(self.img3, (200,200))
        self.screen.blit(self.img4, (436,0))
        self.screen.blit(self.img5, (436,186))
        self.screen.blit(self.img6, (436,436))
        self.screen.blit(self.img1, (0,0))

        for i in range(self.n_agents):
            self.screen.blit(self.ag[i], (self.xy[i+1][0],self.xy[i+1][1]))

        pygame.event.pump()
        pygame.display.update()
        #pygame.image.save(self.screen, "screenshot"+str(self.time)+".jpg")
        
        self.traffic = np.array([[0,0,0,0,0,0,0],
                             [0,0,np.random.randint(1,3),np.random.randint(2,7),0,0,0],
                             [0,np.random.randint(1,3),0,np.random.randint(2,5),0,0,0],
                             [0,np.random.randint(2,7),np.random.randint(2,5),0,np.random.randint(3,7),np.random.randint(4,5),np.random.randint(1,3)],
                             [0,0,0,np.random.randint(3,7),0,np.random.randint(1,3),0],
                             [0,0,0,np.random.randint(4,5),np.random.randint(1,3),0,np.random.randint(1,2)],
                             [0,0,0,np.random.randint(1,3),0,np.random.randint(1,2),0],])
    
    def reset(self):
      
      self.traffic = np.array([[0,0,0,0,0,0,0],
                             [0,0,np.random.randint(1,3),np.random.randint(2,7),0,0,0],
                             [0,np.random.randint(1,3),0,np.random.randint(2,5),0,0,0],
                             [0,np.random.randint(2,7),np.random.randint(2,5),0,np.random.randint(3,7),np.random.randint(4,5),np.random.randint(1,3)],
                             [0,0,0,np.random.randint(3,7),0,np.random.randint(1,3),0],
                             [0,0,0,np.random.randint(4,5),np.random.randint(1,3),0,np.random.randint(1,2)],
                             [0,0,0,np.random.randint(1,3),0,np.random.randint(1,2),0],])
      
      ti = 0
      ord = np.random.poisson(10, size=(self.n_supplier,))
      ti += max(ord)

      while ti < self.total_time:
        ord = np.random.poisson(10, size=(self.n_supplier,))
        for i in range(self.n_supplier):
          self.customer_or[i+1][ti][0] = np.random.choice([4,5,6])
          self.customer_or[i+1][ti][1] = np.random.randint(4,10)
        ti += max(ord)

      obs_n = []
      self.agent_o = []
      self.agent_d = []
      self.time = 0

      for i, agent in enumerate(self.agents):
        p = np.random.randint(1,self.n_customer + self.n_hub + self.n_supplier)
        self.agent_o.append(p)
        self.agent_d.append(p)
        ob = np.hstack([
            0,
            0,
            0,
            0,
            self.agent_v[i],
            p,
            p,
            self.agent_pr[i],
            0,
            0,
            0,
            0,
        ])
        self.xy[i+1][0] += self.x[p]
        self.xy[i+1][1] += self.y[p]
        
        obs_n.append(ob)

      #obs_n = dict(zip(self.agents, obs_n))
      return obs_n
    
env = MultiAgentEnv()

#env.action_space[0].n

#done_n = [False for _ in range(env.n_agents)]
#ep_reward = 0

#obs_n = env.reset()
#print(obs_n)
#while not all(done_n):
#    env.render()
#    action = env.action_space.sample()
#    obs_n, reward_n, done_n = env.step(action)
#    ep_reward += sum(reward_n)
#    for i in action:
#        print(i%7)
#    for i in action:
#        print(i//7)
#    #print(i%7 for i in action)
#    #print(i//7 for i in action)
#    print((obs_n))
#    #print((reward_n))
#    #print((done_n))

import collections
import random

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ma_gym.wrappers import Monitor

USE_WANDB = False  # if enabled, logs data on wandb server


class ReplayBuffer:
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask_lst.append((np.ones(len(done)) - done).tolist())

        return torch.tensor(s_lst, dtype=torch.float), \
               torch.tensor(a_lst, dtype=torch.float), \
               torch.tensor(r_lst, dtype=torch.float), \
               torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst, dtype=torch.float)

    def size(self):
        return len(self.buffer)


class QNet(nn.Module):
    def __init__(self, observation_space, action_space):
        super(QNet, self).__init__()
        self.num_agents = len(observation_space)
        for agent_i in range(self.num_agents):
            n_obs = observation_space[agent_i].shape[0]
            setattr(self, 'agent_{}'.format(agent_i), nn.Sequential(nn.Linear(n_obs, 128),
                                                                    nn.ReLU(),
                                                                    nn.Linear(128, 64),
                                                                    nn.ReLU(),
                                                                    nn.Linear(64, action_space[agent_i].n)))

    def forward(self, obs):
        q_values = [torch.empty(obs.shape[0], )] * self.num_agents
        for agent_i in range(self.num_agents):
            q_values[agent_i] = getattr(self, 'agent_{}'.format(agent_i))(obs[:, agent_i, :]).unsqueeze(1)

        return torch.cat(q_values, dim=1)

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        mask = (torch.rand((out.shape[0],)) <= epsilon)
        action = torch.empty((out.shape[0], out.shape[1],))
        action[mask] = torch.randint(0, out.shape[2], action[mask].shape).float()
        action[~mask] = out[~mask].argmax(dim=2).float()
        return action


def train(q, q_target, memory, optimizer, gamma, batch_size, update_iter=10):
    for _ in range(update_iter):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(2, a.unsqueeze(-1).long()).squeeze(-1)
        max_q_prime = q_target(s_prime).max(dim=2)[0]
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target.detach())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(env, num_episodes, q):
    score = np.zeros(env.n_agents)
    for episode_i in range(num_episodes):
        state = env.reset()
        env.render()
        done = [False for _ in range(env.n_agents)]
        while not all(done):
            action = q.sample_action(torch.Tensor(state).unsqueeze(0), epsilon=0)[0].data.cpu().numpy().tolist()
            print("act", action)
            next_state, reward, done, info = env.step(action)
            print("next_state", next_state)
            score += np.array(reward)
            state = next_state

    return sum(score / num_episodes)


def main(env_name, lr, gamma, batch_size, buffer_limit, log_interval, max_episodes,
         max_epsilon, min_epsilon, test_episodes, warm_up_steps, update_iter, monitor=False):
    env = MultiAgentEnv()
    test_env = MultiAgentEnv()
    if monitor:
        test_env = Monitor(test_env, directory='recordings/idqn/{}'.format(env_name),
                           video_callable=lambda episode_id: episode_id % 5 == 0)
    memory = ReplayBuffer(buffer_limit)

    q = QNet(env.observation_space, env.action_space)
    q_target = QNet(env.observation_space, env.action_space)
    q_target.load_state_dict(q.state_dict())
    optimizer = optim.Adam(q.parameters(), lr=lr)

    score = np.zeros(env.n_agents)
    for episode_i in range(max_episodes):
        epsilon = max(min_epsilon, max_epsilon - (max_epsilon - min_epsilon) * (episode_i / (0.4 * max_episodes)))
        state = env.reset()
        env.render()
        done = [False for _ in range(env.n_agents)]
        while not all(done):
            action = q.sample_action(torch.Tensor(state).unsqueeze(0), epsilon)[0].data.cpu().numpy().tolist()
            print("act", action)
            next_state, reward, done, info = env.step(action)
            print("next_state", next_state)
            memory.put((state, action, (np.array(reward)).tolist(), next_state, np.array(done, dtype=int).tolist()))
            score += np.array(reward)
            state = next_state

        if memory.size() > warm_up_steps:
            train(q, q_target, memory, optimizer, gamma, batch_size, update_iter)

        if episode_i % log_interval == 0 and episode_i != 0:
            q_target.load_state_dict(q.state_dict())
            test_score = test(test_env, test_episodes, q)
            print("#{:<10}/{} episodes , avg train score : {:.1f}, test score: {:.1f} n_buffer : {}, eps : {:.1f}"
                  .format(episode_i, max_episodes, sum(score / log_interval), test_score, memory.size(), epsilon))
            if USE_WANDB:
                wandb.log({'episode': episode_i, 'test-score': test_score,
                           'buffer-size': memory.size(), 'epsilon': epsilon, 'train-score': sum(score / log_interval)})
            score = np.zeros(env.n_agents)

    env.close()
    test_env.close()


if __name__ == '__main__':
    kwargs = {'env_name': 'MultiAgentEnv',
              'lr': 0.0005,
              'batch_size': 32,
              'gamma': 0.99,
              'buffer_limit': 50000,
              'log_interval': 20,
              'max_episodes': 1000,
              'max_epsilon': 0.9,
              'min_epsilon': 0.1,
              'test_episodes': 5,
              'warm_up_steps': 2000,
              'update_iter': 10,
              'monitor': False}
    if USE_WANDB:
        import wandb

        wandb.init(project='minimal-marl', config={'algo': 'idqn', **kwargs}, monitor_gym=True)

    main(**kwargs)
