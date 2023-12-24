import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TradingEnv(gym.Env):
    def __init__(self, data, initial_balance=10000):
        super(TradingEnv, self).__init__()

        self.data = data
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.current_step = 0
        self.done = False
        self.position = None

        # Azioni: 0->Compra, 1->Mantieni, 2->Vendi
        self.action_space = spaces.Discrete(3)
        
        # Spazio degli stati: prezzo attuale + saldo corrente
        self.observation_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.done = False
        self.balance = self.initial_balance
        self.position = None
        return self._next_observation()

    def step(self, action):
        # Calcola il profitto (se ci sono operazioni di compravendita)
        if action == 0:  # Compra
            self.position = self.data['Prezzo'].iloc[self.current_step]
            self.balance -= self.position
        elif action == 2 and self.position is not None:  # Vendi
            profit = self.data['Prezzo'].iloc[self.current_step] - self.position
            self.balance += profit + self.position
            self.position = None

        # Passa al prossimo step
        self.current_step += 1

        # Controllo se siamo alla fine dei dati
        if self.current_step == len(self.data) - 1:
            self.done = True

        return self._next_observation(), self.balance, self.done, {self.current_step}

    def _next_observation(self):
        obs = np.array([self.data['Prezzo'].iloc[self.current_step], self.balance])
        return obs

    def render(self, mode='human', close=False):
        # (Opzionale) Puoi implementare una funzione di rendering per visualizzare l'andamento del trading
        pass

    def close(self):
        pass
