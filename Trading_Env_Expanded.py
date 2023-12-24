from ctypes import Array
import gym as gym
from gym import spaces
import numpy as np
import copy

class ExpandedEnv(gym.Env):
    def __init__(self, data, windows_size, fees, initial_balance = 100000):
        super(ExpandedEnv, self).__init__()

        self.last_qty_both = 0
        self.fees = fees
        self.initial_balance = initial_balance
        self.current_step = 0
        self.done = False
        self.position = 0 # OneHot => 0:wait 1:long 2:short
        self.window_size = windows_size
        self.window = []
        self.current_balance = self.initial_balance
        self.data = data
        self.position_status = np.array([1,0,0], dtype=np.int8)
        self.last_action = np.array([1,0,0] , dtype=np.int8)
        self.last_Reward = 0
        self.current_price = None

        # Definizione dello spazio delle osservazioni e delle azioni
            # Azioni: 0->Compra, 1->Mantieni, 2->Vendi One hot Encoding
            # nota: non contempla le ristrizioni date dalla condizione di una sola posizione simultanea, apprendimento rallentato 
            # da una condizione implementataq solo nel metodo step
        self.action_space = spaces.Box(low=np.array([0,0,0]) , high=np.array([1,1,1]), dtype=np.int8)

            # Spazio degli stati: stato attuale + saldo corrente 
        self.observation_space = spaces.Box(low=np.array([0,0,0,0]), high=np.array([1,1,1,100*initial_balance]), dtype=np.float32)

    def set_statitng_Wind(self):
        wind = [{'posizione': np.array([1, 0, 0]),
                    'prezzo': self.data[i], 
                    'step': -self.window_size + i + 1} for i in range(self.window_size)]
        return wind

    def reset(self):
        # Reimposta l'ambiente allo stato iniziale
        self.current_step = 0
        self.done = False
        self.window = self.set_statitng_Wind()
        # TODO : debugging, passo la finestra anziche la singola osservazione
        observation = self.window #np.append(self.position_status, self.current_price)

        # TODO : fix current return step (verificare anche se partire da -20 o da -19)
        return copy.deepcopy(observation) , copy.deepcopy(self.last_Reward) , copy.deepcopy(self.done), {copy.deepcopy(self.current_step)}

    def step(self, action):
        
        # HACK: porcata di conversione
        try:
            if isinstance(copy.copy(action), np.int64):
                if action == 0:
                    action = np.array([1, 0, 0])

                if action == 1:
                    action = np.array([0, 1, 0])

                if action == 2:
                    action = np.array([0, 0, 1])
        except :
            pass


        #Salvo l'ultima azione
        self.last_action = action

        #Salvo una copia di current step
        passo = copy.copy(self.current_step)

        # Verifico se è l ultima barra
        if passo == len(self.data)-1 - self.window_size:
            self.done = True

        # Chiudo eventuali poosizioni prima di ottenere l'ultima osservazione
        if self.done == True:

            if self.getOneHodtDecoder_Psition(self.position_status ) == 'Long':
                self.last_action = action = np.array([0,0,1])

                
            if self.getOneHodtDecoder_Psition(self.position_status ) == 'Short':
                self.last_action = action = np.array([0,1,0])

        self._buy_and_sell()
        
        # Eseguo il primo passo o tutti i successivi
        self.current_step += 1

        self.set_Window()

        observation = copy.deepcopy(self.window) #np.append(self.position_status, self.current_price)

        return observation, copy.deepcopy(self.last_Reward), copy.deepcopy(self.done), {f'prezzo della barra {self.current_price} ; ultima quantita scambiata {self.last_qty_both } ; step {self.current_step}' }

    def set_Window(self):
         new_bar = {
            'posizione':self.position_status,
            'prezzo': self.data[self.current_step + self.window_size - 1],
            'step': self.current_step }
    
         self.window.pop(0)  # Rimuove la prima barra
         self.window.append(new_bar)

    # posso tipizzare posizione per verificare la compatibilata con questo one hot encoding np.array([0,0,0])??
    def getOneHodtDecoder_Psition(self, posizione):
        if (posizione == np.array([1, 0, 0])).all():
            return 'Waiting'
        elif (posizione == np.array([0, 1, 0])).all():
            return 'Long'
        elif (posizione == np.array([0, 0, 1])).all():
            return 'Short'
        else:
            return 'Error'

    def getOneHodtDecoder_Action(self, azione):

        if (azione == np.array([1, 0, 0])).all():
            return 'Wait'
        elif (azione == np.array([0, 1, 0])).all():
            return 'Buy'
        elif (azione == np.array([0, 0, 1])).all():
            return 'Sell'
        else:
            return 'Error'

    # Aggiorno lo stato e il balance 
    # TODO: sembra non aggiornarsi ne la posizione ne il bilancio
    def _buy_and_sell(self):
        price = self.data[self.current_step + self.window_size]
        self.current_price = price

        if self.getOneHodtDecoder_Action(self.last_action) == 'Buy':

            if self.getOneHodtDecoder_Psition(self.position_status) =='Waiting':
                fees = self.calculate_Fees()
                self.current_balance -= fees
                self.last_qty_both = self.current_balance / price
                self.last_Reward = -self.calculate_Fees()
                self.position_status = np.array([0, 1, 0], dtype=np.int8)
                return

            elif self.getOneHodtDecoder_Psition(self.position_status) == 'Short':
               gain = self.last_qty_both * price
               self.last_Reward = gain - self.calculate_Fees() - self.current_balance
               self.current_balance = gain - self.calculate_Fees()
               self.position_status = np.array([1, 0, 0], dtype=np.int8)
               self.last_qty_both = 0
               return
                
            return

        elif self.getOneHodtDecoder_Action(self.last_action) == 'Sell':

            if self.getOneHodtDecoder_Psition(self.position_status) =='Waiting':
                self.current_balance -= self.calculate_Fees()
                self.last_Reward = - self.calculate_Fees()
                self.last_qty_both = self.current_balance / price
                self.position_status = np.array([0, 0, 1], dtype=np.int8)
                return

            elif self.getOneHodtDecoder_Psition(self.position_status) == 'Long':
                gain = self.last_qty_both * price
                self.last_Reward = gain - self.calculate_Fees() - self.current_balance
                self.current_balance = gain - self.calculate_Fees()
                self.position_status = np.array([1, 0, 0], dtype=np.int8)
                self.last_qty_both = 0
                return

            return

        # HINT : Inserisco le ricompense per stimolarlo ad agire
        else:
            if self.getOneHodtDecoder_Psition(self.position_status) =='Waiting':
                self.last_Reward = -1
            else :
                self.last_Reward = 0

    # Calcolo le fee 
    def calculate_Fees(self):
        return self.current_balance * self.fees

    # Eventuali altri metodi ausiliari
        # Stop
        # Render 
    