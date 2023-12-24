from abc import abstractstaticmethod
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from Trading_Env_Expanded import ExpandedEnv as env

class GeneratoreDatasetTradingClass:
    def __init__(self, prezzo_iniziale=100, num_giorni=252, seed=42):
        self.prezzo_iniziale = prezzo_iniziale
        self.num_giorni = num_giorni
        self.seed = seed

    def genera_dataset(self):
        np.random.seed(self.seed)  # Imposta il seed per la riproducibilità

        # Genera variazioni di prezzo giornaliere
        variazioni = np.random.normal(0, 1, self.num_giorni)  # Variazione giornaliera

        # Calcola i prezzi cumulativi
        prezzi = self.prezzo_iniziale + np.cumsum(variazioni)

        # Converti in DataFrame di Pandas
        return pd.DataFrame(prezzi, columns=['Prezzo'])

    @abstractstaticmethod
    def Genera_Prezzi_Complessi(anni=3):
        giorni = int(anni*365)
        start_date = datetime.now() - timedelta(days=giorni)
        end_date = datetime.now()
        
        # Creare un range di date
        date_range = pd.date_range(start=start_date, end=end_date)
        
        # Generare dati fittizi per i prezzi di apertura, chiusura, massimo e minimo
        np.random.seed(0)  # Per la riproducibilità
        open_prices = np.random.uniform(low=10000, high=60000, size=len(date_range))
        close_prices = open_prices + np.random.uniform(low=-1000, high=1000, size=len(date_range))
        high_prices = np.maximum(open_prices, close_prices) + np.random.uniform(low=0, high=2000, size=len(date_range))
        low_prices = np.minimum(open_prices, close_prices) - np.random.uniform(low=0, high=2000, size=len(date_range))
        volumes = np.random.uniform(low=1000, high=10000, size=len(date_range))
        
        # Creare il DataFrame
        bitcoin_prices_df = pd.DataFrame({
            'Date': date_range,
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Prezzo': close_prices,
            'Volume': volumes
        })

        return bitcoin_prices_df

    def Sample_Group(df, group_size):
        num_groups = len(df) // group_size
        return [df[i*group_size:(i+1)*group_size] for i in range(num_groups)]

    def extract_colums(df,columns):
        return df[columns]


#x = GeneratoreDatasetTradingClass.Genera_Prezzi_Complessi()

#dati = x['Prezzo']

#envoirment = env(dati,20,0.01)

#resoult = envoirment.reset()
#print(resoult)

#resoult = envoirment.step(np.array([0,1,0], dtype=np.int8))
#print(resoult)
#resoult = envoirment.step(np.array([0,0,1], dtype=np.int8))
#print(resoult)
#resoult = envoirment.step(np.array([0,1,0], dtype=np.int8))
#print(resoult)
#resoult = envoirment.step(np.array([0,0,1], dtype=np.int8))


#print(resoult)
#print(envoirment.window_size)
#print(envoirment.data[10:envoirment.window_size+10])
#print(envoirment.window)



