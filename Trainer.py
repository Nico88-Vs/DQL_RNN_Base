# -*- coding: utf-8 -*-

# ALL Import
import copy
from os import stat_result
from CustomDQNModel import CustomDQNModel
from ReplayBuffer import ReplayBuffer
from GeneratoreDatasetTradingEvoluto import GeneratoreDatasetTradingClass as gen
from CustomDQNModel import CustomDQNModel as nn
import tensorflow as tf
import numpy as np
from Trading_Env_Expanded import ExpandedEnv as amb

# TODO: Controllare le variabili
#General variables
n_timesteps = 1000
windows_size = 20
n_episodi = 6
n_variabili = 2
n_action = 3
batch_size = 32
epsilon_start = 0.3
epsilon_end = 0.1
epsilon_decay_steps = 10000


#Load Data
dati = gen.Genera_Prezzi_Complessi(0.2)
set_dati = dati['Prezzo']  # non passo set_dati per mantenere le intestazioni e potere utilizzare l'indice
#campionato = gen.Sample_Group(dati,n_campioni)
print(dati)

#load env
env_evoluto = amb(set_dati, windows_size, 0.01)

#Load NN
main_network = nn(batch_size, windows_size, n_action) #nn(windows_size, 2, n_action) modifico per testare il primo passo

# Compile!! Sostituisci 'optimizer' con l'ottimizzatore che desideri utilizzare (ad esempio, 'adam')
# e 'loss' con la funzione di perdita appropriata per il tuo problema
main_network.compile(optimizer='adam', loss='mean_squared_error')

#Build Target NN
target_network = tf.keras.models.clone_model(main_network)
target_network.set_weights(main_network.get_weights())

#Creazione Buffer
replay_buffer = ReplayBuffer(1000)


#Policy Definition
    #Epsylon Reduction policy
def epsylon_greedy_policy(state, epsilon, model):
    if np.random.rand() < epsilon:
        x = np.random.randint(n_action)
        azione_one_hot = np.zeros(n_action)
        azione_one_hot[x] = 1
        return azione_one_hot
    else:
        Q_values = model.predict(state)
        x = np.argmax(Q_values[0])
        azione_one_hot = np.zeros(n_action)
        azione_one_hot[x] = 1
        # HACK : Sembra che predict funzioni
        return azione_one_hot

    #Epsilon Reducement


# Funzione d'aggiornamento della rete Target 
    #   In questo esempio, tau � un fattore di interpolazione che controlla quanto rapidamente i pesi della target network 
    #   si adattano ai pesi della rete principale. Un valore di tau vicino a 1 farebbe s� che la target network si aggiorni 
    #   quasi completamente ai pesi della rete principale, mentre un valore pi� piccolo fa s� che l'aggiornamento sia pi� graduale.
def update_target_network(main_network, target_network, tau=0.1):
    main_weights = main_network.get_weights()
    target_weights = target_network.get_weights()

    for i in range(len(target_weights)):
        target_weights[i] = tau * main_weights[i] + (1 - tau) * target_weights[i]

    target_network.set_weights(target_weights)

#verificare la funzione di riduzione di epsilon
def reduce_epsilon(epsilon, step):
    decay_rate = (epsilon_start - epsilon_end) / epsilon_decay_steps
    return max(epsilon - decay_rate, epsilon_end)

# Campionamento del Batch
def campionamento(replay_buffer_:ReplayBuffer, battch_size):
    batch = replay_buffer_.sample(battch_size)
    stati, azioni, ricompense, stati_successivi, terminati = zip(*batch)
    return stati, azioni, ricompense, stati_successivi, terminati


def Estrai_Stati(tulpa_di_stati_successivi, ricompense, terminati):
    # estraggo tutti i tensori dall indice zero di ogni elemento della lista
    tensori = [t[0] for t in tulpa_di_stati_successivi]

    # Li combino in un unico tensore
    tensore = tf.stack(tensori)
    ricompense = tf.stack(ricompense)
    terminati = tf.stack(terminati)

    return tensore , ricompense, terminati

# Aggiornamento Rete Principale
#La funzione di aggiornamento della rete principale utilizza il batch campionato per aggiornare la rete in base all'errore tra le Q-value stimate e quelle target.
def Aggiornamento_Main(modello:CustomDQNModel, modello_target:CustomDQNModel, stati, azioni, ricompense, stati_successivi, terminati, gamma):
    
    # NODO: Estrazione di stati successivi, ricompense e flag di terminazione dalle esperienze.
    tensore_stati_successivi, ricompense, terminati = Estrai_Stati(stati_successivi, ricompense, terminati)
    stati_correnti, _, _ = Estrai_Stati(stati, ricompense, terminati)

    # NODO: Utilizzo del modello target per calcolare il valore Q massimo per ogni stato successivo.
    #       Questo passaggio � parte dell'equazione di Bellman per il Q-learning.
    # Converti il tensore booleano in un tensore di interi (1 per True, 0 per False)
    terminati_int = tf.cast(terminati, tf.float32)
    Q_target = ricompense + gamma * np.max(modello_target.predict(tensore_stati_successivi), axis=1)*(1-terminati_int) 

    # NODO: Utilizzo del modello principale per ottenere le stime Q per le azioni in ogni stato del batch corrente.
    # FIX: Q_stime restituisce non restituisce un valore a zero, possibile che resti salvato nello stato un reward vecchio per le azioni inutili
    Q_stime = modello.predict(tensore_stati_successivi)
    debu = Q_stime[0]

    # NODO: Aggiornamento delle stime Q con i valori target Q per le azioni effettivamente intraprese.
    #       Ci� consente di allineare le stime del modello con le ricompense osservate e le stime di ricompensa futura.
    # FIX: Q_stime non e allineato con azioni, forse dovrebbe essere un tensore unico? !FORSE!
    Q_stime[np.arange(len(Q_stime)), azioni] = Q_target

    # TODO: Capire se passare i dati degli stati e degli stati_futuri , appartenenti allo stesso bach la procedura corretta e ottimale
    # NODO: Addestramento del modello con gli stati correnti e le stime Q aggiornate.
    #       Questo passaggio ottimizza il modello per predire meglio i valori Q basati sulle esperienze di addestramento.
    fitness = modello.fit(stati_correnti, Q_stime, epochs=1, verbose=2)

    print(fitness.history)

    # TODO: VAlutare approfonditamente il modello e ripetere il debug

    debu = 'Finisch'

# Aggiornamento della Rete Target
# La rete target viene aggiornata copiando i pesi dalla rete principale. Questo pu� essere fatto completamente o in maniera pi� graduale (soft update).
def Aggiornamento_Target(modello:CustomDQNModel, modello_target:CustomDQNModel, tau):
    pesi_principali = modello.get_weights()
    pesi_target = modello_target.get_weights()

    for i in range(len(pesi_target)):
        pesi_target[i] = tau * pesi_principali[i] + (1-tau) * pesi_target[i]
    modello_target.set_weights(pesi_target)

def estrapola_tensore(stato : np.array , ricompensa, terminato):
    # TODO : aggiustare il conversore in tensori
    posizioni = []
    for dizionario in stato:
        posizione = dizionario['posizione']
        posizione = np.argmax(posizione)
        prezzo = dizionario['prezzo']
        stato_i = np.array((posizione,prezzo))
        # TODO : servira un reshape della lista di tensori
        stato_list = tf.convert_to_tensor(stato_i , dtype=tf.float64)
        posizioni.append(stato_list)

    ricompensa_t = tf.convert_to_tensor(ricompensa, dtype=tf.float32)

    terminato_t = tf.convert_to_tensor(terminato, dtype=tf.bool)

    stato_t = tf.convert_to_tensor(posizioni)

    return stato_t , ricompensa_t , terminato_t

def Train(n_episodi, gamma, intervallo_aggiornamento_taget, tau, model : nn = main_network, ambiente : amb = env_evoluto):
    # Definizione del replay eseguita fuori dalla funzione quindi resettatta fuori dalla funzione
    # STats:
    resoult=[]
    actions = []
    positions = []

    for episodio in range(n_episodi):
        stato = ambiente.reset() # Reset

        # NODO : ottengo il tensore dello stato
        stato = estrapola_tensore(stato[0], stato[1], stato[2])

        # HINT: Libreria Math per operazioni fra tensori
        while not tf.math.equal(stato[2] , True):

            deb = tf.math.equal(stato[2] , True)

            # Aggiungo una dimensione al tensore
            tens = tf.expand_dims(stato[0], axis=0)
            
            # Selezione Azione
            # NODO: Scelgo l'azione con la progressiva riduzione di esplorativita
            # TODO : sistemare l'epsilon Start 
            epsilon = reduce_epsilon(0.1, ambiente.current_step)
            azione = epsylon_greedy_policy(state=tens, model=model, epsilon=epsilon) 

            azione = np.argmax(azione)
            actions.append(azione)

            # NODO: Esecuzione Azione / Ossevazione
            nuovo_stato, ricompensa, done, _ = ambiente.step(azione) 

            # HACK: Pobabilmente inutile
            ## Converto l'azione in Tensore dopo averla eseguita per passare un int all ambiente
            #azione = tf.convert_to_tensor(azione , dtype=tf.int8)


            # Estraggo E Aggiungo Una dimensione al nuvo stato
            positions.append(nuovo_stato[0]['posizione'])
            nuovo_stato = estrapola_tensore(nuovo_stato, ricompensa, done)
            nuovo_stato_tens = tf.expand_dims(nuovo_stato[0], axis=0)

            # TODO: sistemare il banch
            # HINT: E normale che nella logica stato, azione , stato_successivo , ricompensa , la ricompensa arrivi dalla lettura dello stato successivo all'azione?
            
            #seguendo quanto sopra pongo la ricompensa e done pari all ultima ricompensa ottenuta
            ricompensa = nuovo_stato[1]
            done = nuovo_stato[2]

            # Memorizzazione
            # HACK : Debug buffer
            replay_buffer.push(state=tens, action=azione, reward=ricompensa, next_state=nuovo_stato, done=done )

            # Aggiornamento dello stato ???? non converrebbe avere lo stato come proprieta intrinseca dell ambiente? perche devo aggiornare lo stato ad ogni step?
            stato = nuovo_stato

            ## TODO: escluso per mancanza di corrispondenza con azione step
            if len(replay_buffer) > batch_size:
                # Campionamento Hard_Coded
                batch = campionamento(battch_size=batch_size, replay_buffer_=replay_buffer)
                # Aggiornamento rete principale 
                Aggiornamento_Main(main_network, target_network, *batch, gamma)

            print(ambiente.current_step)
            print(stato[0])
            print(stato[2])

        # All interno delle epoche ma non dei passi
        # TODO: escluso per mancanza di corrispondenza con azione step
        if episodio % intervallo_aggiornamento_taget == 0:
            # HACK: debugging target train
            # TODO: verificare funzioni e parametri
            Aggiornamento_Target(main_network, target_network, tau)

        var = copy.copy(ambiente.current_balance)

        resoult.append(f'remained balance: {var}')

    print('Trained')
    print(resoult)
    #print(actions)
    #print(positions)


def converti_in_tensore(stati, n_timesteps, n_features):
    # Converte la tupla di array in un unico array NumPy
    # Assumendo che ogni array in 'stati' rappresenti un timestep
    stati_np = np.array(stati)

    # Ridimensiona l'array per avere la forma [batch_size, n_timesteps, n_features]
    stati_np = stati_np.reshape(-1, n_timesteps, n_features)

    # Converti l'array NumPy in un tensore TensorFlow
    stati_tensor = tf.convert_to_tensor(stati_np, dtype=tf.float32)

    return stati_tensor

def createTensor(batchSize = 1, n_timesteps = 20, n_features = 2):
    # Crea dati casuali per l'esempio
    # np.random.random crea un array con valori casuali compresi tra 0 e 1
    example_data = np.random.random((batch_size, n_timesteps, n_features))
    
    # Converti i dati in un tensore TensorFlow
    example_tensor = tf.convert_to_tensor(example_data, dtype=tf.float32)
    
    # Stampa il tensore per visualizzarlo
    print(example_tensor)
    return example_tensor

Train(n_episodi, 0.5, 10 , 0.5)
# TODO: valutare e salvare il modello



#Evalutation

#Save_Load Model
def Save_Model(model : CustomDQNModel, destination):
    model.save(filepath=destination)