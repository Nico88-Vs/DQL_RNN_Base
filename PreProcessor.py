class Preprocessor:
    def __init__(self):
        # Inizializza qui eventuali parametri o configurazioni necessarie
        pass

    def preprocess(self, step_outputs):
        # step_outputs è una lista di tuple (stato, azione, ricompensa, stato_successivo, terminato)
        batch_stati = []
        batch_azioni = []
        batch_ricompense = []
        batch_stati_successivi = []
        batch_terminati = []

        for stato, azione, ricompensa, stato_successivo, terminato in step_outputs:
            # Preprocessing dei singoli componenti (es: normalizzazione)
            stato = preprocess_stato(stato)
            stato_successivo = preprocess_stato(stato_successivo)
            azione = preprocess_azione(azione)
            ricompensa = preprocess_ricompensa(ricompensa)
            terminato = preprocess_terminato(terminato)

            # Aggiunta al batch
            batch_stati.append(stato)
            batch_azioni.append(azione)
            batch_ricompense.append(ricompensa)
            batch_stati_successivi.append(stato_successivo)
            batch_terminati.append(terminato)

        return batch_stati, batch_azioni, batch_ricompense, batch_stati_successivi, batch_terminati

# Esempio di utilizzo
preprocessor = Preprocessor()
step_outputs = [...]  # Ottieni i dati dal metodo step()
batches = preprocessor.preprocess(step_outputs)
