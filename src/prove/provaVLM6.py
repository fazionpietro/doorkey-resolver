from monorepo import GroqLLM, load_api_keys, GROQ_MULTIMODAL_MODEL_ID
from dotenv import load_dotenv
from PIL import Image

PROMPT1 = """Sei un sistema AI esperto in navigazione spaziale e deduzione logica.
Stai guidando un agente (il triangolo rosso) verso il suo obiettivo finale (il quadrato verde) in un GridWorld 2D. Non conosci le regole specifiche del gioco, devi dedurle dalla storia visiva fornita.

REGOLE VISIVE:
- AGENTE: Freccia rossa. La punta del triangolo indica la direzione frontale (NORD, SUD, EST, OVEST). Osserva bene la sua direzione e non confonderti con gli altri vertici.
- GOAL FINALE: Quadrato verde.
- OSTACOLI FISSI (Muri): Blocchi grigi. Non possono essere attraversati. Se l'agente cerca di andarci contro, non si muoverà.
- OGGETTI INTERATTIVI: Blocchi colorati (es. giallo). Possono essere aggirati o, se bloccano la strada, si può tentare di interagirci.
- Oggetti con lo stesso colore sono spesso collegati (es. chiave gialla apre porta gialla).

REGOLE FONDAMENTALI
- Se vedi un oggetto sulla griglia allora quell'oggetto NON è in mano all'agente.
- Controlla sempre dove sta guardando l'agente, occhio alla punta del triangolo.
- l'agente si puo muovere solo di una casella alla volta e puo fare una azione alla volta.


L'immagine fornita mostra una sequenza di 5 frame: il frame più a destra è il presente, e gli altri quattro a sinistra mostrano il passato recente in ordine cronologico.

IL TUO COMPITO MENTALE OBBIGATORIO:
1. Analizza l'intera sequenza temporale per capire l'intento attuale dell'agente.
2. Basandoti sul frame del presente (quello a destra), identifica l'obiettivo intermedio immediato .
3. Decidi l'azione successiva più efficace per avvicinarsi a quell'obiettivo intermedio.

AZIONI DISPONIBILI:
0: LEFT (Ruota sul posto di 90 gradi a sinistra)
1: RIGHT (Ruota sul posto di 90 gradi a destra)
2: FORWARD (Avanza di un blocco nella direzione in cui guarda)
3: PICKUP (Raccogli l'oggetto sul blocco di fronte)
4: DROP (Lascia l'oggetto se ne hai raccolto uno)
5: TOGGLE (Interagisci con l'oggetto di fronte, es. apri una porta)

FORMATO DI RISPOSTA OBBLIGATORIO:
Non aggiungere testo al di fuori di questi tag.

<DECISIONE>
(Max 150 parole). 
Scrivi il tuo ragionamento conciso. 
1. Obiettivo intermedio attuale. 
2. Giustificazione dell'azione. 
3. Solo il numero dell'azione successiva nel formato 'Azione: X' (es: `Azione: 2`).
</DECISIONE>
"""


PROMPT2 = """Sei un sistema AI esperto in navigazione spaziale e deduzione logica.
Stai guidando un agente (il triangolo rosso) verso il suo obiettivo finale (il quadrato verde) in un GridWorld 2D. Non conosci le regole specifiche del gioco, devi dedurle dalla storia visiva fornita.

REGOLE VISIVE:
- AGENTE: Freccia rossa. La punta del triangolo indica la direzione frontale (NORD, SUD, EST, OVEST). Osserva bene la sua direzione e non confonderti con gli altri vertici.
- GOAL FINALE: Quadrato verde.
- OSTACOLI FISSI (Muri): Blocchi grigi. Non possono essere attraversati. Se l'agente cerca di andarci contro, non si muoverà.
- OGGETTI INTERATTIVI: Blocchi colorati (es. giallo). Possono essere aggirati o, se bloccano la strada, si può tentare di interagirci.
- Oggetti con lo stesso colore sono spesso collegati (es. chiave gialla apre porta gialla).

REGOLE FONDAMENTALI
- Se vedi un oggetto sulla griglia allora quell'oggetto NON è in mano all'agente.
- Controlla sempre dove sta guardando l'agente, occhio alla punta del triangolo.
- l'agente si puo muovere solo di una casella alla volta e puo fare una azione alla volta.


L'immagine fornita mostra una sequenza di 5 frame: il frame più a destra è il presente, 
e gli altri quattro a sinistra mostrano il passato recente in ordine cronologico.

IL TUO COMPITO MENTALE OBBIGATORIO:
1. Analizza l'intera sequenza temporale per capire l'intento attuale dell'agente.
2. Basandoti sul frame del presente (quello a destra), identifica l'obiettivo intermedio immediato .

3. valuta con un numero decimale tra [-1.0,1.0] la performance dell'agente.
4. dai un valore decimale tra [0.0,1.0] che rappresenti la percentuale di completamento della task dell'agente dove 1.0
è aver raggiunto il goal


FORMATO DI RISPOSTA OBBLIGATORIO:
Non aggiungere testo al di fuori di questi tag.

<ANALISI>
(Max 150 parole). 
Scrivi il tuo ragionamento conciso. 
1. Obiettivo intermedio attuale. 
2. Giustificazione dell'azione. 

</ANALISI>
<METRICHE>3. reward, 4. percentuale</METRICHE>
"""
PROMPT3 = """Sei un sistema AI esperto in navigazione spaziale e deduzione logica.
Stai guidando un agente (il triangolo rosso) verso il suo obiettivo finale (il quadrato verde) in un GridWorld 2D. Non conosci le regole specifiche del gioco, devi dedurle dalla storia visiva fornita.

REGOLE VISIVE:
- AGENTE: Freccia rossa. La punta del triangolo indica la direzione frontale (NORD, SUD, EST, OVEST). Osserva bene la sua direzione e non confonderti con gli altri vertici.
- GOAL FINALE: Quadrato verde.
- OSTACOLI FISSI (Muri): Blocchi grigi. Non possono essere attraversati. Se l'agente cerca di andarci contro, non si muoverà.
- OGGETTI INTERATTIVI: Blocchi colorati (es. giallo). Possono essere aggirati o, se bloccano la strada, si può tentare di interagirci.
- Oggetti con lo stesso colore sono spesso collegati (es. chiave gialla apre porta gialla).

REGOLE FONDAMENTALI
- le immagini sono ordinate da destra a sinistra dove A DESTRA ce il PRESENTE e a SINISTRA IL PASSATO.
- Se vedi un oggetto sulla griglia allora quell'oggetto NON è in mano all'agente.
- Se nella sequenza delle immagini, in quelle piu a destra un oggetto è sparito allora è IN MANO all agente
- Controlla sempre dove sta guardando l'agente, occhio alla punta del triangolo.
- l'agente si puo muovere solo di una casella alla volta e puo fare una azione alla volta.
- per interagire con un oggetto l'agente deve essere in una casella adiacente (destra, sinistra, sopra, sotto, NON in diagonale) e 
deve essere rivolto verso l'agente.
- L'agente puo interagire solo con la casella direttamente di fronte all'agente (distanza 1)


IL TUO COMPITO MENTALE OBBIGATORIO:
0. leggi e analizza l'ambiente in base alle regole fondamentali e le regole visive, stai molto attento agli oggetti presenti o non presenti sulla griglia.
1. Analizza l'intera sequenza temporale per capire l'intento attuale dell'agente.
2. Basandoti sul frame del presente (quello a destra), identifica l'obiettivo intermedio immediato .
3. Decidi l'azione successiva più efficace per avvicinarsi a quell'obiettivo intermedio.
4. valuta con un numero decimale tra [-1.0,1.0] la performance dell'agente.
5. dai un valore decimale tra [0.0,1.0] che rappresenti la percentuale di completamento della task dell'agente dove 1.0
è aver raggiunto il goal



AZIONI DISPONIBILI:
0: LEFT (Ruota sul posto di 90 gradi a sinistra)
1: RIGHT (Ruota sul posto di 90 gradi a destra)
2: FORWARD (Avanza di un blocco nella direzione in cui guarda)
3: PICKUP (Raccogli l'oggetto sul blocco di fronte)
4: DROP (Lascia l'oggetto se ne hai raccolto uno)
5: TOGGLE (Interagisci con l'oggetto di fronte, es. apri una porta)




FORMATO DI RISPOSTA OBBLIGATORIO:
Non aggiungere testo al di fuori di questi tag.

<ANALISI>
(Max 150 parole). 
Scrivi il tuo ragionamento conciso. 
1. Obiettivo intermedio attuale. 
2. Giustificazione dell'azione. 
3. Azione successiva consigliata

</ANALISI>
<METRICHE>4. reward, 5. percentuale</METRICHE>
"""

PROMPT4 = """sei un evalutatore di policy di agenti che si muovono su ambienti GridWorld 2d

REGOLE FONDAMENTALI
- la foto a destra è il presente, quella a sinistra il passato
- il GOAL è la casella verde
- l'AGENTE è una freccia rossa, puo interagire solo con la casella che sta puntando con la PUNTA del triangolo
- l'Agente puo interagire solo con quello che ha di fronte nella casella immediatamente successiva
- l'agente vede tutto l'ambiente
- l'agente se fa forward si muove nella direzione da lui puntata, quindi per cambiare direzione deve per forza fare left o right
- per girarsi di 180 gradi l'agente deve fare 2 volte left o 2 volte right
- I muri sono grigi e non possono essere attraversati
- il pavimento è nero
- le caselle o gli oggetti di altri colori sono oggetti che possono essere raccolti o interagiti
- è importante la forma dell'oggetto
- se vedi un oggetto colorato nell'immagine NON è stato raccolto dall'agente
- oggetti dello stesso colore possono interagire assieme (es porta e chiave)
- gli oggetti sono delle forme stilizzate di oggetti reali
- per raccogliere un oggetto l'agente deve puntare la casella dell'oggetto e deve essere subito adiacente.

AZIONI DISPONIBILI:
0: LEFT (Ruota sul posto di 90 gradi a sinistra)
1: RIGHT (Ruota sul posto di 90 gradi a destra)
2: FORWARD (Avanza di un blocco nella direzione in cui guarda)
3: PICKUP (Raccogli l'oggetto sul blocco di fronte)
4: DROP (Lascia l'oggetto se ne hai raccolto uno)
5: TOGGLE (Interagisci con l'oggetto di fronte, es. apri una porta)

ULTIMA AZIONE DELL'AGENTE
- FORWARD
DIREZIONE PUNTATA AGENTE
- DOWN

RISPOSTA OBBLIGATORIA:
<ANALISI>
comprendi le REGOLE FONDAMENTALI
comprendi le ultima azione e la direzione dell'agente, IMPORTANTE
comprendi l'ambiente, dove punta il triangolo dell'agente e il task dell'agente e i vari passi logici che deve completare per raggiungere il goal
suggerisci e giustifica una sequenza di azioni che l'agente dovrebbe fare
seguendo i tuoi suggerimenti di azioni identifica tra le AZIONI DISPONIBILI le 5 successive
</ANALISI>

RISPOSTA OBBLIGATORIA, formato 5 numeri, NO testo aggiuntivo:
<ACTION>dammi una sequenza di 5 numeri rappresentanti le prossime 5 azioni che dovrebbe fare l'agente (es 1,1,3,4)</ACTION>
"""

PROMPT5 = """sei un evalutatore di policy di agenti che si muovono su ambienti GridWorld 2d

REGOLE FONDAMENTALI
- la foto a destra è il presente, quella a sinistra il passato
- il GOAL è la casella verde
- l'AGENTE è una freccia rossa, puo interagire solo con la casella che sta puntando con la PUNTA del triangolo
- l'Agente puo interagire solo con quello che ha di fronte nella casella immediatamente successiva
- l'agente vede tutto l'ambiente
- l'agente se fa forward si muove nella direzione da lui puntata, quindi per cambiare direzione deve per forza fare left o right
- per girarsi di 180 gradi l'agente deve fare 2 volte left o 2 volte right
- l'agente puo fare un azione alla volta
- I muri sono grigi e non possono essere attraversati
- il pavimento è nero
- le caselle o gli oggetti di altri colori sono oggetti che possono essere raccolti o interagiti
- è importante la forma dell'oggetto
- se vedi un oggetto colorato nell'immagine NON è stato raccolto dall'agente
- oggetti dello stesso colore possono interagire assieme (es porta e chiave)
- gli oggetti sono delle forme stilizzate di oggetti reali
- per raccogliere un oggetto l'agente deve puntare la casella dell'oggetto e deve essere subito adiacente.

AZIONI DISPONIBILI:
0: LEFT (Ruota sul posto di 90 gradi a sinistra)
1: RIGHT (Ruota sul posto di 90 gradi a destra)
2: FORWARD (Avanza di un blocco nella direzione in cui guarda)
3: PICKUP (Raccogli l'oggetto sul blocco di fronte)
4: DROP (Lascia l'oggetto se ne hai raccolto uno)
5: TOGGLE (Interagisci con l'oggetto di fronte, es. apri una porta)

ULTIMA AZIONE DELL'AGENTE
- FORWARD
DIREZIONE PUNTATA AGENTE
- DOWN

RISPOSTA OBBLIGATORIA 1:
<ANALISI>
comprendi le REGOLE FONDAMENTALI
comprendi le ultima azione e la direzione dell'agente, IMPORTANTE
comprendi l'ambiente, dove punta il triangolo dell'agente e il task dell'agente e i vari passi logici che deve completare per raggiungere il goal
suggerisci e giustifica una sequenza di azioni che l'agente dovrebbe fare, oggetti da raggiungere o con cui interagire (che chiameremo stage)
seguendo i tuoi suggerimenti di azioni identifica tra le AZIONI DISPONIBILI le 5 successive
Ora valuta la performance dell'agente da l'immagine di sinistra a quella di destra con un numero decimale tra -1.0 e 1.0 dove i numeri positivi sono se progredisce e quelli negativi si regredisce nel raggiungere il goal, la reward deve essre proporzionale al valore dell'azione, quindi per esempio raggiungere il goal è 1, avvicinarsi o utilizzare oggetti utili puo essere 0.3, 0.5 ecc
Ora valuta la percentuale di completamento dell'agente dove 1.0 è aver raggiunto il goal verde
ora valuta la prossima direzione in cui l'agente dovrebbe girarsi
ora valuta la distanza relativa dx e dy rispetto al prossimo oggetto, goal 

</ANALISI>

RISPOSTA OBBLIGATORIA 2. dammi SOLO numeri nel formato seguente:
<METRICHE>reward, progresso, direzione, dx, dy</METRICHE>
"""


PROMPT6 = """

RUOLO
Sei l'occhio e il cervello di un agente in un ambiente GridWorld 2D sconosciuto.
Immagine SINISTRA = PASSATO. Immagine DESTRA = PRESENTE.
IL TUO OBIETTIVO E' CAPIRE LE REGOLE DEL LIVELLO E PRODURRE ESATTAMENTE IL FORMATO RICHIESTO ALLA FINE.

REGOLE FISICHE INVALICABILI (UNIVERSALI)

    AGENTE: È il TRIANGOLO ROSSO. La punta indica dove guarda (Nord=Su, Est=Destra, Sud=Giù, Ovest=Sinistra).
    MOVIMENTO: Si muove di 1 sola casella per volta. FORWARD avanza verso dove punta. LEFT/RIGHT ruotano sul posto di 90 gradi.
    MURI (GRIGI): Sono blocchi SOLIDI e IMPASSABILI. L'agente si muove solo sul pavimento nero (vuoto).
    INTERAZIONE RAGGIO CORTO: L'agente interagisce (raccoglie/usa/apre) SOLO con la singola casella esattamente attaccata alla sua punta.

COME DEDURRE L'AMBIENTE (IL TUO COMPITO)

    GOAL: Il quadrato VERDE è sempre il traguardo finale.
    DEDUZIONE DEGLI OGGETTI: Non conosci a priori cosa facciano gli altri oggetti colorati. Devi dedurlo osservando la scena.
    REGOLE LOGICHE DA CERCARE: Se il percorso verso il goal è bloccato da un ostacolo, cerca sulla mappa altri oggetti. Esiste una regola aurea: Oggetti dello stesso colore o forma sono quasi sempre collegati logicamente. Se vedi un ostacolo di colore X, probabilmente ti serve un oggetto raccoglibile di colore X per superarlo.

AZIONI DISPONIBILI:
0: LEFT, 1: RIGHT, 2: FORWARD, 3: PICKUP (Raccogli), 4: DROP (Lascia), 5: TOGGLE (Interagisci/Usa).

ISTRUZIONI PER L'ANALISI (Esegui i passi in ordine rigoroso)
Passo 1 (Analisi Passato): Compara l'immagine sinistra e destra. Che azione è stata appena fatta?
Passo 2 (Visione frontale): Guarda la SINGOLA casella esattamente attaccata alla punta del triangolo rosso. È vuota, è un muro, o è un oggetto?
Passo 3 (Deduzione Regole): Guarda tutta la mappa. Quali ostacoli bloccano il goal? Quali oggetti raccoglibili/interagibili vedi?DI che colore sono Controlla attentamente per non sbagliarti? Deduci la relazione logica tra loro (es. "Poiché l'ostacolo è colore X e l'oggetto libero è colore X, deduco che l'oggetto serve per rimuovere l'ostacolo").
Passo 4 (Piano): Qual è il prossimo step logico per risolvere il livello?
Passo 5 (Calcolo Metriche): Calcola la reward (-1.0 a 1.0 dove negativa se l'agente regredisce e positiva se si avvicina al completamento del puzzle e al raggiungimento del goal), il progresso (0.0 a 1.0), l'ID dell'azione consigliata (0-5), e la distanza relativa dx e dy (in numero di caselle) dall'agente all'oggetto target corrente.



"""


load_api_keys()

client = GroqLLM(model_id="meta-llama/llama-4-scout-17b-16e-instruct")
image = Image.open("debug.jpg")

response = client.ask(prompt=PROMPT6, images=[image])
print(response)
