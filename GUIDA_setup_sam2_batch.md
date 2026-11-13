# **Guida: eseguire la pipeline SAM2 in locale (batch, senza SLImTAG)**

**Aggiornamento:** ora il punto/box di partenza viene calcolato **automaticamente** (`auto_segment.py`), non serve più cliccare a mano su ogni oggetto. Ho testato la pipeline end-to-end sui due dataset reali che mi avete mandato (berretto 16 frame, cuffie 41 frame), risultati riportati alla fine. `pick_points.py` resta utile solo come correzione manuale per il raro oggetto che l'automatismo segnala come incompleto (vedi punto 9).

Presuppone una macchina Linux (o WSL su Windows) con GPU NVIDIA e driver CUDA già funzionanti (`nvidia-smi` deve rispondere). Se usate Windows nativo i comandi sono gli stessi, cambia solo l'attivazione del venv (`sam2_env\Scripts\activate`).

## 1\. Ambiente Python separato

Per non toccare l'ambiente esistente di SLImTAG (che usa SAM 1):

python3 \-m venv sam2\_env

source sam2\_env/bin/activate   \# Windows: sam2\_env\\Scripts\\activate

## 2\. PyTorch con supporto CUDA

Controllate la vostra versione CUDA (`nvidia-smi`, in alto a destra) e installate la build di PyTorch corrispondente seguendo il selettore ufficiale: [https\://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

Esempio (CUDA 12.x):

pip install torch torchvision \--index-url https\://download.pytorch.org/whl/cu124

Verificate che la GPU sia vista correttamente:

python3 \-c "import torch; print(torch.cuda.is\_available(), torch.cuda.get\_device\_name(0))"

Deve stampare `True` e il nome della vostra scheda. Se stampa `False`, fermatevi qui e sistemate driver/versione CUDA prima di proseguire: senza GPU rilevata lo script funziona comunque ma alla velocità (bassa) vista nei miei test.

Serve anche opencv (usato dal calcolo automatico del box/punti):

pip install opencv-python-headless

## 2bis\. In alternativa: PyTorch CPU-only (senza GPU)

SAM2 gira anche senza GPU: funziona comunque, semplicemente più lento. Se la macchina non ha una GPU NVIDIA (o non volete installare CUDA), installate la build CPU-only invece del comando della sezione 2 — è anche un download molto più leggero:

pip install torch torchvision \--index-url https\://download.pytorch.org/whl/cpu

Verificate l'installazione:

python3 \-c "import torch; print(torch.cuda.is\_available())"

In questo caso è **normale** che stampi `False` — non è un errore, vuol dire solo che la build CPU-only funziona (non cerca nemmeno la GPU). Il resto della guida non cambia: PyTorch sceglie da solo CPU o GPU in base a cosa trova disponibile.

Tempi di riferimento misurati su CPU con il modello "small" (pipeline completa, incluso il calcolo automatico del prompt): berretto 16 frame \~4m20s, cuffie 41 frame \~7m30s. Per un batch con molti oggetti i tempi si moltiplicano di conseguenza. Su CPU conviene preferire i checkpoint **"tiny"** o **"small"**: "large" è circa 8x più lento di "tiny", per un guadagno di qualità marginale rispetto a "small" sui nostri dati.

## 3\. Clonare e installare SAM 2

git clone https\://github.com/facebookresearch/sam2.git

cd sam2

pip install \-e .

## 4\. Scaricare i checkpoint

**Metodo consigliato (fonte ufficiale Meta)** — a differenza del mio ambiente di test, voi non dovreste avere restrizioni di rete verso `fbaipublicfiles.com`:

cd checkpoints

bash download\_ckpts.sh   \# scarica tutte le taglie; ci vuole un po', sono \~5GB in totale

cd ..

Se volete solo una o due taglie specifiche invece di tutte, scaricatele singolarmente:

cd checkpoints

curl \-L \-O https\://dl.fbaipublicfiles.com/segment\_anything\_2/092824/sam2.1\_hiera\_small.pt

curl \-L \-O https\://dl.fbaipublicfiles.com/segment\_anything\_2/092824/sam2.1\_hiera\_large.pt

cd ..

**Se la vostra rete blocca `fbaipublicfiles.com`** (es. firewall aziendale) — è il problema che ho avuto io in sandbox — Ultralytics rimirrora gli stessi pesi identici (verificato: si caricano senza alcuna chiave mancante/inattesa) come release GitHub:

cd checkpoints

curl \-L \-o sam2.1\_hiera\_tiny.pt   https\://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1\_t.pt

curl \-L \-o sam2.1\_hiera\_small.pt  https\://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1\_s.pt

curl \-L \-o sam2.1\_hiera\_base\_plus.pt https\://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1\_b.pt

curl \-L \-o sam2.1\_hiera\_large.pt  https\://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1\_l.pt

cd ..

I file di configurazione corrispondenti sono già nel repo clonato, non serve scaricare altro:

configs/sam2.1/sam2.1\_hiera\_t.yaml   (tiny)

configs/sam2.1/sam2.1\_hiera\_s.yaml   (small)

configs/sam2.1/sam2.1\_hiera\_b+.yaml  (base\_plus)

configs/sam2.1/sam2.1\_hiera\_l.yaml   (large)

Dai test fatti in sandbox: "small" è un buon compromesso di partenza qualità/velocità per oggetti con dettagli sottili (cavi, montature); "tiny" è troppo grossolano su quei casi; "large" è leggermente meglio ma molto più lento (su CPU quasi 8x più lento di "tiny" per un guadagno di qualità marginale rispetto a "small" — su GPU va verificato, il rapporto potrebbe cambiare).

## 5\. Posizionare gli script

Copiate questi 4 file dentro la cartella `sam2/` appena clonata (vanno lanciati con working directory `sam2/` perché il caricamento dei file di config si aspetta i percorsi relativi mostrati sopra):

- `auto_prompt.py` — calcola box/punti automaticamente (nessuna dipendenza da torch/sam2, solo opencv/numpy)  
- `sam2_video_propagate.py` — propagazione SAM2 sulla sequenza (contiene la funzione `propagate()` riusata da `auto_segment.py`, e resta usabile da sola con `--point`/`--box` scelti a mano se preferite il controllo manuale)  
- `auto_segment.py` — **script principale**: mette insieme i due sopra \+ controllo automatico di qualità del prompt, singolo oggetto o intero batch  
- `pick_points.py` — solo per correggere a mano l'oggetto raro che l'automatismo segnala come incompleto (vedi punto 9\)

## 6\. Preparare i dati

Un oggetto \= una cartella con dentro le sue viste (stesso formato dello zip `headphone40.zip` che mi avete mandato: `RGB_00_....png`, `RGB_01_....png`, ecc. — va bene qualunque nome, basta che l'ordine alfabetico corrisponda all'ordine di rotazione).

dataset/

  oggetto\_01/

    RGB\_00\_....png

    RGB\_01\_....png

    ...

  oggetto\_02/

    ...

## 7\. Eseguire su un singolo oggetto (box/punti automatici)

Non serve più cliccare nulla: `auto_segment.py` calcola da solo box e punti iniziali (bordo immagine \= sfondo, distanza colore in Lab, chiusura morfologica per includere anche parti sottili come i cavi — dettagli in testa a `auto_prompt.py`) e li passa a `sam2_video_propagate.py`:

python auto\_segment.py \\

  \--frames\_dir dataset/oggetto\_01 \\

  \--out\_dir output/oggetto\_01 \\

  \--checkpoint checkpoints/sam2.1\_hiera\_small.pt \\

  \--config configs/sam2.1/sam2.1\_hiera\_s.yaml \\

  \--full\_loop

`--full_loop` va aggiunto se le viste coprono l'intera rotazione di 360° (72 viste standard): propaga anche all'indietro dall'ultimo frame e riporta quanto la stima "in avanti" e quella "all'indietro" si assomigliano a metà sequenza (`loop_closure_iou_midpoint`, vicino a 1 \= buon segno).

Prima di lanciare la propagazione, lo script salva `output/oggetto_01/auto_prompt_overlay.png`: un'immagine col box e i punti calcolati disegnati sopra al primo frame, utile per un controllo visivo rapido prima di aspettare tutta la sequenza.

## 8\. Eseguire su tutto il dataset (batch)

python auto\_segment.py \\

  \--batch\_root dataset/ \\

  \--out\_root output/ \\

  \--checkpoint checkpoints/sam2.1\_hiera\_small.pt \\

  \--config configs/sam2.1/sam2.1\_hiera\_s.yaml \\

  \--full\_loop

`dataset/` deve contenere una sottocartella per oggetto (stessa struttura del punto 6). I pesi SAM2 vengono caricati **una sola volta** e riusati per tutti gli oggetti — molto più veloce che rilanciare lo script uno per uno. Alla fine trovate `output/batch_summary.csv` e `.json` con, riga per riga, lo stato di ogni oggetto (`ok`/`error`), il `prompt_quality_ratio` e se è da rivedere (vedi punto 9).

## 9\. Cosa trovate in output, e come sapere se un oggetto è da rivedere

Per ogni oggetto, dentro `output/<nome_oggetto>/`:

- `color_mask/<stesso nome del frame originale>.png` — maschera RGBA: pixel oggetto colorati `(158,31,99)` pieno, sfondo trasparente (stesso colore "dark magenta" di `default_mask_colors` in `config.toml` di SLImTAG; per cambiarlo: `--mask_color R G B`).  
- `auto_prompt.json` / `auto_prompt_overlay.png` — box e punti calcolati automaticamente sul primo frame.  
- `qc_report.json` — due controlli automatici, **nessuno dei due richiede di guardare le maschere a occhio**:  
  1. **`prompt_quality_ratio`** \= area della maschera SAM sul frame 0 / area della stima grezza indipendente (classica, non-SAM) sullo stesso frame. Risponde direttamente alla domanda "il punto/box di partenza era buono?" — un rapporto molto sotto 1 (soglia di default 0.5, `--prompt_quality_threshold`) vuol dire che SAM ha preso solo una parte dell'oggetto. Validato sui vostri dati: con un solo punto senza box sulle cuffie il rapporto viene 0.17 (SAM prende solo un pezzo, \~16.000px contro \~96.000px attesi) — con box \+ punti automatici viene 1.04, cioè sostanzialmente l'intero oggetto.  
  2. `outlier_filenames` — frame dentro la sequenza dove l'area della maschera cambia troppo rispetto alla media o rispetto al frame precedente (soglie `--area_qc_threshold` e `--area_jump_qc_threshold`), possibile segno di deriva della propagazione durante la rotazione.

Se `needs_manual_review_prompt` è `true` (o vedete `outlier_filenames` non vuoto), l'oggetto va controllato: aprite `auto_prompt_overlay.png` per capire cosa ha visto l'automatismo, e se serve correggete a mano con `pick_points.py`

+ un lancio diretto di `sam2_video_propagate.py --point ... --box ...` su quel solo oggetto (vedi il suo `--help` per tutte le opzioni). Nei test fatti sui vostri due dataset (berretto, cuffie) l'automatismo non ha mai avuto bisogno di questa correzione.

## Risultati dei test end-to-end (berretto 16 frame, cuffie 41 frame, modello "small")

| Oggetto | prompt\_quality\_ratio | Frame da rivedere (outlier area) | loop\_closure\_iou |
| :---- | :---- | :---- | :---- |
| Berretto | 1.08 | 0/16 | 0.88 (arco parziale, non 360°) |
| Cuffie | 1.04 | 6/41 (variazione legittima, oggetto articolato — vedi sotto) | 0.91 |

I 6 frame delle cuffie segnalati riflettono una variazione reale e prevista dell'area proiettata (le cuffie non sono un oggetto convesso), non un errore di SAM: il controllo "salto frame-a-frame" (più robusto per oggetti articolati) infatti non segnala nessun frame lì. Ho anche verificato direttamente che con un solo punto e senza box (il caso di fallimento visto in origine sulle cuffie) `prompt_quality_ratio` scende a 0.17 e l'oggetto verrebbe correttamente marcato da rivedere — conferma che la soglia di default (0.5) distingue bene i due casi sui vostri dati reali.

## Problemi noti / cose da controllare

- Se `python -c "import torch; print(torch.cuda.is_available())"` stampa `False`, lo script funziona ma è lentissimo (i miei tempi di riferimento su CPU, pipeline completa incluso il calcolo automatico del prompt: berretto 16 frame \~4m20s, cuffie 41 frame \~7m30s, modello "small" — su GPU dovrebbero scendere parecchio, ma non ho potuto verificare quanto).  
- SAM2 a volte cambia leggermente l'API tra versioni del repo: se `add_new_points_or_box` o `propagate_in_video` danno errori di firma, controllate la versione del repo clonato contro quella con cui ho sviluppato lo script (verificata contro il repo `facebookresearch/sam2` clonato a settembre 2026).  
- L'automatismo assume che lo sfondo sia visibile sui bordi del frame e che l'oggetto sia centrato (come nel vostro setup con pedana). Se qualche oggetto è ripreso molto vicino al bordo dell'inquadratura, il calcolo del colore di sfondo può risultare contaminato: in quel caso il `prompt_quality_ratio` basso ve lo segnala comunque.

