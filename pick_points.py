"""
Piccola utility per leggere le coordinate pixel da cliccare sul primo frame
di ogni oggetto, da passare poi a --point / --neg_point / --box di
sam2_video_propagate.py. Serve solo a leggere le coordinate: non fa alcuna
inferenza, quindi non richiede torch/sam2 -- gira anche in un ambiente Python
minimale con solo matplotlib installato (pip install matplotlib).

Uso:
    python pick_points.py /path/to/oggetto_XYZ/frames/primo_frame.png

Si apre una finestra con l'immagine.
- Click SINISTRO = punto positivo (parte dell'oggetto)
- Click DESTRO = punto negativo (sfondo/buco da escludere)
- SHIFT + tieni premuto il tasto SINISTRO e trascina = disegna il box che
  racchiude l'intero oggetto: il rettangolo (giallo) segue il mouse mentre
  trascini, il rilascio del tasto fissa l'angolo opposto. Ripetere
  (nuovo Shift+trascinamento) ridisegna e sostituisce il box precedente, se
  serve correggerlo.

Il box e' importante quanto i punti quando SAM viene lanciato in automatico
su tutta la sequenza: e' quello che dice a SAM in quale zona dell'immagine
cercare l'oggetto, mentre i punti da soli possono selezionare solo una
sotto-parte se l'oggetto e' composto da piu' pezzi (vedi il caso delle
cuffie: un punto sul solo archetto puo' escludere i padiglioni). Per
correggere a mano un oggetto segnalato come incompleto conviene quindi quasi
sempre ridefinire ANCHE il box, non solo aggiungere punti.

Chiudi la finestra quando hai finito: lo script stampa direttamente i flag
pronti da incollare nel comando (--point, --neg_point, e --box se ne hai
disegnato uno).
"""
import sys

import matplotlib.pyplot as plt
from PIL import Image

pos_points = []
neg_points = []
box_corners = []  # 2 angoli (x, y) quando il box e' completo, altrimenti vuoto
box_patch = [None]  # rettangolo disegnato, per poterlo rimuovere/ridisegnare
drag_start = [None]  # angolo di partenza del trascinamento in corso (o None)
MIN_BOX_SIZE = 3  # px: sotto questa soglia un trascinamento e' considerato un click accidentale


def _clear_box():
    box_corners.clear()
    if box_patch[0] is not None:
        box_patch[0].remove()
        box_patch[0] = None


def on_press(event):
    if event.xdata is None or event.ydata is None:
        return
    if event.button == 1 and event.key == "shift":
        # inizio trascinamento: rimuove un box precedente e parte da zero
        _clear_box()
        drag_start[0] = (event.xdata, event.ydata)
        box_patch[0] = plt.Rectangle((event.xdata, event.ydata), 0, 0,
                                      fill=False, edgecolor="yellow", linewidth=2)
        plt.gca().add_patch(box_patch[0])
        plt.draw()
        return

    x, y = int(round(event.xdata)), int(round(event.ydata))
    if event.button == 1:  # sinistro senza shift = positivo
        pos_points.append((x, y))
        plt.plot(x, y, "g+", markersize=14, markeredgewidth=2)
        plt.draw()
    elif event.button == 3:  # destro = negativo
        neg_points.append((x, y))
        plt.plot(x, y, "rx", markersize=14, markeredgewidth=2)
        plt.draw()


def on_motion(event):
    if drag_start[0] is None or event.xdata is None or event.ydata is None:
        return
    x0, y0 = drag_start[0]
    left, right = sorted((x0, event.xdata))
    bottom, top = sorted((y0, event.ydata))
    box_patch[0].set_bounds(left, bottom, right - left, top - bottom)
    plt.draw()


def on_release(event):
    if drag_start[0] is None or event.button != 1:
        return
    x0, y0 = drag_start[0]
    drag_start[0] = None
    if event.xdata is None or event.ydata is None:
        _clear_box()
        plt.draw()
        return
    x1, y1 = event.xdata, event.ydata
    if abs(x1 - x0) < MIN_BOX_SIZE or abs(y1 - y0) < MIN_BOX_SIZE:
        # trascinamento troppo piccolo: probabile click accidentale, si scarta
        _clear_box()
        plt.draw()
        return
    box_corners.clear()
    box_corners.append((int(round(x0)), int(round(y0))))
    box_corners.append((int(round(x1)), int(round(y1))))
    plt.draw()


def main():
    if len(sys.argv) != 2:
        print("Uso: python pick_points.py /path/al/primo_frame.png")
        sys.exit(1)

    img = Image.open(sys.argv[1]).convert("RGB")
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(
        "SINISTRO = positivo, DESTRO = negativo, SHIFT+SINISTRO trascinato = box.\n"
        "Chiudi la finestra quando hai finito."
    )
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    plt.show()

    print("\n--- Copia questi flag nel comando sam2_video_propagate.py ---")

    # tutto su UNA riga sola, senza "\" di continuazione: su alcuni prompt
    # (es. cmd.exe di Windows, che usa "^" e non "\") il backslash a fine
    # riga non viene interpretato e rompe il comando.
    flags = []
    if len(box_corners) == 2:
        (x1, y1), (x2, y2) = box_corners
        left, right = sorted((x1, x2))
        bottom, top = sorted((y1, y2))
        flags.append(f"--box {left} {bottom} {right} {top}")
    for x, y in pos_points:
        flags.append(f"--point {x} {y}")
    for x, y in neg_points:
        flags.append(f"--neg_point {x} {y}")

    if flags:
        print(" ".join(flags))
    else:
        print("(nessun punto o box definito)")


if __name__ == "__main__":
    main()
