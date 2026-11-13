"""
Pipeline end-to-end IDENTICA a auto_segment.py (stessa CLI: singolo oggetto
con --frames_dir, o intero dataset con --batch_root, stesso QC, stesso
output color_mask/, overlay/, qc_report.json, batch_summary in modalita'
batch) MA con box/punti scelti A MANO invece che calcolati automaticamente.

Per ogni oggetto si apre, sul primo frame, la STESSA interazione di
pick_points.py (che resta invariato -- qui se ne riusa solo la logica,
adattata a funzione richiamabile una volta per oggetto invece che script
standalone che stampa e esce):
    - click SINISTRO       = punto positivo (parte dell'oggetto)
    - click DESTRO         = punto negativo (sfondo/buco da escludere)
    - SHIFT + trascinamento SINISTRO = box che racchiude l'intero oggetto
      (il rettangolo giallo segue il mouse durante il trascinamento, il
      rilascio del tasto fissa l'angolo opposto)
Chiudere la finestra fa partire SAM2 su quell'oggetto (oggetto singolo) o
passa all'oggetto successivo (batch).

Uso su un singolo oggetto (una vista):

    python manual_segment.py \
        --frames_dir dataset/oggetto_01 \
        --out_dir output/oggetto_01 \
        --checkpoint checkpoints/sam2.1_hiera_small.pt \
        --config configs/sam2.1/sam2.1_hiera_s.yaml

Uso batch sull'intero dataset (stessa struttura oggetto/vista/color di
auto_segment.py --batch_root): si apre una finestra interattiva per OGNI
vista trovata, una dopo l'altra.

    python manual_segment.py \
        --batch_root dataset/ \
        --checkpoint checkpoints/sam2.1_hiera_small.pt \
        --config configs/sam2.1/sam2.1_hiera_s.yaml

Rispetto a auto_segment.py mancano solo i parametri che riguardano il
calcolo AUTOMATICO del prompt (--max_points, --close_kernel,
--cluster_kernel, --min_component_area, --no_intersection,
--prompt_quality_threshold): non si applicano qui, dato che box e punti li
sceglie chi guarda lo schermo. Per lo stesso motivo qc_report.json non
contiene prompt_quality_ratio / needs_manual_review_prompt (non c'e' una
stima automatica indipendente con cui confrontare il risultato di SAM) --
il file manual_prompt.json accanto registra invece esattamente cosa e'
stato scelto a mano, per riferimento/riproducibilita'.
"""
import argparse
import csv
import glob
import json
import os

import cv2

# IMPORTANTE (bug noto di PyTorch su Windows, versioni 2.9.0/2.9.1): se una
# libreria Qt (es. PyQt5/PyQt6, usata sotto come backend grafico di
# matplotlib per la finestra di selezione punti) viene importata PRIMA di
# torch, l'inizializzazione delle DLL di torch fallisce con
# "OSError: [WinError 1114] ... Error loading ...c10.dll". Importare torch
# per primo, prima ancora di toccare matplotlib, evita il problema.
# https://github.com/pytorch/pytorch/issues/166628
import torch  # noqa: F401  (import qui solo per fissare l'ordine di caricamento delle DLL su Windows)

# Forza un backend grafico INTERATTIVO invece di lasciare che matplotlib
# scelga da solo il backend non interattivo "Agg" (che puo' solo salvare
# immagini su file, non aprire una finestra con cui cliccare/trascinare).
#
# NOTA: matplotlib.use() chiamato PRIMA di importare pyplot non valida
# nulla -- si limita a impostare una stringa in rcParams senza importare
# davvero il modulo del backend, quindi il try/except attorno non
# intercetta mai un backend mancante (il fallimento vero, tipo
# "No module named 'tkinter'", salta fuori piu' avanti, non gestito).
# Si importa quindi prima pyplot con il backend di default (qualunque
# esso sia) e poi si passa ESPLICITAMENTE da un backend candidato
# all'altro con switch_backend(), che l'import lo tenta davvero -- vedi
# la stessa nota in pick_points.py.
import matplotlib.pyplot as plt

for _backend in ("TkAgg", "QtAgg", "Qt5Agg"):
    try:
        plt.switch_backend(_backend)
        break
    except ImportError:
        continue
else:
    raise RuntimeError(
        "Nessun backend grafico interattivo disponibile (serve Tkinter o PyQt5/PySide). "
        "Su Windows: se Python e' stato installato dal Microsoft Store, provate a "
        "installare Python da python.org (include Tkinter), oppure lanciate "
        "'pip install PyQt5' nello stesso ambiente e riprovate."
    )

from PIL import Image

from auto_segment import find_color_dirs
from sam2_video_propagate import MASK_COLOR_RGB, OVERLAY_COLOR_RGB, propagate

MIN_BOX_SIZE = 3  # px: sotto questa soglia un trascinamento e' un click accidentale, si scarta


def list_frame_paths(frames_dir, exts=(".png", ".jpg", ".jpeg")):
    return sorted(
        p for p in glob.glob(os.path.join(frames_dir, "*"))
        if p.lower().endswith(exts)
    )


def pick_points_and_box(image_path, title_extra=""):
    """Apre una finestra interattiva sul frame indicato e ritorna
    (pos_points, neg_points, box) -- box=[x1, y1, x2, y2] o None se non ne
    e' stato disegnato uno. Stessa interazione di pick_points.py, qui
    reimplementata come funzione richiamabile piu' volte (una per oggetto in
    modalita' batch) invece che script standalone a se stante."""
    pos_points = []
    neg_points = []
    box_corners = []  # 2 angoli quando il box e' completo, altrimenti vuoto
    box_patch = [None]
    drag_start = [None]

    def _clear_box():
        box_corners.clear()
        if box_patch[0] is not None:
            box_patch[0].remove()
            box_patch[0] = None

    def on_press(event):
        if event.xdata is None or event.ydata is None:
            return
        if event.button == 1 and event.key == "shift":
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

    img = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots()
    ax.imshow(img)
    title = "SINISTRO = positivo, DESTRO = negativo, SHIFT+SINISTRO trascinato = box.\nChiudi la finestra per continuare."
    if title_extra:
        title = f"{title_extra}\n{title}"
    ax.set_title(title)
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    plt.show()

    box = None
    if len(box_corners) == 2:
        (x1, y1), (x2, y2) = box_corners
        left, right = sorted((x1, x2))
        bottom, top = sorted((y1, y2))
        box = [left, bottom, right, top]
    return pos_points, neg_points, box


def save_prompt_overlay(frame0_path, box, points, neg_points, out_path):
    img = cv2.imread(frame0_path)
    overlay = img.copy()
    if box:
        cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)
    for x, y in points:
        cv2.drawMarker(overlay, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
    for x, y in neg_points:
        cv2.drawMarker(overlay, (x, y), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 20, 2)
    cv2.imwrite(out_path, overlay)


def manual_segment_object(frames_dir, out_dir, checkpoint, config, predictor=None,
                           full_loop=False, area_qc_threshold=0.08, area_jump_qc_threshold=0.15,
                           mask_color=MASK_COLOR_RGB, overlay_color=OVERLAY_COLOR_RGB, overlay_alpha=0.45,
                           title_extra="", verbose=True):
    """Un oggetto, end-to-end: box/punti scelti a mano sul frame 0 (finestra
    interattiva), poi SAM2 su tutta la sequenza -- stesso output/QC di
    auto_segment_object in auto_segment.py, senza calcolo automatico del
    prompt. Ritorna il report completo (uguale a qc_report.json)."""
    os.makedirs(out_dir, exist_ok=True)
    paths = list_frame_paths(frames_dir)
    if not paths:
        raise RuntimeError(f"Nessuna immagine trovata in {frames_dir}")
    frame0_path = paths[0]

    pos_points, neg_points, box = pick_points_and_box(frame0_path, title_extra=title_extra)
    if not pos_points and not neg_points and box is None:
        raise ValueError(
            f"[{os.path.basename(frames_dir)}] nessun punto/box scelto -- "
            "serve almeno un punto positivo o un box per indicare l'oggetto"
        )

    prompt = {
        "frame0_path": frame0_path,
        "n_frames": len(paths),
        "box": box,
        "points": [list(p) for p in pos_points],
        "neg_points": [list(p) for p in neg_points],
    }
    with open(os.path.join(out_dir, "manual_prompt.json"), "w") as f:
        json.dump(prompt, f, indent=2)
    save_prompt_overlay(frame0_path, box, pos_points, neg_points,
                         os.path.join(out_dir, "manual_prompt_overlay.png"))

    if verbose:
        print(f"[{os.path.basename(frames_dir)}] box={box} "
              f"n_punti_pos={len(pos_points)} n_punti_neg={len(neg_points)}")

    report = propagate(
        frames_dir=frames_dir, out_dir=out_dir, checkpoint=checkpoint, config=config,
        points=pos_points, neg_points=neg_points, box=box, full_loop=full_loop,
        area_qc_threshold=area_qc_threshold, area_jump_qc_threshold=area_jump_qc_threshold,
        mask_color=mask_color, overlay_color=overlay_color, overlay_alpha=overlay_alpha,
        predictor=predictor, verbose=verbose,
    )

    report["manual_prompt"] = prompt
    with open(os.path.join(out_dir, "qc_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    return report


def run_batch(batch_root, out_root, checkpoint, config, color_dirname="color", **kwargs):
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Carico SAM2 una sola volta (device={device})...")
    predictor = build_sam2_video_predictor(config, checkpoint, device=device)

    color_dirs = find_color_dirs(batch_root, color_dirname=color_dirname)
    if not color_dirs:
        raise RuntimeError(
            f"Nessuna cartella '{color_dirname}' trovata sotto {batch_root} "
            f"(controllate --color_dirname se il nome e' diverso)"
        )

    summary = []
    for i, color_dir in enumerate(color_dirs, 1):
        view_dir = os.path.dirname(color_dir)  # cartella-vista: color/, color_mask/, overlay/ sono qui, sorelle
        rel = os.path.relpath(view_dir, batch_root)
        name = rel if rel != "." else os.path.basename(os.path.normpath(batch_root))

        frames_dir = color_dir
        out_dir = os.path.join(out_root, rel) if out_root else view_dir
        try:
            report = manual_segment_object(
                frames_dir, out_dir, checkpoint, config, predictor=predictor,
                title_extra=f"[{i}/{len(color_dirs)}] {name}", **kwargs
            )
            summary.append({
                "object": name,
                "status": "ok",
                "n_frames": report["n_frames"],
                "n_outlier_frames": len(report["outlier_frames_idx"]),
                "outlier_filenames": report["outlier_filenames"],
            })
        except Exception as e:
            print(f"[{name}] ERRORE: {e}")
            summary.append({"object": name, "status": "error", "error": str(e)})

    summary_dir = out_root if out_root else batch_root
    os.makedirs(summary_dir, exist_ok=True)
    with open(os.path.join(summary_dir, "batch_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    csv_path = os.path.join(summary_dir, "batch_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["object", "status", "n_frames", "n_outlier_frames"])
        for row in summary:
            writer.writerow([
                row.get("object"), row.get("status"),
                row.get("n_frames", ""), row.get("n_outlier_frames", ""),
            ])

    n_error = sum(1 for r in summary if r.get("status") == "error")
    print(f"\n=== Batch manuale completato: {len(color_dirs)} viste ===")
    print(f"Errori: {n_error}")
    print(f"Riepilogo: {csv_path}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--frames_dir", help="Un singolo oggetto: cartella con le sue viste")
    mode.add_argument("--batch_root", help="Batch: cartella radice del dataset (viene cercata ricorsivamente ogni cartella 'color')")
    ap.add_argument("--out_dir", help="Richiesto con --frames_dir")
    ap.add_argument("--out_root", help="Solo con --batch_root: se dato, rispecchia li' l'albero oggetto/vista invece di scrivere in loco dentro --batch_root")
    ap.add_argument("--color_dirname", default="color",
                     help="Solo con --batch_root: nome della cartella con le immagini da segmentare dentro ogni vista (default 'color')")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--full_loop", action="store_true")
    ap.add_argument("--area_qc_threshold", type=float, default=0.08)
    ap.add_argument("--area_jump_qc_threshold", type=float, default=0.15)
    ap.add_argument("--mask_color", type=int, nargs=3, default=list(MASK_COLOR_RGB), metavar=("R", "G", "B"))
    ap.add_argument("--overlay_color", type=int, nargs=3, default=list(OVERLAY_COLOR_RGB), metavar=("R", "G", "B"),
                     help="Colore RGB dell'overlay di verifica (frame originale + maschera in trasparenza + bordo a piena opacita'), default verde acceso")
    ap.add_argument("--overlay_alpha", type=float, default=0.45,
                     help="Opacita' del riempimento della maschera nell'overlay (il bordo resta sempre a piena opacita')")
    args = ap.parse_args()

    common_kwargs = dict(
        full_loop=args.full_loop, area_qc_threshold=args.area_qc_threshold,
        area_jump_qc_threshold=args.area_jump_qc_threshold,
        mask_color=tuple(args.mask_color), overlay_color=tuple(args.overlay_color),
        overlay_alpha=args.overlay_alpha,
    )

    if args.frames_dir:
        if not args.out_dir:
            ap.error("--out_dir e' richiesto con --frames_dir")
        manual_segment_object(args.frames_dir, args.out_dir, args.checkpoint, args.config, **common_kwargs)
    else:
        run_batch(args.batch_root, args.out_root, args.checkpoint, args.config,
                   color_dirname=args.color_dirname, **common_kwargs)


if __name__ == "__main__":
    main()
