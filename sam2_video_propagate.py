"""
Prototipo "Livello 4": propagazione della maschera lungo la sequenza turntable
usando il video predictor nativo di SAM 2, invece di ri-cliccare su ognuna
delle N viste come fa oggi il wand interattivo di SLImTAG.

STATO: validato end-to-end (non solo scritto/letto contro l'API) su due
sequenze reali (berretto 16 frame, cuffie 41 frame con cavo aggrovigliato)
con i checkpoint SAM 2.1 tiny/small/large, su CPU in un ambiente sandbox
senza GPU. Su una macchina con GPU dedicata dovrebbe funzionare identico,
solo molto più veloce.

Uso tipico (CLI, box/punti scelti a mano):

    python sam2_video_propagate.py \
        --frames_dir /path/to/oggetto_XYZ/frames \
        --out_dir /path/to/oggetto_XYZ/masks \
        --point 400 380 \
        --checkpoint checkpoints/sam2.1_hiera_small.pt \
        --config configs/sam2.1/sam2.1_hiera_s.yaml \
        --full_loop

Per l'uso automatico (box/punti calcolati senza click manuale + controllo
qualità automatico), vedi auto_segment.py, che importa la funzione
propagate() definita qui sotto.

--point sono le coordinate (x,y) in pixel sul PRIMO frame della sequenza,
esattamente il click che oggi fate a mano con il wand di SLImTAG -- qui se
ne fa uno solo (o pochi, vedi --extra_anchor) invece che uno per ognuna
delle 72 viste. Per oggetti multi-parte (es. cuffie) usare più --point
su parti diverse e/o --box per racchiudere l'intero oggetto.

--full_loop va usato quando i frame coprono l'intera rotazione di 360°:
in quel caso lo script propaga in avanti dal primo anchor E all'indietro
dall'ultimo frame, e confronta le due stime a metà sequenza (IoU) come
controllo di qualità (drift della propagazione).

Output: le maschere finiscono in <out_dir>/color_mask/<stesso nome del
frame originale>.png (RGBA: colore pieno = oggetto, trasparente = sfondo;
colore di default = [158,31,99], lo stesso "dark magenta" di
default_mask_colors in config.toml di SLImTAG); lo stesso frame con la
maschera sovrapposta (per controllare a colpo d'occhio eventuali
imperfezioni: buchi, pezzi mancanti, sbordature) finisce in
<out_dir>/overlay/<stesso nome>.png; più <out_dir>/qc_report.json con le
statistiche di controllo qualità.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image


def prepare_jpeg_folder(src_dir, dst_dir, exts=(".png", ".jpg", ".jpeg")):
    """SAM2's video predictor expects a folder of JPEGs named 00000.jpg, 00001.jpg, ...
    (ordine = ordine della sequenza). Converte da qualunque formato di partenza.
    Ritorna la lista dei nomi file originali, nello stesso ordine (indice i <-> i:05d.jpg),
    cosi' le maschere in output possono essere salvate con lo stesso nome del frame di partenza."""
    os.makedirs(dst_dir, exist_ok=True)
    files = sorted(
        f for f in os.listdir(src_dir) if f.lower().endswith(exts)
    )
    if not files:
        raise RuntimeError(f"Nessuna immagine trovata in {src_dir}")
    for i, fname in enumerate(files):
        img = Image.open(os.path.join(src_dir, fname)).convert("RGB")
        img.save(os.path.join(dst_dir, f"{i:05d}.jpg"), quality=95)
    return files


# Colore convenzionale usato da SLImTAG per la prima maschera (config.toml,
# default_mask_colors: "dark magenta" [158, 31, 99]) invece del classico bianco/255.
MASK_COLOR_RGB = (158, 31, 99)


def mask_to_png(mask_bool, path, color_rgb=MASK_COLOR_RGB):
    """Salva la maschera come PNG RGBA: pixel oggetto = color_rgb pieno (alpha 255),
    sfondo = trasparente (alpha 0), coerente con la convenzione colore di SLImTAG."""
    h, w = mask_bool.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = color_rgb[0]
    rgba[..., 1] = color_rgb[1]
    rgba[..., 2] = color_rgb[2]
    rgba[..., 3] = mask_bool.astype(np.uint8) * 255
    Image.fromarray(rgba, mode="RGBA").save(path)


OVERLAY_COLOR_RGB = (0, 255, 0)  # verde acceso: buon contrasto sia su sfondo bianco che su oggetti scuri


def save_overlay_png(orig_frame_path, mask_bool, path, color_rgb=OVERLAY_COLOR_RGB, alpha=0.45):
    """Salva il frame originale con la maschera sovrapposta in trasparenza (colore acceso a
    pieno contrasto sul bordo), cosi' si vede a colpo d'occhio se la maschera copre bene tutto
    l'oggetto (buchi non voluti, pezzi mancanti, "sbordature" sullo sfondo) senza dover aprire
    la maschera RGBA da sola su sfondo trasparente."""
    img = Image.open(orig_frame_path).convert("RGB")
    arr = np.array(img).astype(np.float32)
    color = np.array(color_rgb, dtype=np.float32)
    arr[mask_bool] = arr[mask_bool] * (1 - alpha) + color * alpha

    # bordo della maschera a piena opacita', per vedere subito i contorni esatti
    from numpy.lib.stride_tricks import sliding_window_view
    pad = np.pad(mask_bool, 1, mode="constant", constant_values=False)
    eroded = np.all(sliding_window_view(pad, (3, 3)), axis=(-1, -2))
    boundary = mask_bool & ~eroded
    arr[boundary] = color

    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(path)


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 1.0


def propagate(frames_dir, out_dir, checkpoint, config,
              points=None, neg_points=None, box=None, extra_anchors=None,
              full_loop=False, area_qc_threshold=0.08, area_jump_qc_threshold=0.15,
              mask_color=MASK_COLOR_RGB, overlay_color=OVERLAY_COLOR_RGB, overlay_alpha=0.45,
              extra_report_fields=None, predictor=None,
              verbose=True):
    """Funzione riusabile (usata sia dalla CLI sotto, sia da auto_segment.py):
    dato un box/punti gia' decisi, fa girare SAM2 su tutta la sequenza e salva
    le maschere. Ritorna il dict di report (lo stesso salvato in qc_report.json).

    predictor: se gia' costruito altrove (es. per riusarlo su piu' oggetti nello
    stesso batch senza ricaricare i pesi ogni volta), passarlo qui; altrimenti
    viene costruito internamente da checkpoint/config."""
    points = points or []
    neg_points = neg_points or []
    extra_anchors = extra_anchors or []

    if not points and box is None:
        raise ValueError("serve almeno un punto o un box per indicare l'oggetto sul primo frame")

    import torch
    from sam2.build_sam import build_sam2_video_predictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and verbose:
        print("ATTENZIONE: nessuna GPU rilevata, sarà molto lento.", file=sys.stderr)

    jpeg_dir = os.path.join(out_dir, "_frames_jpeg")
    orig_filenames = prepare_jpeg_folder(frames_dir, jpeg_dir)
    n_frames = len(orig_filenames)
    if verbose:
        print(f"{n_frames} frame preparati in {jpeg_dir}")

    if predictor is None:
        predictor = build_sam2_video_predictor(config, checkpoint, device=device)

    state = predictor.init_state(video_path=jpeg_dir)
    predictor.reset_state(state)

    OBJ_ID = 1
    pts = list(points) + list(neg_points)
    labels = [1] * len(points) + [0] * len(neg_points)
    kwargs = {}
    if box is not None:
        x1, y1, x2, y2 = box
        kwargs["box"] = [[x1, y1], [x2, y2]]
    predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=0,
        obj_id=OBJ_ID,
        points=pts if pts else None,
        labels=labels if labels else None,
        **kwargs,
    )
    # eventuali anchor aggiuntivi su frame intermedi (self-occlusione)
    for frame_idx, x, y in extra_anchors:
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=frame_idx,
            obj_id=OBJ_ID,
            points=[[x, y]],
            labels=[1],
        )

    os.makedirs(out_dir, exist_ok=True)
    masks_by_frame = {}

    # propagazione in avanti dal primo anchor
    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
        mask = (mask_logits[0] > 0.0).cpu().numpy().squeeze()
        masks_by_frame[frame_idx] = mask

    closure_iou = None
    if full_loop and n_frames > 2:
        # propagazione all'indietro dall'ultimo frame, per chiudere il cerchio
        # e verificare che la stima converga con quella "in avanti" a metà sequenza
        state_back = predictor.init_state(video_path=jpeg_dir)
        predictor.reset_state(state_back)
        predictor.add_new_points_or_box(
            inference_state=state_back,
            frame_idx=n_frames - 1,
            obj_id=OBJ_ID,
            points=pts if pts else None,  # stessi punti/box usati per l'anchor in avanti
            labels=labels if labels else None,
            **kwargs,
        )
        masks_backward = {}
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(
            state_back, start_frame_idx=n_frames - 1, reverse=True
        ):
            mask = (mask_logits[0] > 0.0).cpu().numpy().squeeze()
            masks_backward[frame_idx] = mask

        mid = n_frames // 2
        if mid in masks_by_frame and mid in masks_backward:
            closure_iou = iou(masks_by_frame[mid], masks_backward[mid])
            if verbose:
                print(f"IoU di chiusura del cerchio a meta' sequenza (frame {mid}): {closure_iou:.3f}")

    # salvataggio maschere (nome = stesso nome del frame originale, sempre .png per l'alpha) + QC su area
    color_mask_dir = os.path.join(out_dir, "color_mask")
    overlay_dir = os.path.join(out_dir, "overlay")
    os.makedirs(color_mask_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    areas = []
    frame_names_ordered = []
    for frame_idx in sorted(masks_by_frame.keys()):
        mask = masks_by_frame[frame_idx]
        orig_name = orig_filenames[frame_idx]
        stem = os.path.splitext(orig_name)[0]
        out_name = f"{stem}.png"
        mask_to_png(mask, os.path.join(color_mask_dir, out_name), color_rgb=tuple(mask_color))
        save_overlay_png(os.path.join(frames_dir, orig_name), mask, os.path.join(overlay_dir, out_name),
                          color_rgb=tuple(overlay_color), alpha=overlay_alpha)
        areas.append(int(mask.sum()))
        frame_names_ordered.append(out_name)

    areas = np.array(areas)
    mean_area = areas.mean() if len(areas) else 0
    rel_dev = np.abs(areas - mean_area) / mean_area if mean_area > 0 else areas
    outliers_global = np.where(rel_dev > area_qc_threshold)[0].tolist()

    # variazione relativa frame-a-frame (derivata): un oggetto rigido che ruota con continuità
    # non dovrebbe avere salti bruschi da un frame al successivo, anche se l'area assoluta
    # cambia molto lungo l'intera sequenza (caso non convesso/articolato)
    if len(areas) > 1:
        frame_jump = np.abs(np.diff(areas)) / np.maximum(areas[:-1], 1)
        jump_outliers = (np.where(frame_jump > area_jump_qc_threshold)[0] + 1).tolist()
    else:
        frame_jump = np.array([])
        jump_outliers = []

    outliers = sorted(set(outliers_global) | set(jump_outliers))

    report = {
        "n_frames": n_frames,
        "frame_filenames": frame_names_ordered,
        "areas_px": areas.tolist(),
        "mean_area_px": float(mean_area),
        "rel_deviation_from_mean": rel_dev.round(4).tolist() if mean_area > 0 else [],
        "frame_to_frame_rel_jump": frame_jump.round(4).tolist(),
        "outlier_frames_idx_global_dev": outliers_global,
        "outlier_frames_idx_frame_jump": jump_outliers,
        "outlier_frames_idx": outliers,
        "outlier_filenames": [frame_names_ordered[i] for i in outliers],
        "loop_closure_iou_midpoint": closure_iou,
    }
    if extra_report_fields:
        report.update(extra_report_fields)

    with open(os.path.join(out_dir, "qc_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    if verbose:
        print(json.dumps(report, indent=2))
        print(f"\nFatto. Maschere salvate in {color_mask_dir}/, overlay di verifica in {overlay_dir}/ "
              f"(stesso nome dei frame originali).")
        print(f"{len(outliers)}/{n_frames} frame marcati per revisione manuale "
              f"(deviazione globale > {area_qc_threshold*100:.0f}% OPPURE salto frame-a-frame > {area_jump_qc_threshold*100:.0f}%): "
              f"{[frame_names_ordered[i] for i in outliers]}")

    shutil.rmtree(jpeg_dir, ignore_errors=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True, help="Cartella con le viste della sequenza (ordine alfabetico = ordine di rotazione)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--point", type=int, nargs=2, action="append", default=[], metavar=("X", "Y"), help="Click positivo sul primo frame (pixel). Ripetibile: per oggetti multi-parte (es. cuffie: archetto + padiglioni) un solo punto spesso seleziona solo il sotto-pezzo cliccato, servono più punti su parti diverse")
    ap.add_argument("--neg_point", type=int, nargs=2, action="append", default=[], metavar=("X", "Y"), help="Click negativo sul primo frame (pixel), ripetibile")
    ap.add_argument("--box", type=int, nargs=4, default=None, metavar=("X1", "Y1", "X2", "Y2"), help="Box che racchiude l'intero oggetto sul primo frame: il modo più robusto per disambiguare 'tutto l'oggetto' su forme complesse/multi-parte")
    ap.add_argument("--extra_anchor", type=int, nargs=3, action="append", default=[], metavar=("FRAME_IDX", "X", "Y"), help="Anchor aggiuntivo (utile se l'oggetto si auto-occlude durante la rotazione); ripetibile")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--full_loop", action="store_true", help="I frame coprono l'intera rotazione a 360°: propaga anche all'indietro e controlla la chiusura del cerchio")
    ap.add_argument("--area_qc_threshold", type=float, default=0.08, help="Soglia di deviazione relativa (rispetto alla media) dell'area per marcare un frame come outlier -- adatta a oggetti compatti/convessi la cui area proiettata resta ~costante")
    ap.add_argument("--area_jump_qc_threshold", type=float, default=0.15, help="Soglia di variazione RELATIVA FRAME-A-FRAME (derivata) dell'area: più robusta della precedente per oggetti non convessi/articolati (es. cuffie con archetto e cavo) la cui area proiettata varia molto ma con continuità lungo la rotazione")
    ap.add_argument("--mask_color", type=int, nargs=3, default=list(MASK_COLOR_RGB), metavar=("R", "G", "B"), help="Colore RGB della maschera in output (sfondo sempre trasparente); default = 'dark magenta' [158,31,99] come il primo colore di default_mask_colors in config.toml di SLImTAG")
    ap.add_argument("--overlay_color", type=int, nargs=3, default=list(OVERLAY_COLOR_RGB), metavar=("R", "G", "B"), help="Colore RGB usato per l'overlay di verifica (frame originale + maschera in trasparenza + bordo a piena opacita'), default verde acceso")
    ap.add_argument("--overlay_alpha", type=float, default=0.45, help="Opacita' del riempimento della maschera nell'overlay (il bordo resta sempre a piena opacita')")
    args = ap.parse_args()

    propagate(
        frames_dir=args.frames_dir, out_dir=args.out_dir,
        checkpoint=args.checkpoint, config=args.config,
        points=args.point, neg_points=args.neg_point, box=args.box, extra_anchors=args.extra_anchor,
        full_loop=args.full_loop, area_qc_threshold=args.area_qc_threshold,
        area_jump_qc_threshold=args.area_jump_qc_threshold, mask_color=tuple(args.mask_color),
        overlay_color=tuple(args.overlay_color), overlay_alpha=args.overlay_alpha,
    )


if __name__ == "__main__":
    main()
