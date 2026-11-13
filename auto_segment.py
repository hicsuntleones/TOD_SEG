"""
Pipeline automatica completa: box + punti calcolati senza click manuale
(auto_prompt.py) -> propagazione SAM2 (sam2_video_propagate.py) -> controllo
automatico di qualita' del prompt (confronto tra l'area della maschera SAM
sul frame 0 e la stima grezza indipendente, invece di ispezionare a occhio).

Uso su un singolo oggetto:

    python auto_segment.py \
        --frames_dir dataset/oggetto_01 \
        --out_dir output/oggetto_01 \
        --checkpoint checkpoints/sam2.1_hiera_small.pt \
        --config configs/sam2.1/sam2.1_hiera_s.yaml

Uso batch su un intero dataset, struttura a piu' livelli (oggetto/vista/...):

    dataset/
      hat/
        view0/color/*.png   <- immagini da segmentare (frame_000.png, ...)
        view1/color/*.png
      headphone/
        view0/color/*.png

    python auto_segment.py \
        --batch_root dataset/ \
        --checkpoint checkpoints/sam2.1_hiera_small.pt \
        --config configs/sam2.1/sam2.1_hiera_s.yaml

In modalita' batch lo script cerca RICORSIVAMENTE, dentro --batch_root, ogni
cartella chiamata "color" (nome configurabile con --color_dirname) a
qualunque profondita' -- una per ogni vista di ogni oggetto -- e la tratta
come una nuova sequenza da segmentare, qualunque sia la struttura sopra
(oggetto/vista, o anche piu'/meno livelli). Senza --out_root l'output di ogni
vista (color_mask/, overlay/, auto_prompt.json, qc_report.json) viene scritto
IN LOCO, cioe' dentro la stessa cartella-vista, come cartelle sorelle di
color/ (e di depth/ se presente) -- lo stesso posto dove secondo la vostra
struttura ci si aspetta color_mask e overlay. Se invece preferite non
scrivere dentro l'albero dati originale, passate --out_root: l'albero
oggetto/vista viene rispecchiato li' dentro.

I pesi SAM2 vengono caricati UNA volta sola e riusati per tutte le viste
(molto piu' veloce che rilanciare lo script per ognuna). Alla fine viene
scritto batch_summary.json/.csv (in --out_root se dato, altrimenti nella
radice di --batch_root) con, per ogni vista, l'esito e se serve revisione
manuale.

Logica del prompt automatico (vedi auto_prompt.py per i dettagli):
- box = bounding box del foreground grezzo sul frame 0 (chiuso morfologicamente
  per includere anche le parti sottili come i cavi)
- punti = uno per ogni componente "solida" del frame 0 (distance transform)
  + alcuni punti extra dai pixel sempre-foreground lungo tutta la rotazione
  (idea validata sperimentalmente: da sola non basta sempre, ma combinata col
  box non ha mai peggiorato i risultati nei test fatti)

Controllo qualita' automatico: rapporto = area maschera SAM (frame 0) /
area stima grezza indipendente (frame 0). Un rapporto molto minore di 1
(sotto --prompt_quality_threshold, default 0.5) segnala che SAM ha preso solo
una parte dell'oggetto -- il caso di fallimento osservato con un solo punto
su un oggetto multi-parte (es. cuffie) senza box: 9.645px di maschera contro
~100.000px attesi, rapporto ~0.1.
"""
import argparse
import csv
import glob
import json
import os

import cv2
import numpy as np

from auto_prompt import (
    box_touches_border,
    close_and_bbox,
    compute_object_points,
    crude_foreground_mask,
    filter_border_components,
    filter_components_by_box,
    sample_points_greedy,
)
from sam2_video_propagate import MASK_COLOR_RGB, OVERLAY_COLOR_RGB, propagate


def list_frame_paths(frames_dir, exts=(".png", ".jpg", ".jpeg")):
    return sorted(
        p for p in glob.glob(os.path.join(frames_dir, "*"))
        if p.lower().endswith(exts)
    )


def find_color_dirs(batch_root, color_dirname="color"):
    """Cerca ricorsivamente, sotto batch_root, ogni cartella chiamata
    color_dirname (default 'color') -- una per ogni vista di ogni oggetto
    nella struttura oggetto/vista/{color,color_mask,overlay,depth} --
    e la ritorna come cartella-sequenza da segmentare, qualunque sia la
    profondita' a cui si trova (funziona anche con piu' o meno livelli
    intermedi rispetto a oggetto/vista)."""
    found = []
    for dirpath, dirnames, _ in os.walk(batch_root):
        if os.path.basename(dirpath) == color_dirname:
            found.append(dirpath)
            dirnames[:] = []  # non scendere dentro una cartella "color" gia' trovata
    return sorted(found)


def _point_inside_box(pt, box, pad):
    if box is None:
        return True  # nessun box: non c'e' nulla su cui filtrare, si tiene il punto
    x, y = pt
    x1, y1, x2, y2 = box
    return (x1 - pad) <= x <= (x2 + pad) and (y1 - pad) <= y <= (y2 + pad)


def compute_auto_prompt(frames_dir, border_px=20, close_kernel=25, max_points=8,
                         min_component_area=200, use_intersection=True,
                         intersection_stride=1, min_point_separation=30,
                         point_box_pad=30, border_margin_frac=0.02, cluster_kernel=100,
                         zone_K=6, min_zone_frac=0.02, zone_min_separation=60):
    """Calcola box + punti per un oggetto, senza alcun input manuale.
    Ritorna anche le aree grezze, usate poi per il controllo qualita'.

    Il box (vedi close_and_bbox in auto_prompt.py) scarta gia' le componenti
    a contatto DAVVERO col bordo (per pixel effettivi, non bounding box) e
    raggruppa le rimanenti per vicinanza.

    I punti su frame 0 usano l'approccio ibrido compute_object_points (vedi
    auto_prompt.py): prima un punto per ogni zona di colore/illuminazione
    distinta dell'oggetto (es. coperchio chiaro vs corpo scuro vs cinghia
    nera), poi farthest-point sampling per il budget residuo, per raggiungere
    appendici sottili dello stesso colore del corpo (es. un cavo nero su un
    oggetto nero) che le zone di colore da sole non catturano. Opera
    direttamente su object_mask (gia' ripulito da bordo e clustering), non
    sulla maschera grezza -- niente filtro aggiuntivo di contenimento nel box
    e' quindi necessario per i punti di frame 0 (lo resta per quelli
    dell'intersezione multi-frame sotto, che lavora su una maschera diversa)."""
    paths = list_frame_paths(frames_dir)
    if not paths:
        raise RuntimeError(f"Nessuna immagine trovata in {frames_dir}")

    frame0 = cv2.imread(paths[0])
    clean0, bg_lab = crude_foreground_mask(frame0, border_px=border_px,
                                            min_component_area=min_component_area)
    box, _, object_mask = close_and_bbox(clean0, close_kernel=close_kernel, border_margin_frac=border_margin_frac,
                                          cluster_kernel=cluster_kernel)
    if object_mask is not None:
        # maschera grezza (non chiusa), ristretta al box: usata da compute_object_points
        # per la piccola riserva geometrica di fase 0 (n_fps_seed_phase), che deve
        # lavorare sulla forma NON alterata dalla chiusura morfologica -- altrimenti
        # un'appendice sottile e penzolante (es. il cavo delle cuffie) rischia di
        # perdere il proprio punto dedicato (vedi nota in compute_object_points).
        clean0_filtered = filter_components_by_box(clean0, box, pad=point_box_pad)
        points = compute_object_points(frame0, clean0_filtered, object_mask, max_points=max_points, K=zone_K,
                                        min_zone_frac=min_zone_frac,
                                        zone_min_separation=zone_min_separation)
    else:
        points = []
    # area di riferimento per il controllo qualita' (prompt_quality_ratio): quella
    # dell'oggetto ripulito dal rumore di bordo scartato, non della maschera grezza
    # intera -- altrimenti rumore di sfondo incluso nella stima grezza abbasserebbe
    # artificialmente il rapporto anche quando SAM ha segmentato l'oggetto per intero
    area0 = int((object_mask > 0).sum()) if object_mask is not None else int((clean0 > 0).sum())

    intersection_area = None
    if use_intersection and len(paths) >= 5:
        intersection = np.ones_like(clean0, dtype=bool)
        for p in paths[::intersection_stride]:
            img = cv2.imread(p)
            m, _ = crude_foreground_mask(img, bg_lab=bg_lab, min_component_area=min_component_area)
            intersection &= (m > 0)
        intersection_area = int(intersection.sum())
        if intersection_area > 0:
            intersection_no_border = filter_border_components(
                (intersection.astype(np.uint8) * 255), border_margin_frac=border_margin_frac
            )
            intersection_for_points = filter_components_by_box(
                intersection_no_border, box, pad=point_box_pad
            ) > 0
            inter_points = sample_points_greedy(intersection_for_points, n_points=max_points,
                                                 min_separation=max(20, close_kernel))
            inter_points = [p for p in inter_points if _point_inside_box(p, box, point_box_pad)]
            for p in inter_points:
                if all(abs(p[0] - q[0]) + abs(p[1] - q[1]) > min_point_separation for q in points):
                    points.append(p)

    return {
        "frame0_path": paths[0],
        "frame0_shape": [int(frame0.shape[0]), int(frame0.shape[1])],
        "n_frames": len(paths),
        "box": list(box) if box else None,
        "points": [list(p) for p in points],
        "crude_area_frame0_px": area0,
        "always_foreground_intersection_area_px": intersection_area,
        "box_touches_border": box_touches_border(box, frame0.shape),
    }


def save_prompt_overlay(frame0_path, box, points, out_path):
    img = cv2.imread(frame0_path)
    overlay = img.copy()
    if box:
        cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)
    for x, y in points:
        cv2.drawMarker(overlay, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
    cv2.imwrite(out_path, overlay)


def auto_segment_object(frames_dir, out_dir, checkpoint, config, predictor=None,
                         prompt_quality_threshold=0.5, full_loop=False,
                         area_qc_threshold=0.08, area_jump_qc_threshold=0.15,
                         mask_color=MASK_COLOR_RGB, overlay_color=OVERLAY_COLOR_RGB, overlay_alpha=0.45,
                         auto_prompt_kwargs=None, verbose=True):
    """Un oggetto, end-to-end: calcola box/punti automaticamente, fa girare SAM2,
    aggiunge al report il controllo di qualita' del prompt. Ritorna il report completo."""
    os.makedirs(out_dir, exist_ok=True)
    auto_prompt_kwargs = auto_prompt_kwargs or {}
    prompt = compute_auto_prompt(frames_dir, **auto_prompt_kwargs)

    with open(os.path.join(out_dir, "auto_prompt.json"), "w") as f:
        json.dump(prompt, f, indent=2)
    save_prompt_overlay(prompt["frame0_path"], prompt["box"], prompt["points"],
                         os.path.join(out_dir, "auto_prompt_overlay.png"))

    if verbose:
        print(f"[{os.path.basename(frames_dir)}] box={prompt['box']} "
              f"n_punti={len(prompt['points'])} area_grezza_frame0={prompt['crude_area_frame0_px']}px")

    report = propagate(
        frames_dir=frames_dir, out_dir=out_dir, checkpoint=checkpoint, config=config,
        points=prompt["points"], box=prompt["box"], full_loop=full_loop,
        area_qc_threshold=area_qc_threshold, area_jump_qc_threshold=area_jump_qc_threshold,
        mask_color=mask_color, overlay_color=overlay_color, overlay_alpha=overlay_alpha,
        predictor=predictor, verbose=verbose,
    )

    sam_area0 = report["areas_px"][0] if report["areas_px"] else 0
    crude_area0 = prompt["crude_area_frame0_px"]
    ratio = (sam_area0 / crude_area0) if crude_area0 > 0 else 0.0
    box_at_border = prompt.get("box_touches_border", False)
    review_reasons = []
    if ratio < prompt_quality_threshold:
        review_reasons.append("prompt_quality_ratio_basso")
    if box_at_border:
        review_reasons.append("box_tocca_il_bordo")
    needs_review = len(review_reasons) > 0

    report["auto_prompt"] = prompt
    report["prompt_quality_ratio"] = round(ratio, 4)
    report["box_touches_border"] = box_at_border
    report["needs_manual_review_prompt"] = needs_review
    report["review_reasons"] = review_reasons

    with open(os.path.join(out_dir, "qc_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    if verbose:
        esito = "OK" if not needs_review else f"DA RIVEDERE ({', '.join(review_reasons)})"
        print(f"[{os.path.basename(frames_dir)}] rapporto area SAM/stima_grezza = {ratio:.2f} -> {esito}")

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
    for color_dir in color_dirs:
        view_dir = os.path.dirname(color_dir)  # cartella-vista: color/, color_mask/, overlay/, depth/ sono qui, sorelle
        # nome leggibile per il riepilogo: percorso relativo della vista rispetto a batch_root (es. "hat/view0")
        rel = os.path.relpath(view_dir, batch_root)
        name = rel if rel != "." else os.path.basename(os.path.normpath(batch_root))

        frames_dir = color_dir
        out_dir = os.path.join(out_root, rel) if out_root else view_dir
        try:
            report = auto_segment_object(frames_dir, out_dir, checkpoint, config,
                                          predictor=predictor, **kwargs)
            summary.append({
                "object": name,
                "status": "ok",
                "n_frames": report["n_frames"],
                "prompt_quality_ratio": report["prompt_quality_ratio"],
                "box_touches_border": report["box_touches_border"],
                "needs_manual_review_prompt": report["needs_manual_review_prompt"],
                "review_reasons": report["review_reasons"],
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
        writer.writerow(["object", "status", "n_frames", "prompt_quality_ratio",
                          "box_touches_border", "needs_manual_review_prompt",
                          "review_reasons", "n_outlier_frames"])
        for row in summary:
            writer.writerow([
                row.get("object"), row.get("status"), row.get("n_frames", ""),
                row.get("prompt_quality_ratio", ""), row.get("box_touches_border", ""),
                row.get("needs_manual_review_prompt", ""),
                ";".join(row.get("review_reasons", []) or []),
                row.get("n_outlier_frames", ""),
            ])

    n_review = sum(1 for r in summary if r.get("needs_manual_review_prompt"))
    n_error = sum(1 for r in summary if r.get("status") == "error")
    print(f"\n=== Batch completato: {len(color_dirs)} viste ===")
    print(f"Da rivedere (prompt automatico probabilmente incompleto): {n_review}")
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
    ap.add_argument("--prompt_quality_threshold", type=float, default=0.5,
                     help="Sotto questo rapporto (area maschera SAM frame0 / area stima grezza) l'oggetto viene marcato needs_manual_review_prompt")
    ap.add_argument("--area_qc_threshold", type=float, default=0.08)
    ap.add_argument("--area_jump_qc_threshold", type=float, default=0.15)
    ap.add_argument("--mask_color", type=int, nargs=3, default=list(MASK_COLOR_RGB), metavar=("R", "G", "B"))
    ap.add_argument("--overlay_color", type=int, nargs=3, default=list(OVERLAY_COLOR_RGB), metavar=("R", "G", "B"),
                     help="Colore RGB dell'overlay di verifica (frame originale + maschera in trasparenza + bordo a piena opacita'), default verde acceso")
    ap.add_argument("--overlay_alpha", type=float, default=0.45,
                     help="Opacita' del riempimento della maschera nell'overlay (il bordo resta sempre a piena opacita')")
    ap.add_argument("--border_px", type=int, default=20)
    ap.add_argument("--close_kernel", type=int, default=25)
    ap.add_argument("--cluster_kernel", type=int, default=100,
                     help="Kernel di chiusura usato SOLO per decidere quali componenti raggruppare come stesso oggetto "
                          "(piu' grande di --close_kernel apposta): parti vicine (es. un pattern che spezza il rilevamento "
                          "in piu' isole) restano incluse nel box, rumore di sfondo isolato piu' lontano no")
    ap.add_argument("--max_points", type=int, default=6)
    ap.add_argument("--min_component_area", type=int, default=200)
    ap.add_argument("--no_intersection", action="store_true",
                     help="Disattiva il calcolo dei punti dall'intersezione multi-frame (piu' veloce, meno robusto)")
    args = ap.parse_args()

    auto_prompt_kwargs = dict(
        border_px=args.border_px, close_kernel=args.close_kernel, cluster_kernel=args.cluster_kernel,
        max_points=args.max_points, min_component_area=args.min_component_area,
        use_intersection=not args.no_intersection,
    )
    common_kwargs = dict(
        full_loop=args.full_loop, prompt_quality_threshold=args.prompt_quality_threshold,
        area_qc_threshold=args.area_qc_threshold, area_jump_qc_threshold=args.area_jump_qc_threshold,
        mask_color=tuple(args.mask_color), overlay_color=tuple(args.overlay_color),
        overlay_alpha=args.overlay_alpha, auto_prompt_kwargs=auto_prompt_kwargs,
    )

    if args.frames_dir:
        if not args.out_dir:
            ap.error("--out_dir e' richiesto con --frames_dir")
        auto_segment_object(args.frames_dir, args.out_dir, args.checkpoint, args.config, **common_kwargs)
    else:
        run_batch(args.batch_root, args.out_root, args.checkpoint, args.config,
                   color_dirname=args.color_dirname, **common_kwargs)


if __name__ == "__main__":
    main()
