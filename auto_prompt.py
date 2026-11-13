"""
Stima automatica di box + punti da dare a SAM2, senza click manuale.

Idea:
1. Lo sfondo è affidabilmente ai BORDI del frame (l'oggetto è sempre centrato
   sulla pedana) -> si campiona il colore di sfondo dai bordi, non serve
   nessun input manuale.
2. Foreground grezzo = pixel lontani (in Lab) da quel colore, su frame 0
   (e opzionalmente sull'intersezione di più frame della rotazione, come
   proposto: i pixel sempre-foreground nonostante la rotazione).
3. Chiusura morfologica per "cucire" insieme le parti sottili (fili, cerniere)
   che altrimenti risultano componenti separate, poi si sceglie SOLO la
   componente connessa piu' grande tra quelle che NON toccano il bordo
   dell'immagine (un oggetto vero, per come e' fatta l'acquisizione, non
   arriva mai al bordo del frame: sempre centrato sulla pedana, che se
   l'oggetto e' grande puo' essere occlusa sopra ma e' sempre visibile
   almeno in basso) -> bounding box di quella sola componente = box da dare
   a SAM. Scartare le componenti a contatto col bordo invece di includerle
   nell'unione e' fondamentale per gli oggetti a basso contrasto col fondo
   (es. bianco su bianco): li' la soglia di Otsu cattura anche variazioni di
   trama/ombra del tessuto, che pero' quasi sempre toccano il bordo
   dell'immagine (sono "sfondo che continua fuori dal frame"), mentre
   l'oggetto vero no.
4. Un punto per ogni componente "solida" pre-chiusura (via distance transform,
   punto più interno = più lontano da qualunque bordo) -> uno o più --point,
   automaticamente uno per ogni "pezzo" visivamente separato (utile proprio
   per oggetti multi-parte come le cuffie, senza bisogno di specificare a
   mano quante parti ci sono).
5. Salva anche l'area/bbox del foreground grezzo: serve come riferimento
   indipendente per il controllo di qualità automatico dopo la propagazione
   SAM (vedi check_prompt_quality.py) -- risponde a "come faccio a sapere se
   è stato un buon punto" confrontando il risultato di SAM con questa stima
   indipendente, invece di ispezionare a occhio ogni maschera.
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np


def sample_border_background(img_lab, border_px=20):
    h, w = img_lab.shape[:2]
    top = img_lab[:border_px, :, :].reshape(-1, 3)
    bottom = img_lab[-border_px:, :, :].reshape(-1, 3)
    left = img_lab[:, :border_px, :].reshape(-1, 3)
    right = img_lab[:, -border_px:, :].reshape(-1, 3)
    border = np.concatenate([top, bottom, left, right], axis=0)
    return np.median(border, axis=0)


def crude_foreground_mask(img_bgr, bg_lab=None, border_px=20, min_component_area=60):
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    if bg_lab is None:
        bg_lab = sample_border_background(img_lab, border_px)
    diff = np.linalg.norm(img_lab - bg_lab, axis=2)
    diff_u8 = np.clip(diff / (diff.max() + 1e-6) * 255, 0, 255).astype(np.uint8)
    diff_u8 = cv2.GaussianBlur(diff_u8, (5, 5), 0)
    _, mask = cv2.threshold(diff_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # rimuove rumore isolato piccolo (puntini) mantenendo TUTTE le componenti
    # abbastanza grandi (non solo la piu' grande: un cavo puo' spezzarsi in
    # piu' pezzi separati dal resto)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_component_area:
            clean[labels == lbl] = 255
    return clean, bg_lab


def sample_points_greedy(mask_bool, n_points=4, min_separation=50):
    """Campiona piu' punti interni a una maschera binaria, con soppressione
    greedy: prende il punto piu' 'profondo' (distance transform massima),
    poi azzera un intorno attorno ad esso e ripete. Usato sia sulla maschera
    grezza del frame 0 sia sull'intersezione multi-frame (pixel
    sempre-foreground nonostante la rotazione, come suggerito) per fornire
    a SAM piu' punti campione invece di uno solo -- con SAM1 piu' punti
    spesso danno un risultato piu' completo, l'API di SAM2 accetta allo
    stesso modo una lista di punti."""
    work = mask_bool.astype(np.uint8).copy()
    points = []
    for _ in range(n_points):
        dist = cv2.distanceTransform(work, cv2.DIST_L2, 5)
        _, max_val, _, max_loc = cv2.minMaxLoc(dist)
        if max_val < 3:  # non resta piu' nulla di abbastanza 'interno'
            break
        points.append(list(max_loc))
        cv2.circle(work, max_loc, min_separation, 0, thickness=-1)
    return points


def sample_points_spread(mask_bool, n_points, seed_point=None, seed_points=None,
                          min_interior=3, min_separation=20):
    """Campiona punti spalmati sull'INTERA estensione geometrica di una
    maschera (farthest-point sampling), invece che nelle sole zone piu'
    'grasse' della sagoma come farebbe la sola distance transform greedy
    (sample_points_greedy): quella, ad ogni iterazione, prende sempre il
    punto piu' interno rimasto, quindi su un oggetto con una zona molto piu'
    massiccia di un'altra (es. il coperchio panciuto di una custodia rigida
    contro la parte frontale piu' sottile) continua a piazzare punti nella
    stessa zona finche' non e' "consumata", invece di distribuirli sull'intero
    oggetto come farebbe una persona a mano.

    Dopo il/i punto/i di partenza (seed_points, o il singolo seed_point per
    compatibilita', o il piu' interno in assoluto se nessuno dei due e'
    dato), ogni punto successivo e' quello -- tra tutti i pixel abbastanza
    "interni" (dist >= min_interior, per non cadere su rumore di bordo) --
    che massimizza la distanza MINIMA da TUTTI i punti gia' scelti (semi
    inclusi): cade quindi naturalmente il piu' lontano possibile da cio' che
    e' gia' coperto (es. la parte opposta dell'oggetto, o un'appendice
    sottile come un cavo non catturata da un raggruppamento per colore),
    garantendo copertura reale invece che ripetuta concentrazione in una
    sola area. seed_points (plurale) e' usato nell'approccio ibrido
    zone-di-colore + FPS: i punti gia' scelti per zona di colore diventano i
    semi da cui allontanarsi il piu' possibile per il resto del budget."""
    mask_u8 = mask_bool.astype(np.uint8)
    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    ys, xs = np.where(dist >= min_interior)
    if len(xs) == 0:
        return []
    coords = np.stack([xs, ys], axis=1).astype(np.float32)

    if seed_points:
        seeds = [tuple(p) for p in seed_points]
    elif seed_point is not None:
        seeds = [tuple(seed_point)]
    else:
        _, max_val, _, max_loc = cv2.minMaxLoc(dist)
        if max_val < min_interior:
            return []
        seeds = [max_loc]
    has_external_seed = bool(seed_points) or seed_point is not None

    points = list(seeds)
    min_dist_to_chosen = np.min(
        [np.linalg.norm(coords - np.array(s, dtype=np.float32), axis=1) for s in seeds], axis=0
    )
    new_points = []
    target_total = n_points + len(seeds) if has_external_seed else n_points
    while len(points) < target_total:
        idx = int(np.argmax(min_dist_to_chosen))
        if min_dist_to_chosen[idx] < min_separation:
            break
        cand = (int(coords[idx, 0]), int(coords[idx, 1]))
        points.append(cand)
        new_points.append(cand)
        new_dist = np.linalg.norm(coords - np.array(cand, dtype=np.float32), axis=1)
        min_dist_to_chosen = np.minimum(min_dist_to_chosen, new_dist)
    return new_points if has_external_seed else points


def find_anchor_points(clean_mask, max_points=6, min_component_area=200, min_separation=40):
    """Punti campione dalla maschera grezza, in due fasi.

    Fase 1 (comportamento originale, invariato): un punto per ogni componente
    'solida', nel pixel piu' interno (distance transform) -- robusto anche su
    parti sottili come l'archetto delle cuffie, perche' lavora componente per
    componente invece che sull'intera maschera unita, e garantisce che una
    parte piccola non venga mai "affamata" di punti da una piu' grande.

    Fase 2 (nuova): se avanza budget (max_points non ancora raggiunto), si
    aggiungono ULTERIORI punti dentro le componenti piu' grandi, con
    farthest-point sampling (sample_points_spread) invece della sola distance
    transform -- necessario per un oggetto che e' un unico blob connesso ma
    ha comunque zone visivamente distinte (es. coperchio panciuto e corpo
    piu' sottile di una custodia rigida): la sola distance transform, presa
    ripetutamente, resta quasi sempre nella zona piu' "spessa" della sagoma
    (es. sempre sul coperchio, mai sulla parte inferiore piu' sottile),
    mentre un umano che clicca a mano ne metterebbe uno per zona visiva. Il
    farthest-point sampling invece, dopo il primo punto (il piu' interno),
    sceglie ad ogni passo il punto piu' lontano da quelli gia' presi,
    garantendo copertura reale di tutta l'estensione dell'oggetto. Senza
    questa fase i punti finivano tutti ammassati in una sola area
    dell'oggetto quando questo non si spezzava in piu' componenti separate."""
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)
    candidates = []
    for lbl in range(1, n_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < min_component_area:
            continue
        candidates.append((area, lbl))
    # ordina per area decrescente (le parti piu' grandi/affidabili prima)
    candidates.sort(key=lambda c: -c[0])

    points = []
    comp_first_point = {}
    for area, lbl in candidates:
        if len(points) >= max_points:
            break
        comp = (labels == lbl).astype(np.uint8)
        dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
        _, max_val, _, max_loc = cv2.minMaxLoc(dist)
        x, y = max_loc
        if any(abs(x - px) + abs(y - py) < min_separation for px, py in points):
            continue
        points.append((x, y))
        comp_first_point[lbl] = (x, y)

    for area, lbl in candidates:
        if len(points) >= max_points:
            break
        if lbl not in comp_first_point:
            continue
        comp_bool = (labels == lbl)
        remaining_budget = max_points - len(points)
        extra = sample_points_spread(comp_bool, n_points=remaining_budget,
                                      seed_point=comp_first_point[lbl],
                                      min_separation=min_separation)
        for x, y in extra:
            if any(abs(x - px) + abs(y - py) < min_separation for px, py in points):
                continue
            points.append((x, y))
            if len(points) >= max_points:
                break
    return points


def zone_color_points(img_bgr, object_mask, K=6, min_zone_frac=0.02):
    """Un punto per ogni zona di colore/illuminazione visivamente distinta
    dell'oggetto (es. coperchio chiaro vs corpo scuro vs cinghia nera vs
    fibbia bianca di uno zaino), invece che per componente 'grezza' o per
    estremo geometrico: raggruppa i pixel dell'oggetto in K cluster di
    colore (k-means in Lab, la stessa scala usata altrove nel modulo), poi
    per ogni cluster prende le sue componenti connesse abbastanza grandi
    (>= min_zone_frac della superficie totale dell'oggetto, cosi' la soglia
    si adatta a oggetti piccoli o grandi) e ne calcola il punto piu' interno
    (distance transform). Trova bene superfici ampie e coerenti, ma non
    elementi sottili che condividono il colore col resto dell'oggetto (es.
    un cavo nero attaccato a un corpo nero): quelli restano compito di
    sample_points_spread (vedi compute_object_points).

    Ritorna una lista di ((x,y), area) ordinata per area decrescente --
    l'area serve a chi chiama per decidere quali zone tenere se il numero
    di zone valide supera il budget di punti disponibile."""
    ys, xs = np.where(object_mask > 0)
    total_area = len(xs)
    if total_area == 0:
        return []
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    samples = lab[ys, xs]
    K = min(K, total_area)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels_km, _ = cv2.kmeans(samples, K, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    labels_km = labels_km.flatten()
    label_map = np.full(img_bgr.shape[:2], -1, dtype=np.int32)
    label_map[ys, xs] = labels_km

    min_zone_area = max(300, int(min_zone_frac * total_area))
    zone_points = []
    for k in range(K):
        zone_mask = (label_map == k).astype(np.uint8)
        n, comp_labels, stats, _ = cv2.connectedComponentsWithStats(zone_mask, connectivity=8)
        for lbl in range(1, n):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area < min_zone_area:
                continue
            comp = (comp_labels == lbl).astype(np.uint8)
            dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
            _, _, _, max_loc = cv2.minMaxLoc(dist)
            zone_points.append((max_loc, area))
    zone_points.sort(key=lambda p: -p[1])
    return zone_points


def compute_object_points(img_bgr, clean_mask, object_mask, max_points=8, K=6, min_zone_frac=0.02,
                           zone_min_separation=60, min_separation=40, n_fps_seed_phase=2):
    """Approccio ibrido, in TRE fasi, per i punti campione su un oggetto gia'
    localizzato (object_mask, da close_and_bbox):

    Fase 0 (riserva geometrica minima, NON seminata): un piccolo numero
    (n_fps_seed_phase) di punti da farthest-point sampling puro sulla
    maschera GREZZA non chiusa (clean_mask), prima ancora delle zone di
    colore. Garantisce che un'appendice sottile/penzolante (es. il cavo di
    un paio di cuffie) riceva quasi sempre un punto anche quando la sua
    zona di colore e' troppo frammentata per superare la soglia di
    zone_color_points (un cavo intrecciato su se stesso, dello stesso nero
    di tutto il resto, si spezza in tanti pezzettini piccoli nel
    clustering colore). Si usa la maschera NON chiusa apposta: la chiusura
    morfologica (close_and_bbox) puo' alterare la larghezza locale di una
    parte sottile e spostare il punto di massima distanza lontano dalla
    sua vera posizione fisica. Il budget e' tenuto volutamente piccolo
    (default 2) per non sottrarre troppo spazio alla fase 1 sottostante,
    che resta la fase principale per oggetti con zone ad alto contrasto
    (es. il corpo viola chiaro/viola scuro/fibbia bianca di uno zaino).

    Fase 1 (zone di colore): un punto al centro di ogni zona cromatica
    significativa (zone_color_points), non gia' coperta da un punto di
    fase 0 -- il comportamento che ha sostituito il solo FPS, che si
    accumulava sugli estremi della sagoma invece che sulle zone visive
    (es. sempre il coperchio di una custodia, mai la fibbia). Le zone
    vengono dedotte per vicinanza (zone_min_separation, piu' ampio del
    min_separation usato per FPS): un oggetto a righe o pattern ripetuto
    (es. un berretto a righe) produce altrimenti una zona per ogni singola
    riga, quasi tutte ravvicinate e ridondanti.

    Fase 2 (completamento FPS): il budget eventualmente ancora libero
    viene riempito con farthest-point sampling sull'intera object_mask,
    seminato da TUTTI i punti gia' scelti nelle fasi precedenti, per
    raggiungere le estremita' geometriche non coperte da nessuna delle due
    fasi sopra."""
    points = list(sample_points_spread(clean_mask > 0, n_points=min(n_fps_seed_phase, max_points),
                                        min_separation=min_separation))

    zone_pts = zone_color_points(img_bgr, object_mask, K=K, min_zone_frac=min_zone_frac)
    for (x, y), area in zone_pts:
        if len(points) >= max_points:
            break
        if any(abs(x - px) + abs(y - py) < zone_min_separation for px, py in points):
            continue
        points.append((x, y))

    if len(points) < max_points:
        remaining = max_points - len(points)
        extra = sample_points_spread(object_mask > 0, n_points=remaining,
                                      seed_points=points if points else None,
                                      min_separation=min_separation)
        for x, y in extra:
            if any(abs(x - px) + abs(y - py) < min_separation for px, py in points):
                continue
            points.append((x, y))
            if len(points) >= max_points:
                break
    return points


def _component_border_contact(comp_mask_bool, W, H, margin):
    """Quanto UNA componente e' realmente 'fatta di bordo', in pixel effettivi
    (non nel bounding box: un oggetto vero puo' avere una punta/ombra che
    sfiora il bordo per una manciata di pixel senza che l'oggetto stesso sia
    rumore di sfondo -- osservato con un borsone il cui bounding box toccava
    x=0 per sole 55 pixel su 97.900, un'ombra incidentale, non il vero
    contorno dell'oggetto)."""
    ys, xs = np.where(comp_mask_bool)
    if len(xs) == 0:
        return 0, 0.0
    in_band = (xs < margin) | (xs >= W - margin) | (ys < margin) | (ys >= H - margin)
    n_border = int(in_band.sum())
    return n_border, n_border / len(xs)


def _component_touches_border(comp_mask_bool, W, H, margin, min_border_px=100, min_border_frac=0.5):
    n_border, frac = _component_border_contact(comp_mask_bool, W, H, margin)
    return (n_border >= min_border_px) or (frac >= min_border_frac)


def box_touches_border(box, frame_shape, margin_frac=0.01, span_frac_threshold=0.85):
    """Controllo di sanita' esplicito, indipendente dal resto della pipeline:
    nel vostro setup nessun oggetto arriva mai al bordo dell'immagine, quindi
    un box "esploso" -- che tocca un bordo E occupa quasi tutta quella
    dimensione del frame (default: 85%) -- e' un segno affidabile che la
    detection e' fallita, a prescindere da qualunque altro controllo di
    qualita'. Richiedere ANCHE lo span, non solo il contatto col bordo, evita
    di segnalare come fallito un box altrimenti corretto e di dimensioni
    normali che tocca il bordo solo per una manciata di pixel incidentali
    (stessa causa del caso sopra)."""
    if box is None:
        return True
    H, W = frame_shape[:2]
    margin = max(1, int(margin_frac * min(H, W)))
    x1, y1, x2, y2 = box
    width_frac = (x2 - x1) / W
    height_frac = (y2 - y1) / H
    x_exploded = (x1 <= margin or x2 >= (W - 1 - margin)) and width_frac >= span_frac_threshold
    y_exploded = (y1 <= margin or y2 >= (H - 1 - margin)) and height_frac >= span_frac_threshold
    return x_exploded or y_exploded


def filter_border_components(clean_mask, border_margin_frac=0.02, min_border_touch_px=100,
                              min_border_touch_frac=0.5):
    """Rimuove dalla maschera le componenti che toccano davvero il bordo (vedi
    _component_touches_border). Riusata sia da close_and_bbox per il box, sia
    a monte di find_anchor_points per i punti: senza questo, un punto
    campione potrebbe finire su una componente di rumore anche quando il box
    stesso non tocca il bordo (o, peggio, filtrare i punti *dopo* solo in
    base al box non basta se anche il box finisce per toccare il bordo).
    Se TUTTE le componenti risultano a contatto col bordo (caso limite), la
    maschera originale viene ritornata invariata -- non c'e' nulla di sicuro
    da rimuovere, e box_touches_border() a valle intercetta comunque il
    problema sul box finale."""
    H, W = clean_mask.shape
    margin = max(1, int(border_margin_frac * min(H, W)))
    n_labels, labels, _, _ = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)
    filtered = np.zeros_like(clean_mask)
    any_kept = False
    for lbl in range(1, n_labels):
        comp = (labels == lbl)
        if not _component_touches_border(comp, W, H, margin, min_border_touch_px, min_border_touch_frac):
            filtered[comp] = 255
            any_kept = True
    return filtered if any_kept else clean_mask


def filter_components_by_box(clean_mask, box, pad):
    """Tiene solo le componenti il cui centroide cade dentro il box (con
    margine pad) gia' calcolato per l'oggetto. Usata PRIMA di scegliere i
    punti campione (find_anchor_points), non solo dopo: senza questo, rumore
    lontano dall'oggetto (es. macchioline sulla trama dello sfondo) viene
    comunque contato come 'componente candidata' e consuma budget di
    max_points prima ancora di arrivare alle parti vere dell'oggetto -- i
    punti di rumore vengono si' scartati alla fine perche' fuori dal box, ma
    a quel punto il budget e' gia' esaurito e non resta margine per
    distribuire piu' punti sulle zone vere (es. sia sul coperchio sia sulla
    parte inferiore di una custodia). Se il box e' None o nessuna componente
    ricade al suo interno, la maschera originale viene ritornata invariata."""
    if box is None:
        return clean_mask
    H, W = clean_mask.shape
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = x1 - pad, y1 - pad, x2 + pad, y2 + pad
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)
    filtered = np.zeros_like(clean_mask)
    any_kept = False
    for lbl in range(1, n_labels):
        cx, cy = centroids[lbl]
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            filtered[labels == lbl] = 255
            any_kept = True
    return filtered if any_kept else clean_mask


def close_and_bbox(clean_mask, close_kernel=25, prefer_central=True, border_margin_frac=0.02,
                    cluster_kernel=100, min_border_touch_px=100, min_border_touch_frac=0.5):
    """Chiusura morfologica + bounding box.

    prefer_central=True (default): prima di chiudere, scarta le componenti
    della maschera grezza che toccano DAVVERO il bordo dell'immagine -- non
    per bounding box (un oggetto vero puo' avere un'ombra o una punta che
    sfiora il bordo per pochi pixel senza essere rumore), ma per numero/
    frazione di pixel EFFETTIVI nella fascia di margine (min_border_touch_px,
    min_border_touch_frac): vedi _component_touches_border. Poi, invece di
    tenere SOLO la componente
    connessa piu' grande (rischia di scartare parti vere ma fisicamente
    separate dal corpo principale -- es. un motivo/pattern sulla stoffa che
    spezza il rilevamento in piu' isolette, come le foglioline su una
    pochette a fiori), si raggruppano le componenti superstiti per VICINANZA:
    una chiusura con un kernel piu' grande (cluster_kernel, default 100px)
    decide quali componenti sono abbastanza vicine da poter essere lo stesso
    oggetto fisico; si tiene il gruppo con l'area totale maggiore (il vero
    oggetto, non rumore sparso) e il box finale e' il bbox di TUTTI i pixel
    di quel gruppo -- rumore di sfondo isolato a decine/centinaia di pixel di
    distanza resta escluso, ma parti vicine della stessa cosa restano incluse.
    Se nessuna componente sopravvive al filtro bordo (caso limite/detection
    completamente fallita), si ricade sul comportamento precedente (bbox
    dell'unione di tutto): il controllo box_touches_border() a valle
    intercetta comunque il problema.

    Ritorna (box, closed, object_mask): closed e' la maschera chiusa PRIMA
    del filtro per cluster (utile per debug), object_mask e' la maschera
    SOLO dell'oggetto scelto (None se non e' stato possibile calcolare un
    box) -- usata altrove per stimare un'area di riferimento pulita, senza
    il rumore di sfondo scartato."""
    mask_to_close = clean_mask
    if prefer_central:
        mask_to_close = filter_border_components(clean_mask, border_margin_frac,
                                                   min_border_touch_px, min_border_touch_frac)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    closed = cv2.morphologyEx(mask_to_close, cv2.MORPH_CLOSE, kernel)
    ys, xs = np.where(closed > 0)
    if len(xs) == 0:
        return None, closed, None

    object_mask = closed
    if prefer_central:
        cluster_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cluster_kernel, cluster_kernel))
        clustered = cv2.morphologyEx(mask_to_close, cv2.MORPH_CLOSE, cluster_k)
        n_labels2, labels2, stats2, _ = cv2.connectedComponentsWithStats(clustered, connectivity=8)
        if n_labels2 > 1:
            areas2 = stats2[1:, cv2.CC_STAT_AREA]
            best_cluster = 1 + int(np.argmax(areas2))
            final_mask = closed & (labels2 == best_cluster)
            ys_final, xs_final = np.where(final_mask > 0)
            if len(xs_final) > 0:
                ys, xs = ys_final, xs_final
                object_mask = final_mask

    box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return box, closed, object_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--border_px", type=int, default=20)
    ap.add_argument("--close_kernel", type=int, default=25)
    ap.add_argument("--max_points", type=int, default=6)
    ap.add_argument("--min_component_area", type=int, default=200)
    ap.add_argument("--use_all_frames_intersection", action="store_true",
                     help="Oltre al frame 0, calcola l'intersezione della maschera grezza su TUTTI i frame "
                          "(idea: pixel sempre-foreground nonostante la rotazione) e la salva come riferimento aggiuntivo")
    args = ap.parse_args()

    paths = sorted(
        p for p in glob.glob(os.path.join(args.frames_dir, "*"))
        if p.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if not paths:
        raise RuntimeError(f"Nessuna immagine trovata in {args.frames_dir}")

    frame0 = cv2.imread(paths[0])
    clean0, bg_lab = crude_foreground_mask(frame0, border_px=args.border_px,
                                            min_component_area=args.min_component_area)
    box, closed0, _ = close_and_bbox(clean0, close_kernel=args.close_kernel)
    points = find_anchor_points(clean0, max_points=args.max_points,
                                 min_component_area=args.min_component_area)
    area0 = int((clean0 > 0).sum())

    result = {
        "frame0_path": paths[0],
        "n_frames": len(paths),
        "box": list(box) if box else None,
        "points": [list(p) for p in points],
        "crude_area_frame0_px": area0,
    }

    if args.use_all_frames_intersection:
        intersection = np.ones_like(clean0, dtype=bool)
        for p in paths:
            img = cv2.imread(p)
            m, _ = crude_foreground_mask(img, bg_lab=bg_lab, min_component_area=args.min_component_area)
            intersection &= (m > 0)
        inter_area = int(intersection.sum())
        result["always_foreground_intersection_area_px"] = inter_area
        if inter_area > 0:
            multi_points = sample_points_greedy(intersection, n_points=args.max_points,
                                                 min_separation=max(20, args.close_kernel))
            result["always_foreground_deepest_point"] = multi_points[0] if multi_points else None
            result["always_foreground_multi_points"] = multi_points
        cv2.imwrite(os.path.join(os.path.dirname(args.out_json), "always_foreground_intersection.png"),
                    (intersection * 255).astype(np.uint8))

    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)

    # overlay di verifica visiva
    overlay = frame0.copy()
    if box:
        cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)
    for x, y in points:
        cv2.drawMarker(overlay, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
    overlay_path = os.path.join(os.path.dirname(args.out_json), "auto_prompt_overlay.png")
    cv2.imwrite(overlay_path, overlay)

    print(json.dumps(result, indent=2))
    print(f"\nOverlay di verifica: {overlay_path}")


if __name__ == "__main__":
    main()
