#!/usr/bin/env python3
"""Per-class AP + confusion matrix on proxy_val_v2 for E1 anchor.

Outputs:
  - per_class_ap.json: AP and AP@50 per class
  - confusion_matrix.csv: predicted_class x true_class (at IoU=0.5)
  - Also logs which classes have the worst AP (to identify miscalibrated classes)
"""
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np

REPO = Path('/workspace/fathomnet-2026')
TRAIN_IDX_TO_TEST_ID = [1,2,3,4,5,6,7,8,9,10,11,13,15,17,19,20,22,24,25,26,27,28,29,30,32,33,35,36,37,38,40,41]
ID_TO_NAME = {
    1:'amphipod',2:'anemone',3:'barnacle',4:'benthic worm',5:'bivalve',6:'black coral',
    7:'bony fish',8:'brittle star',9:'calycophoran siph.',10:'chiton',11:'crab',
    13:'feather star',15:'hydroid',17:'isopod',19:'jelly',20:'larvacean',22:'octopus',
    24:'physonect siph.',25:'pyrosome',26:'sea cucumber',27:'sea fan',28:'sea pen',
    29:'sea slug',30:'sea snail',32:'sea squirt',33:'sea star',35:'shrimp',36:'soft coral',
    37:'sponge',38:'squat lobster',40:'stony coral',41:'urchin',
}


def run_inference(weights, val_paths, imgsz=1024, batch=8):
    from ultralytics import YOLO
    m = YOLO(str(weights))
    rows = []
    for i in range(0, len(val_paths), batch):
        chunk = val_paths[i:i+batch]
        results = m.predict(source=[str(p) for p in chunk], imgsz=imgsz,
                            conf=0.001, iou=0.65, max_det=100, verbose=False, device=0)
        for off, (p, r) in enumerate(zip(chunk, results)):
            img_idx = i + off + 1
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
                cls = int(b.cls[0].item())
                sc = float(b.conf[0].item())
                tid = TRAIN_IDX_TO_TEST_ID[cls] if cls < len(TRAIN_IDX_TO_TEST_ID) else cls + 1
                rows.append({
                    'image_id': img_idx, 'category_id': tid,
                    'bbox_x': x1, 'bbox_y': y1,
                    'bbox_width': max(0, x2-x1), 'bbox_height': max(0, y2-y1),
                    'score': sc,
                })
    return rows


def build_gt(val_paths, label_dir):
    """Returns gt_coco, gt_by_img (for confusion matrix)."""
    images, anns = [], []
    cats = [{'id': i, 'name': ID_TO_NAME.get(i, f'cat{i}')} for i in TRAIN_IDX_TO_TEST_ID]
    aid = 1
    gt_by_img = defaultdict(list)
    for img_idx, p in enumerate(val_paths, 1):
        img = cv2.imread(str(p))
        if img is None:
            continue
        H, W = img.shape[:2]
        images.append({'id': img_idx, 'file_name': p.name, 'width': W, 'height': H})
        lab = Path(label_dir) / (p.stem + '.txt')
        if not lab.exists():
            continue
        for line in lab.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            c = int(float(parts[0]))
            xc = float(parts[1]) * W
            yc = float(parts[2]) * H
            bw = float(parts[3]) * W
            bh = float(parts[4]) * H
            x = xc - bw/2
            y = yc - bh/2
            tid = TRAIN_IDX_TO_TEST_ID[c] if c < len(TRAIN_IDX_TO_TEST_ID) else c + 1
            anns.append({'id': aid, 'image_id': img_idx, 'category_id': tid,
                         'bbox': [x, y, bw, bh], 'area': bw*bh, 'iscrowd': 0})
            gt_by_img[img_idx].append({'cat': tid, 'bbox_xyxy': (x, y, x+bw, y+bh)})
            aid += 1
    coco = {'images': images, 'annotations': anns, 'categories': cats}
    return coco, gt_by_img


def per_class_ap(gt_path, preds):
    """Use pycocotools to get per-class AP."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt = COCO(str(gt_path))
    if not preds:
        return {}, {}
    coco_preds = [{'image_id': r['image_id'], 'category_id': r['category_id'],
                   'bbox': [r['bbox_x'], r['bbox_y'], r['bbox_width'], r['bbox_height']],
                   'score': r['score']} for r in preds]
    dt = gt.loadRes(coco_preds)
    e = COCOeval(gt, dt, 'bbox')
    e.evaluate()
    e.accumulate()
    e.summarize()

    # Per-class AP = mean over IoU thresholds, all areas, max_det=100, per category
    # e.eval['precision'].shape = (T_iou, R_recall, K_cat, A_area, M_maxdet)
    precision = e.eval['precision']
    cats = sorted([c['id'] for c in gt.dataset['categories']])
    cat_id_to_idx = {cid: i for i, cid in enumerate(cats)}
    per_class_ap = {}
    per_class_ap50 = {}
    for cid in cats:
        i = cat_id_to_idx[cid]
        # area=all (idx 0), maxdet=100 (idx 2)
        prec_all = precision[:, :, i, 0, 2]
        prec_all = prec_all[prec_all > -1]
        ap = float(prec_all.mean()) if prec_all.size else 0.0
        prec_50 = precision[0, :, i, 0, 2]
        prec_50 = prec_50[prec_50 > -1]
        ap50 = float(prec_50.mean()) if prec_50.size else 0.0
        per_class_ap[cid] = ap
        per_class_ap50[cid] = ap50
    return per_class_ap, per_class_ap50


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2-ix1)
    ih = max(0, iy2-iy1)
    inter = iw * ih
    a_area = max(0, ax2-ax1) * max(0, ay2-ay1)
    b_area = max(0, bx2-bx1) * max(0, by2-by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def confusion_matrix(rows, gt_by_img, iou_thr=0.5, conf_thr=0.05):
    """Cross-class confusion: pred can match GT of different class for analysis."""
    by_img_pred = defaultdict(list)
    for r in rows:
        if r['score'] < conf_thr:
            continue
        by_img_pred[r['image_id']].append(r)

    confusion = defaultdict(lambda: defaultdict(int))
    classes_seen = set()
    missed = defaultdict(int)
    for img_idx, gts in gt_by_img.items():
        preds = by_img_pred.get(img_idx, [])
        gt_used = [False] * len(gts)
        preds_sorted = sorted(preds, key=lambda p: -p['score'])
        for p in preds_sorted:
            pbox = (p['bbox_x'], p['bbox_y'], p['bbox_x']+p['bbox_width'], p['bbox_y']+p['bbox_height'])
            best_iou, best_j = 0.0, -1
            for j, gt in enumerate(gts):
                if gt_used[j]:
                    continue
                iou = iou_xyxy(pbox, gt['bbox_xyxy'])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= iou_thr and best_j >= 0:
                true_cls = gts[best_j]['cat']
                gt_used[best_j] = True
                confusion[p['category_id']][true_cls] += 1
                classes_seen.add(true_cls)
            else:
                confusion[p['category_id']]['BG'] += 1
            classes_seen.add(p['category_id'])
        for j, used in enumerate(gt_used):
            if not used:
                missed[gts[j]['cat']] += 1
                classes_seen.add(gts[j]['cat'])

    return confusion, missed, sorted(classes_seen)


def per_class_tp_fp_fn(rows, gt_by_img, iou_thr=0.5, conf_thr=0.05):
    """Per-class TP/FP/FN at fixed IoU + conf threshold (class-aware matching).

    Standard detection metrics:
      TP_c = preds of class c that match a GT of class c at IoU>=thr
      FP_c = preds of class c that don't match any GT of class c (background or wrong-class)
      FN_c = GTs of class c not matched by any pred of class c
    """
    pred_by_img_class = defaultdict(list)
    for r in rows:
        if r['score'] < conf_thr:
            continue
        pred_by_img_class[(r['image_id'], r['category_id'])].append(r)
    gt_by_img_class = defaultdict(list)
    for img_idx, gts in gt_by_img.items():
        for g in gts:
            gt_by_img_class[(img_idx, g['cat'])].append(g)

    classes = set()
    for k in pred_by_img_class:
        classes.add(k[1])
    for k in gt_by_img_class:
        classes.add(k[1])

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    n_pred = defaultdict(int)
    n_gt = defaultdict(int)

    # Iterate over all (img, class) pairs
    all_keys = set(pred_by_img_class.keys()) | set(gt_by_img_class.keys())
    for (img_idx, cls) in all_keys:
        preds = pred_by_img_class.get((img_idx, cls), [])
        gts = gt_by_img_class.get((img_idx, cls), [])
        n_pred[cls] += len(preds)
        n_gt[cls] += len(gts)
        gt_used = [False] * len(gts)
        for p in sorted(preds, key=lambda x: -x['score']):
            pbox = (p['bbox_x'], p['bbox_y'], p['bbox_x']+p['bbox_width'], p['bbox_y']+p['bbox_height'])
            best_iou, best_j = 0.0, -1
            for j, gt in enumerate(gts):
                if gt_used[j]:
                    continue
                iou = iou_xyxy(pbox, gt['bbox_xyxy'])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= iou_thr and best_j >= 0:
                gt_used[best_j] = True
                tp[cls] += 1
            else:
                fp[cls] += 1
        fn[cls] += sum(1 for u in gt_used if not u)

    return tp, fp, fn, n_pred, n_gt, sorted(classes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default=str(REPO / 'runs/detect/weights/runs/p2_full_mbari_315k_aug_e50_imgsz1024_fold0/weights/best.pt'))
    ap.add_argument('--val-list', default=str(REPO / 'data/proxy_val_v2.txt'))
    ap.add_argument('--label-dir', default=str(REPO / 'data/raw/labels/train'))
    ap.add_argument('--imgsz', type=int, default=1024)
    ap.add_argument('--out-dir', default='/tmp/_per_class_eval')
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    val_paths = []
    for line in Path(a.val_list).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        p = Path(line) if Path(line).is_absolute() else REPO / line
        if p.exists():
            val_paths.append(p)
    print(f'Val imgs: {len(val_paths)}')

    print('1. E1 inference')
    rows = run_inference(a.weights, val_paths, imgsz=a.imgsz)
    print(f'  got {len(rows)} preds')

    print('2. Build GT')
    gt, gt_by_img = build_gt(val_paths, a.label_dir)
    gt_path = out_dir / 'gt_coco.json'
    gt_path.write_text(json.dumps(gt))

    print('3. Per-class AP via pycocotools')
    per_ap, per_ap50 = per_class_ap(gt_path, rows)

    print()
    print('=== PER-CLASS AP (sorted by mAP, worst first) ===')
    sorted_classes = sorted(per_ap.items(), key=lambda x: x[1])
    print(f'{"Class":<25} {"AP":>8} {"AP50":>8}')
    for cid, ap in sorted_classes:
        nm = ID_TO_NAME.get(cid, f'cat{cid}')
        print(f'{nm:<25} {ap:>8.4f} {per_ap50.get(cid, 0):>8.4f}')

    (out_dir / 'per_class_ap.json').write_text(json.dumps({
        'mAP_per_class': {str(k): v for k, v in per_ap.items()},
        'mAP50_per_class': {str(k): v for k, v in per_ap50.items()},
    }, indent=2))

    print()
    print('4. Per-class TP/FP/FN/Precision/Recall (class-aware match, IoU>=0.5, conf>=0.05)')
    tp, fp, fn, n_pred, n_gt, classes_tpfp = per_class_tp_fp_fn(rows, gt_by_img, iou_thr=0.5, conf_thr=0.05)
    print()
    print(f'{"Class":<25} {"TP":>6} {"FP":>6} {"FN":>6} {"Precision":>10} {"Recall":>8} {"#GT":>6} {"#Pred":>7} {"AP":>7} {"AP50":>7}')
    rows_metrics = []
    for cls in classes_tpfp:
        T = tp.get(cls, 0); F = fp.get(cls, 0); N = fn.get(cls, 0)
        prec = T / (T+F) if (T+F) > 0 else 0.0
        rec = T / (T+N) if (T+N) > 0 else 0.0
        ap = per_ap.get(cls, 0.0)
        ap50 = per_ap50.get(cls, 0.0)
        nm = ID_TO_NAME.get(cls, f'cat{cls}')
        rows_metrics.append({
            'class': nm, 'cat_id': cls, 'TP': T, 'FP': F, 'FN': N,
            'precision': prec, 'recall': rec, 'n_gt': n_gt.get(cls, 0), 'n_pred': n_pred.get(cls, 0),
            'AP': ap, 'AP50': ap50,
        })
        print(f'{nm:<25} {T:>6} {F:>6} {N:>6} {prec:>10.3f} {rec:>8.3f} {n_gt.get(cls, 0):>6} {n_pred.get(cls, 0):>7} {ap:>7.3f} {ap50:>7.3f}')
    # Save as CSV
    metrics_csv = out_dir / 'per_class_metrics.csv'
    with open(metrics_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['class', 'cat_id', 'TP', 'FP', 'FN', 'precision', 'recall', 'n_gt', 'n_pred', 'AP', 'AP50'])
        w.writeheader()
        for r in rows_metrics:
            w.writerow(r)
    print(f'Wrote {metrics_csv}')

    print()
    print('5. Confusion matrix (cross-class, IoU>=0.5, conf>=0.05)')
    confusion, missed, classes = confusion_matrix(rows, gt_by_img, iou_thr=0.5, conf_thr=0.05)

    # Save confusion matrix as CSV: rows=predicted, cols=true (+BG), values=counts
    cm_path = out_dir / 'confusion_matrix.csv'
    with open(cm_path, 'w', newline='') as f:
        w = csv.writer(f)
        header = ['predicted_class'] + [ID_TO_NAME.get(c, f'cat{c}') for c in classes] + ['BG_FP']
        w.writerow(header)
        for pred_cls in classes:
            row = [ID_TO_NAME.get(pred_cls, f'cat{pred_cls}')]
            for true_cls in classes:
                row.append(confusion.get(pred_cls, {}).get(true_cls, 0))
            row.append(confusion.get(pred_cls, {}).get('BG', 0))
            w.writerow(row)
        # Add MISSED row
        row = ['MISSED_GT (no pred matches)']
        for true_cls in classes:
            row.append(missed.get(true_cls, 0))
        row.append(0)
        w.writerow(row)

    print(f'Wrote {cm_path}')

    print()
    print('=== TOP-10 MOST CONFUSED PAIRS (predicted -> true, where pred != true) ===')
    confused_pairs = []
    for pred_cls, dct in confusion.items():
        for true_cls, cnt in dct.items():
            if pred_cls != true_cls and true_cls != 'BG':
                confused_pairs.append((cnt, pred_cls, true_cls))
    confused_pairs.sort(reverse=True)
    for cnt, pcls, tcls in confused_pairs[:10]:
        print(f'  {ID_TO_NAME.get(pcls, "cat"+str(pcls)):<22} -> {ID_TO_NAME.get(tcls, "cat"+str(tcls)):<22} : {cnt}')

    print()
    print('=== TOP-10 BG_FP (predicted X but no GT match - hallucinations) ===')
    bg_fps = sorted(((confusion.get(c, {}).get('BG', 0), c) for c in classes), reverse=True)
    for cnt, c in bg_fps[:10]:
        print(f'  {ID_TO_NAME.get(c, "cat"+str(c)):<22} : {cnt}')

    print()
    print('=== TOP-10 MISSED GT (model didn\'t predict any matching box) ===')
    missed_sorted = sorted(missed.items(), key=lambda x: -x[1])
    for cls, cnt in missed_sorted[:10]:
        print(f'  {ID_TO_NAME.get(cls, "cat"+str(cls)):<22} : {cnt}')

    print(f'\nWrote {out_dir}/per_class_ap.json + {out_dir}/confusion_matrix.csv')


if __name__ == '__main__':
    main()
