#!/usr/bin/env python3
"""
Standalone GradCAM accumulation script for DualSSN · T25kpc · beam_crop.

Runs GradCAM over every test-set image for all 30 runs (10 folds × 3 experiments)
and saves results incrementally to OUTPUT_DIR/gradcam_cache/run_XX.npz.

Each run is saved immediately on completion.  Re-running the script skips runs
whose cache file already exists, so interrupted jobs can be resumed without
losing work.

Usage
-----
    python run_gradcam_accumulation.py             # resume / skip existing
    python run_gradcam_accumulation.py --force     # overwrite all cached runs
    python run_gradcam_accumulation.py --force 3 7 # overwrite only runs 3 and 7
    python run_gradcam_accumulation.py --runs 0 5  # only process runs 0–5
"""

import argparse
import os
import re
import glob
import pickle
import itertools
import random as _random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch as _torch
import torch.nn.functional as _F
from sklearn.metrics import roc_curve, auc

from dcreclass.data.loaders import load_galaxies as _load_galaxies
from dcreclass.models.classifiers import DualScatterSqueezeNet as _DualSSN
from dcreclass.utils.calc_tools import (
    recalculate_metrics_with_correct_positive_class, round_to_1,
)
from kymatio.torch import Scattering2D as _Scat2D

# ── Paths (mirror notebook) ───────────────────────────────────────────────────
OUTPUT_DIR  = (
    '/users/mbredber/p2_DCRECLASS/outputs/scratch/figures/classifying'
    '/explore_classification_results_outputs'
)
CACHE_DIR   = os.path.join(OUTPUT_DIR, 'gradcam_cache')
MODELS_DIR  = '/users/mbredber/scratch/data/models/DualSSN_beam_crop_circular_5e-05_0.1_30_99_0.1'
IMAGE_CACHE = '/users/mbredber/p2_DCRECLASS/outputs/scratch/.cache/images'

os.makedirs(CACHE_DIR, exist_ok=True)

# ── RunConfig — full copy from notebook ───────────────────────────────────────
@dataclass
class RunConfig:
    classifier:      str   = 'ImageCNN'
    version:         str   = 'RAW'
    crop_mode:       str   = 'beam_crop'
    blur_method:     str   = 'circular'
    lr:              float = 5e-5
    reg:             float = 1e-1
    label_smoothing: float = 0.1
    percentile_lo:   int   = 30
    percentile_hi:   int   = 99
    noise_level:     float = 0.0
    folds:           List[int] = field(default_factory=lambda: [0])
    num_experiments: int   = 2
    galaxy_classes:  List[int] = field(default_factory=lambda: [50, 51])
    crop_size:       Tuple[int, int] = (512, 512)
    downsample_size: Tuple[int, int] = (128, 128)
    global_norm_mode: str  = 'percentile'
    adjust_positive_class: bool = True
    data_run_dir:    str   = '/users/mbredber/p2_DCRECLASS/outputs/scratch'
    preferred_old_idx: int = 0

    @property
    def metrics_dir(self) -> str:
        sub = (f"{self.classifier}_{self.crop_mode}_{self.blur_method}_"
               f"{self.lr}_{self.reg}_{self.percentile_lo}_"
               f"{self.percentile_hi}_{self.label_smoothing}")
        return os.path.join(self.data_run_dir, 'data', 'metrics', sub)

    @property
    def old_metrics_dir(self) -> str:
        return os.path.join(self.data_run_dir, 'data', 'metrics', 'old')

    @property
    def dataset_sizes(self) -> Dict[int, List[int]]:
        if self.galaxy_classes == [50, 51]:
            return {fold: [3000] for fold in range(10)}
        if self.galaxy_classes == [52, 53]:
            return {fold: [2, 16, 168] for fold in range(10)}
        raise ValueError(f'No dataset_sizes defined for galaxy_classes={self.galaxy_classes}')

    def base_key(self, fold: int, subset_size: int) -> str:
        return (f"{self.classifier}_ver{self.version}_cm{self.crop_mode}"
                f"_lr{self.lr}_reg{self.reg}_ls{self.label_smoothing}"
                f"_lo{self.percentile_lo}_hi{self.percentile_hi}"
                f"_nl{self.noise_level}"
                f"_f{fold}_ss{round_to_1(subset_size)}")

    def _base_key_no_nl(self, fold: int, subset_size: int) -> str:
        return (f"{self.classifier}_ver{self.version}_cm{self.crop_mode}"
                f"_lr{self.lr}_reg{self.reg}_ls{self.label_smoothing}"
                f"_lo{self.percentile_lo}_hi{self.percentile_hi}"
                f"_f{fold}_ss{round_to_1(subset_size)}")

    def pkl_path(self, fold: int, subset_size: int, experiment: int) -> str:
        return os.path.join(
            self.metrics_dir,
            f"{self.base_key(fold, subset_size)}_e{experiment}.pkl",
        )

    def _pkl_path_no_nl(self, fold: int, subset_size: int, experiment: int) -> str:
        return os.path.join(
            self.metrics_dir,
            f"{self._base_key_no_nl(fold, subset_size)}_e{experiment}.pkl",
        )

    def find_old_pkls(self, fold: int, experiment: int) -> List[str]:
        if self.blur_method != 'circular':
            return []
        if self.version != 'RAW' and self.crop_mode != 'beam_crop':
            return []
        old_version          = re.sub(r'^Blur(\d+kpc)$', r'RT\1',     self.version)
        old_version_prefixed = re.sub(r'^Blur(\d+kpc)$', r'RAW+RT\1', self.version)
        cs = f'cs{self.crop_size[0]}x{self.crop_size[1]}'
        ds = f'ds{self.downsample_size[0]}x{self.downsample_size[1]}'
        matches = []
        for ver in [old_version, old_version_prefixed]:
            base = (f"CNN_*_reg{self.reg}_lo{self.percentile_lo}_hi{self.percentile_hi}"
                    f"_{cs}_{ds}_ver{ver}_f{fold}_ss*_e{experiment}")
            for suffix in [f"_{self.global_norm_mode}_metrics_data.pkl", "_metrics_data.pkl"]:
                matches.extend(glob.glob(os.path.join(self.old_metrics_dir, base + suffix)))
        return sorted(set(matches))

    def find_pkl(self, fold: int, subset_size: int, experiment: int,
                 verbose: bool = True) -> Optional[str]:
        new_path = self.pkl_path(fold, subset_size, experiment)
        if os.path.exists(new_path):
            return new_path
        no_nl_path = self._pkl_path_no_nl(fold, subset_size, experiment)
        if os.path.exists(no_nl_path):
            return no_nl_path
        old_matches = self.find_old_pkls(fold, experiment)
        if not old_matches:
            return None
        idx = self.preferred_old_idx % len(old_matches)
        return old_matches[idx]


# ── LegacyRunConfig (needed for isinstance check inside load_run) ─────────────
@dataclass
class LegacyRunConfig:
    classifier:    str
    section:       str
    legacy_path:   str = '/users/mbredber/p2_DCRECLASS/outputs/scratch/data/PSZ2/legacy_results/legacy_metrics.json'
    version:       str = 'RAW'
    blur_method:   str = 'circular'
    lr:            float = 5e-5
    reg:           float = 0.1
    folds:         List[int] = field(default_factory=lambda: [0, 1])
    num_experiments: int = 1
    dataset_sizes: dict = field(default_factory=lambda: {0: [3000], 1: [3000]})


# ── load_run — full copy from notebook ────────────────────────────────────────
def load_run(cfg, verbose: bool = True) -> dict:
    if isinstance(cfg, LegacyRunConfig):
        raise NotImplementedError('Legacy runs not needed in this script')

    tot: dict = {}
    cluster = {'errors': [], 'distances': [], 'std_devs': []}
    loaded, failed = 0, 0
    old_keys: set = set()

    _cluster_map = {
        'cluster_error':    'errors',
        'cluster_distance': 'distances',
        'cluster_std_dev':  'std_devs',
    }

    # Auto-discover subset sizes from metrics_dir
    _disc_ss: set = set()
    _ver_pat = re.compile(r'_ver' + re.escape(cfg.version) + r'_')
    for _f in glob.glob(os.path.join(cfg.metrics_dir, '*.pkl')):
        _bn = os.path.basename(_f)
        if not _ver_pat.search(_bn):
            continue
        _m = re.search(r'_ss(\d+(?:\.\d+)?)', _bn)
        if _m:
            _disc_ss.add(int(round_to_1(float(_m.group(1)))))
    _sizes = {f: sorted(set(cfg.dataset_sizes.get(f, [])) | _disc_ss)
              for f in cfg.folds}

    _loaded_paths: set = set()
    for experiment, fold in itertools.product(range(cfg.num_experiments), cfg.folds):
        for subset_size in _sizes[fold]:
            path = cfg.find_pkl(fold, subset_size, experiment, verbose=verbose)
            if path is None:
                failed += 1
                continue
            if path in _loaded_paths:
                continue
            _loaded_paths.add(path)

            is_old = (os.path.dirname(os.path.abspath(path)) ==
                      os.path.abspath(cfg.old_metrics_dir))

            try:
                with open(path, 'rb') as fh:
                    data = pickle.load(fh)
            except FileNotFoundError:
                failed += 1
                continue
            except Exception as exc:
                if verbose:
                    print(f'  [err] {os.path.basename(path)}: {exc}')
                failed += 1
                continue

            for src_key, dst_list in _cluster_map.items():
                val = data.get(src_key)
                if val is not None:
                    cluster[dst_list].append(val)

            # Resolve the inner dict key used to store labels/probs
            new_base = cfg.base_key(fold, subset_size)
            if new_base in data.get('all_true_labels', {}):
                base = new_base
            else:
                available = list(data.get('all_true_labels', {}).keys())
                if not available:
                    failed += 1
                    continue
                base = available[0]

            y_true  = data['all_true_labels'].get(base, [])
            y_pred  = data['all_pred_labels'].get(base, [])
            y_probs = data['all_pred_probs'].get(base, [])

            if not y_true or not y_pred:
                failed += 1
                continue

            if cfg.adjust_positive_class:
                acc, prec, rec, f1 = recalculate_metrics_with_correct_positive_class(
                    y_true, y_pred, pos_label=0)
            else:
                m = data['metrics']
                acc  = m.get('accuracy',  [0.0])[0]
                prec = m.get('precision', [0.0])[0]
                rec  = m.get('recall',    [0.0])[0]
                f1   = m.get('f1_score',  [0.0])[0]

            auc_val = float('nan')
            if y_probs:
                try:
                    p = np.asarray(y_probs)
                    scores = p[:, 1] if p.ndim == 2 and p.shape[1] > 1 else p.ravel()
                    if np.unique(np.asarray(y_true)).size >= 2:
                        fpr_, tpr_, _ = roc_curve(y_true, scores)
                        auc_val = auc(fpr_, tpr_)
                except Exception:
                    pass

            k = f'{subset_size}_{fold}_{experiment}_{cfg.lr}_{cfg.reg}'
            if is_old:
                old_keys.add(k)

            for metric, val in [('accuracy', acc), ('precision', prec),
                                 ('recall', rec), ('f1_score', f1), ('auc', auc_val)]:
                tot.setdefault(f'{metric}_{k}', []).append(val)

            for store_key, obj in [
                ('all_true_labels', data['all_true_labels']),
                ('all_pred_labels', data['all_pred_labels']),
                ('all_pred_probs',  data['all_pred_probs']),
                ('history',         data.get('history', {})),
                ('training_times',  data.get('training_times', {})),
            ]:
                tot.setdefault(f'{store_key}_{k}', []).append(obj)

            loaded += 1

    if verbose:
        print(f'Loaded {loaded} pkl files, failed/missing {failed}.')
    tot['_cluster']  = cluster
    tot['_cfg']      = cfg
    tot['_old_keys'] = old_keys
    return tot


# ── GradCAM ───────────────────────────────────────────────────────────────────
def _gradcam(model, img_t, scat_t, layer, class_idx):
    act_buf, grad_buf = {}, {}
    h1 = layer.register_forward_hook(
        lambda m, i, o: act_buf.update(x=o.detach()))
    h2 = layer.register_full_backward_hook(
        lambda m, gi, go: grad_buf.update(x=go[0].detach()))
    model.zero_grad()
    with _torch.enable_grad():
        logits = model(img_t, scat_t)
        logits[0, class_idx].backward()
    h1.remove(); h2.remove()
    act = act_buf['x']
    grad = grad_buf['x']
    w   = grad.mean(dim=(2, 3), keepdim=True)
    cam = _F.relu((w * act).sum(dim=1, keepdim=True))
    cam = _F.interpolate(cam, (128, 128), mode='bilinear', align_corners=False)
    cam = cam.squeeze().numpy()
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    return cam


# ── Cache I/O ─────────────────────────────────────────────────────────────────
def cache_path(run_idx: int) -> str:
    return os.path.join(CACHE_DIR, f'run_{run_idx:02d}.npz')


def save_cache(run_idx: int, cam_img: dict, cam_scat: dict, raw: dict) -> None:
    arrays = {}
    for q in ('TP', 'FN', 'TN', 'FP'):
        arrays[f'{q}_cam_img']  = np.stack(cam_img[q],  0) if cam_img[q]  else np.empty((0, 128, 128))
        arrays[f'{q}_cam_scat'] = np.stack(cam_scat[q], 0) if cam_scat[q] else np.empty((0, 128, 128))
        arrays[f'{q}_raw']      = np.stack(raw[q],      0) if raw[q]      else np.empty((0, 128, 128))
    tmp = cache_path(run_idx)[:-4] + '.tmp.npz'  # end with .npz so numpy doesn't append it
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, cache_path(run_idx))   # atomic rename


# ── Setup: reproduce notebook cell 57 ────────────────────────────────────────
def build_run_quadrants(cfg: RunConfig):
    """Load pkl results and image data to build _all_run_quadrants and _eval_imgs_perm."""
    metrics   = load_run(cfg, verbose=False)
    ex_subset = max(sz for sizes in cfg.dataset_sizes.values() for sz in sizes)

    all_y_true  = None
    all_y_probs = []

    for fold in range(10):
        for exp in range(cfg.num_experiments):
            k      = f'{ex_subset}_{fold}_{exp}_{cfg.lr}_{cfg.reg}'
            tl_lst = metrics.get(f'all_true_labels_{k}', [])
            pp_lst = metrics.get(f'all_pred_probs_{k}',  [])
            if not tl_lst or not pp_lst:
                continue
            tl_d = tl_lst[0] if isinstance(tl_lst, list) else tl_lst
            pp_d = pp_lst[0] if isinstance(pp_lst, list) else pp_lst
            key  = next(iter(tl_d), None)
            if key is None:
                continue
            yt = np.array(tl_d[key])
            yp = np.array(pp_d.get(key, []))
            if yt.size == 0 or yp.size == 0:
                continue
            if all_y_true is None:
                all_y_true = yt
            all_y_probs.append(yp)

    assert all_y_true is not None, (
        f'No valid pkl entries found in {cfg.metrics_dir} for version={cfg.version}')
    n_test_aug = len(all_y_true)
    print(f'Test samples (aug): {n_test_aug},  runs pooled: {len(all_y_probs)}')

    # Reproduce perm_test — must exactly match the notebook seed sequence
    _random.seed(42); np.random.seed(42); _torch.manual_seed(42)
    # Scat2D construction consumes torch RNG — replicate notebook behaviour
    _scat_tmp = _Scat2D(J=2, L=12, shape=(128, 128), max_order=2)
    del _scat_tmp

    tst_out = _load_galaxies(
        galaxy_classes=[50, 51], versions=['T25kpc'], fold=9,
        crop_size=(512, 512), downsample_size=(128, 128), sample_size=1_000_000,
        REMOVEOUTLIERS=False, BALANCE=False, AUGMENT=True, STRETCH=True,
        percentile_lo=30, percentile_hi=99, NORMALISE=True, NORMALISETOPM=False,
        USE_GLOBAL_NORMALISATION=False, global_norm_mode='none',
        PRINTFILENAMES=True, USE_CACHE=True, DEBUG=False,
        crop_mode='beam_crop', blur_method='circular',
        cache_dir=IMAGE_CACHE,
        train=False,
    )
    _, _, eval_imgs_t, eval_lbl_t, _, eval_fns = tst_out

    eval_labels_raw = (eval_lbl_t.numpy()
                       if hasattr(eval_lbl_t, 'numpy') else np.array(eval_lbl_t))
    cls_to_idx      = {cls: i for i, cls in enumerate(cfg.galaxy_classes)}
    eval_labels_arr = np.array([cls_to_idx[int(l)] for l in eval_labels_raw])

    assert len(eval_fns) == n_test_aug, (
        f'Image cache size mismatch: got {len(eval_fns)}, expected {n_test_aug}')

    perm_test      = _torch.randperm(n_test_aug)
    fns_perm       = [eval_fns[j] for j in perm_test.numpy()]
    labels_perm    = eval_labels_arr[perm_test.numpy()]
    eval_imgs_perm = eval_imgs_t[perm_test]

    n_mismatch = int(np.sum(labels_perm != all_y_true))
    if n_mismatch == 0:
        print(f'perm_test validated — all {n_test_aug} labels aligned')
    else:
        print(f'WARNING: perm_test mismatch: {n_mismatch}/{n_test_aug} labels differ')

    PRED_CLASS = {'TP': 0, 'FP': 0, 'TN': 1, 'FN': 1}
    all_run_quadrants = []
    for run_probs in all_y_probs:
        prob_DE = run_probs[:, 0]
        rq = {'TP': [], 'FN': [], 'TN': [], 'FP': []}
        for i in range(n_test_aug):
            slug = fns_perm[i]
            prob = float(prob_DE[i])
            tl   = int(labels_perm[i])
            if   tl == 0 and prob >= 0.5: rq['TP'].append((slug, i, prob))
            elif tl == 0 and prob <  0.5: rq['FN'].append((slug, i, 1 - prob))
            elif tl == 1 and prob <  0.5: rq['TN'].append((slug, i, 1 - prob))
            else:                         rq['FP'].append((slug, i, prob))
        all_run_quadrants.append(rq)

    return all_run_quadrants, eval_imgs_perm, PRED_CLASS


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--force', nargs='*', metavar='RUN_IDX',
                        help='Overwrite cache. No args = overwrite all; '
                             'provide run indices to overwrite specific runs.')
    parser.add_argument('--runs', nargs='+', type=int, metavar='RUN_IDX',
                        help='Only process these run indices (default: all 30).')
    args = parser.parse_args()

    cfg = RunConfig(
        classifier='DualSSN', version='T25kpc', crop_mode='beam_crop',
        folds=list(range(10)), num_experiments=3,
    )

    if args.force is None:
        force_set = set()
    elif args.force == []:
        force_set = set(range(30))
    else:
        force_set = {int(x) for x in args.force}

    print('Building run quadrants (reproducing notebook cell 57)…')
    all_run_quadrants, eval_imgs_perm, PRED_CLASS = build_run_quadrants(cfg)

    n_runs     = len(all_run_quadrants)
    run_subset = args.runs if args.runs is not None else list(range(n_runs))

    scat_cam   = _Scat2D(J=2, L=12, shape=(128, 128), max_order=2)
    scat_shape = tuple(scat_cam(_torch.zeros(1, 1, 128, 128)).flatten(1, 2).shape[1:])
    print(f'scat_shape: {scat_shape}')

    def prep_input(aug_idx):
        img_t  = eval_imgs_perm[aug_idx:aug_idx+1].float()
        scat_t = scat_cam(img_t).flatten(1, 2)
        arr    = img_t.squeeze().numpy()
        return img_t, scat_t, arr

    cam_subset = 2000

    def load_model(run_idx):
        fold = run_idx // cfg.num_experiments
        ckpt = f'{MODELS_DIR}/{cfg.base_key(fold, cam_subset)}_best_model.pth'
        m    = _DualSSN(img_shape=(1, 128, 128), scat_shape=scat_shape, num_classes=2)
        m.load_state_dict(_torch.load(ckpt, map_location='cpu'))
        m.eval()
        return m

    print(f'\nProcessing {len(run_subset)} run(s): {run_subset}')
    print(f'Cache dir: {CACHE_DIR}\n')

    for run_idx in run_subset:
        cp = cache_path(run_idx)

        if os.path.exists(cp) and run_idx not in force_set:
            npz    = np.load(cp)
            n_imgs = sum(len(npz[f'{q}_cam_img']) for q in ('TP', 'FN', 'TN', 'FP'))
            print(f'Run {run_idx:2d}/{n_runs}  [cached]  ({n_imgs} images)')
            continue

        if run_idx >= n_runs:
            print(f'Run {run_idx}: index out of range (max {n_runs-1}), skipping.')
            continue

        model  = load_model(run_idx)
        rq     = all_run_quadrants[run_idx]
        n_imgs = sum(len(rq[q]) for q in ('TP', 'FN', 'TN', 'FP'))
        print(f'Run {run_idx:2d}/{n_runs}  ({n_imgs} images):', end=' ', flush=True)

        run_ci = {q: [] for q in ('TP', 'FN', 'TN', 'FP')}
        run_cs = {q: [] for q in ('TP', 'FN', 'TN', 'FP')}
        run_ra = {q: [] for q in ('TP', 'FN', 'TN', 'FP')}

        for q in ('TP', 'FN', 'TN', 'FP'):
            ci_idx = PRED_CLASS[q]
            for slug, aug_idx, _ in rq[q]:
                try:
                    img_t, scat_t, arr = prep_input(aug_idx)
                    run_ci[q].append(_gradcam(model, img_t, scat_t, model.conv_to_latent_img,  ci_idx))
                    run_cs[q].append(_gradcam(model, img_t, scat_t, model.conv_to_latent_scat, ci_idx))
                    run_ra[q].append(arr)
                except Exception as exc:
                    print(f'\n  skip {slug}[{aug_idx}]: {exc}', end='', flush=True)
            print(f'{q}({len(run_ci[q])})', end=' ', flush=True)

        save_cache(run_idx, run_ci, run_cs, run_ra)
        print(f'  -> saved to {os.path.basename(cp)}')

    print('\nDone.')
    for q in ('TP', 'FN', 'TN', 'FP'):
        total = sum(
            len(np.load(cache_path(i))[f'{q}_cam_img'])
            for i in run_subset
            if os.path.exists(cache_path(i))
        )
        print(f'  {q}: {total} CAM maps in cache')


if __name__ == '__main__':
    main()
