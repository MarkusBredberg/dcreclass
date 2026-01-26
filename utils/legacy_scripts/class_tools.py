
# --- 3-column (RAW | rtXX | T50kpc) quicklook for TEST split ---
def _get_t50_for_names(basenames, crop_size, downsample_size):
    """Return {base: 2D numpy image} for any bases that exist on disk as T50kpc."""
    # Load only T50kpc for the test split – same preprocessing (crop/downsample/stretch)
    out = _loader(
        galaxy_classes=galaxy_classes,
        versions=['T50kpc'],
        fold=max(folds),                      # same "test" fold rule
        crop_size=crop_size,
        downsample_size=downsample_size,
        sample_size=10**9,                    # don't trim
        REMOVEOUTLIERS=FILTERED,
        BALANCE=False,
        STRETCH=STRETCH,
        percentile_lo=percentile_lo,
        percentile_hi=percentile_hi,
        AUGMENT=not LATE_AUG,
        NORMALISE=NORMALISEIMGS,
        NORMALISETOPM=NORMALISEIMGSTOPM,
        USE_GLOBAL_NORMALISATION=USE_GLOBAL_NORMALISATION,
        GLOBAL_NORM_MODE=GLOBAL_NORM_MODE,
        PRINTFILENAMES=True,
        train=False
    )
    # Unpack
    timgs, tfns = (out[2], out[5]) if len(out) == 6 else (out[2], None)
    tmap = {}
    if tfns is None: return tmap
    for img, fn in zip(timgs, tfns):
        base = Path(str(fn)).stem.split('T', 1)[0]
        if base in basenames:
            arr = img.squeeze(0).detach().cpu().numpy()
            tmap[base] = arr
    return tmap


def _to_base_name(fn):
    return Path(str(fn)).stem.split('T', 1)[0]

def _img2np(img_t):
    x = img_t.detach().cpu()
    if x.ndim == 4: x = x[0]                 # [T/C,1,H,W] → [1,H,W]
    if x.ndim == 3: x = x[0] if x.shape[0] > 1 else x.squeeze(0)
    return x.numpy()

def _first_recursive(pattern: str):
    hits = sorted(glob.glob(pattern, recursive=True))
    return hits[0] if hits else None

def _t50_path_for(base):
    """
    Try several common layouts and then fall back to a deep, recursive search
    anywhere under PSZ2_ROOT. Returns the first hit or None.
    """
    # Typical local directory under /fits/<base>/...
    base_dir = _first(f"{PSZ2_ROOT}/fits/{base}*") or f"{PSZ2_ROOT}/fits/{base}"

    # Fast, specific patterns first
    patterns = [
        f"{base_dir}/{base}T50kpc*.fits",
        f"{base_dir}/{base}_T50kpc*.fits",
        f"{PSZ2_ROOT}/classified/T50kpc/*/{base}.fits",
        f"{PSZ2_ROOT}/classified/T50kpcSUB/*/{base}.fits",
    ]
    for pat in patterns:
        p = _first(pat)
        if p: return p

    # Robust fallbacks (recursive, cover many installs)
    deep_patterns = [
        f"{PSZ2_ROOT}/fits/**/{base}T50kpc*.fits",
        f"{PSZ2_ROOT}/fits/**/{base}_T50kpc*.fits",
        f"{PSZ2_ROOT}/**/T50kpc*/**/{base}.fits",
        f"{PSZ2_ROOT}/**/T50kpc*/**/{base}*.fits",
        f"{PSZ2_ROOT}/**/{base}T50kpc*.fits",
        f"{PSZ2_ROOT}/**/{base}_T50kpc*.fits",
    ]
    for pat in deep_patterns:
        p = _first_recursive(pat)
        if p: return p

    return None


def _format_T50_for_display(base):
    """Load fixed T50 FITS and format like the pipeline (crop/downsample + stretch + asinh)."""
    p = _t50_path_for(base)
    if not p: 
        return None
    arr = np.squeeze(fits.getdata(p)).astype(float)
    arr = np.nan_to_num(arr, copy=False)
    t = torch.from_numpy(arr).float().unsqueeze(0)  # [1,H,W]
    formatted = apply_formatting(
        t,
        crop_size=(1, crop_size[-2], crop_size[-1]),
        downsample_size=(1, downsample_size[-2], downsample_size[-1])
    ).squeeze(0)  # [1,h,w]
    if STRETCH:
        formatted = _per_image_percentile_stretch(
            formatted.squeeze(0), percentile_lo, percentile_hi
        ).unsqueeze(0)
    img = torch.asinh(10.0 * formatted) / math.asinh(10.0)
    return img.squeeze(0).numpy()           # 2D numpy

def plot_first_rows_by_source(images, filenames, versions, out_path, n_show=10):
    """
    Plot the first N rows titled by source name. If images are [B,T,1,H,W] or [B,T,H,W],
    show 2 columns (left/right = first/second plane). If single version, plot 1 column.
    """

    if isinstance(images, (list, tuple)):
        images = torch.stack([torch.as_tensor(x) for x in images], dim=0)

    # Normalize shape to convenient form
    if images.dim() == 5:                 # [B, T, 1, H, W]
        images = images.flatten(2, 3)     # [B, T, H, W]
    elif images.dim() == 4:               # [B, C, H, W]
        pass
    else:
        raise ValueError(f"Unsupported images ndim={images.ndim}")

    B = images.shape[0]
    n_show = min(n_show, B)

    # Row titles from filenames (strip trailing T*kpc or T*kpcSUB)
    names = (filenames[:n_show] if filenames else [f"idx_{i}" for i in range(n_show)])
    def _src_name(s):
        b = os.path.splitext(os.path.basename(str(s)))[0]
        return re.sub(r'(?:T\d+kpc(?:SUB)?)$', '', b)
    row_titles = [_src_name(s) for s in names]

    is_two_cols = images.shape[1] >= 2      # have at least two planes (e.g., RT/T)
    if is_two_cols:
        fig, axes = plt.subplots(n_show, 2, figsize=(5.4, 2.6*n_show), constrained_layout=True)
        if n_show == 1:
            axes = np.array([axes])
        for i in range(n_show):
            left  = images[i, 0].detach().cpu().numpy()
            right = images[i, 1].detach().cpu().numpy()
            axes[i, 0].imshow(left, cmap="viridis", origin="lower")
            axes[i, 0].set_title(f"{row_titles[i]} — {versions[0] if isinstance(versions,(list,tuple)) else 'v0'}")
            axes[i, 0].axis('off')
            axes[i, 1].imshow(right, cmap="viridis", origin="lower")
            axes[i, 1].set_title(f"{row_titles[i]} — {versions[1] if isinstance(versions,(list,tuple)) else 'v1'}")
            axes[i, 1].axis('off')
    else:
        fig, axes = plt.subplots(n_show, 1, figsize=(2.7, 2.6*n_show), constrained_layout=True)
        if n_show == 1:
            axes = [axes]
        for i in range(n_show):
            img = images[i, 0].detach().cpu().numpy()   # first/only channel
            ax = axes[i]
            ax.imshow(img, cmap="viridis", origin="lower")
            ax.set_title(f"{row_titles[i]} — {versions if not isinstance(versions,(list,tuple)) else versions[0]}")
            ax.axis('off')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[quicklook] wrote {out_path}")


def plot_before_after_rt_3col(raw_imgs, raw_fns, rt_imgs, rt_fns, tag='rt50',
                              outdir='./classifier/debug_rt_before_after', per_page=24):
    os.makedirs(outdir, exist_ok=True)
    bmap = { _to_base_name(fn): i for i, fn in enumerate(raw_fns or []) }
    amap = { _to_base_name(fn): i for i, fn in enumerate(rt_fns  or []) }
    common = sorted(set(bmap) & set(amap))
    if not common:
        print("[rt-debug] no overlap between RAW and RT filename sets.")
        return

    # Try to fetch T50 images once (fast path), then fallback per-row if missing
    t50_map = _get_t50_for_names(set(common), crop_size, downsample_size)

    for page in range(0, len(common), per_page):
        chunk = common[page:page+per_page]
        n = len(chunk)
        fig, axes = plt.subplots(n, 3, figsize=(9.2, 3.0*n))
        if n == 1: axes = np.array([axes])
        for r, name in enumerate(chunk):
            i_raw, i_rt = bmap[name], amap[name]
            im_raw = _img2np(raw_imgs[i_raw])
            im_rt  = _img2np(rt_imgs[i_rt])
            im_t50 = t50_map.get(name, None)
            if im_t50 is None:
                im_t50 = _format_T50_for_display(name)  # fallback search

            axL, axM, axR = axes[r, 0], axes[r, 1], axes[r, 2]
            axL.imshow(im_raw, origin='lower', cmap='viridis'); axL.set_title(f"{name} — RAW");  axL.axis('off')
            axM.imshow(im_rt,  origin='lower', cmap='viridis'); axM.set_title(f"{name} — {tag}"); axM.axis('off')
            if im_t50 is not None:
                axR.imshow(im_t50, origin='lower', cmap='viridis'); axR.set_title(f"{name} — T50kpc"); axR.axis('off')
            else:
                axR.axis('off'); axR.text(0.5, 0.5, 'no T50', ha='center', va='center', transform=axR.transAxes)
        fig.tight_layout()
        out = os.path.join(outdir, f"test_before_after_{tag}_with_T50_p{page//per_page+1:02d}.png")
        fig.savefig(out, dpi=160, bbox_inches='tight'); plt.close(fig)
        print(f"[rt-debug] wrote {out}")


# --- DEBUG: save 5 examples in RAW and RT form, titled by source name ---
def _format_raw_for_display(fn):
    """Load the RAW FITS for a source name and format it like the pipeline (crop/resize + stretch)."""
    base = _name_base_from_fn(fn)
    _, _, _, _, _, raw_path = _headers_for_name(base)
    raw_native = np.squeeze(fits.getdata(raw_path)).astype(float)
    raw_native = np.nan_to_num(raw_native, copy=False)

    t = torch.from_numpy(raw_native).float().unsqueeze(0)  # [1,H,W]
    formatted = apply_formatting(
        t,
        crop_size=(1, crop_size[-2], crop_size[-1]),
        downsample_size=(1, downsample_size[-2], downsample_size[-1])
    ).squeeze(0)  # [1,H',W']

    img = _per_image_percentile_stretch(
        formatted.squeeze(0), percentile_lo, percentile_hi
    ).unsqueeze(0) if STRETCH else formatted

    # same as the main pipeline
    img = torch.asinh(10.0 * img) / math.asinh(10.0)
    return img  # [1,H',W']

def save_raw_and_rt_examples(fns, want_rt='rt50', n=5,
                             outdir='./classifier/debug_rt_examples'):
    """Save n sources in RAW and RT form. Titles = source names."""
    Hout, Wout = downsample_size[-2], downsample_size[-1]
    os.makedirs(os.path.join(outdir, 'raw'), exist_ok=True)
    os.makedirs(os.path.join(outdir, want_rt), exist_ok=True)

    picked = []
    for fn in map(str, fns or []):
        if len(picked) >= n:
            break
        base = _name_base_from_fn(fn)
        try:
            # RAW (formatted like the pipeline)
            raw_img = _format_raw_for_display(base)  # [1,H,W]

            # RT (synthesized from RAW headers)
            dummy = torch.zeros((1, 1, Hout, Wout))  # content ignored; apply_taper_to_tensor reads FITS itself
            rt_img, keep_mask, kept_fns, skipped = apply_taper_to_tensor(
                dummy, want_rt, filenames=[base],
                crop_size=crop_size,
                downsample_size=downsample_size,
                percentile_lo=percentile_lo,
                percentile_hi=percentile_hi,
                do_stretch=STRETCH,
                require_fixed_header=False,
            )
            if rt_img.shape[0] == 0:
                continue  # skip if this source can't produce the requested RT

            # Save RAW
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.imshow(raw_img.squeeze(0).numpy(), origin='lower', cmap='viridis')
            ax.set_title(base)   # title = source name only
            ax.axis('off')
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, 'raw', f"{base}_raw{versions}.png"), dpi=150)
            plt.close(fig)

            # Save RT
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.imshow(rt_img.squeeze(0).squeeze(0).cpu().numpy(), origin='lower', cmap='viridis')
            ax.set_title(base)   # title = source name only
            ax.axis('off')
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, want_rt, f"{base}_{want_rt}{versions}.png"), dpi=150)
            plt.close(fig)

            picked.append(base)
        except Exception as e:
            print(f"[debug-rt] skipping {base}: {e}")

    print(f"[debug-rt] wrote {len(picked)} examples under {outdir}")
    return picked


def image_to_vis(img, img_hdr, beam_hdr=None, divide_by_beam=True,
                 window='tukey', alpha=0.35, pad_factor=2, roi=0.92,
                 subtract_mean=True):
    """
    Convert a 2D sky image to its discrete Fourier transform (visibilities).

    • Optional conversion to Jy/sr (divide by Ω_beam) before the FFT.
    • Robust DC removal: subtract median inside a central elliptical ROI.
    • Strong Tukey apodization to tame edge/aliasing.
    • Optional zero-padding (×2 by default) to reduce wrap-around.
    """
    B = np.array(img, dtype=float)                  # keep NaNs for stats
    A = np.where(np.isfinite(B), B, 0.0)            # NaNs → 0 for FFT path

    # Jy/beam → Jy/sr if requested
    if divide_by_beam and (beam_hdr is not None):
        try:
            A = A / _beam_solid_angle_sr(beam_hdr)
        except Exception:
            pass

    ny, nx = A.shape

    # DC removal using central ellipse
    if subtract_mean and (0.0 < roi <= 1.0):
        cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
        ry, rx = (roi * ny) / 2.0, (roi * nx) / 2.0
        yy, xx = np.ogrid[:ny, :nx]
        mask_roi = (((yy - cy) / ry)**2 + ((xx - cx) / rx)**2) <= 1.0
        if np.any(mask_roi):
            dc = np.nanmedian(np.where(mask_roi, B, np.nan))
            if np.isfinite(dc):
                A = A - dc

    # Apodize
    if window == 'tukey':
        A *= radial_tukey(ny, nx, alpha=alpha)
    elif window == 'hann':
        A *= radial_hann(ny, nx)
    # Zero-padding
    if pad_factor and pad_factor > 1:
        ny2, nx2 = int(ny * pad_factor), int(nx * pad_factor)
        py = (ny2 - ny) // 2; px = (nx2 - nx) // 2
        A = np.pad(A, ((py, ny2 - ny - py), (px, nx2 - nx - px)),
                   mode='constant', constant_values=0.0)
        ny, nx = A.shape

    # Pixel size [rad]
    dx = abs(_cdelt_deg(img_hdr, 1)) * np.pi / 180.0
    dy = abs(_cdelt_deg(img_hdr, 2)) * np.pi / 180.0

    # Continuous-norm FFT ⇒ |F| in Jy
    F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(A))) * (dx * dy)

    # Frequency axes (cycles/radian = wavelengths)
    u = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    v = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    U, V = np.meshgrid(u, v)
    return U, V, F, np.abs(F)

# --- helper: plot RAW (as loaded) vs RT (after _replace_with_rt) for TEST split ---
def _to_base_name(fn):
    return Path(str(fn)).stem.split('T', 1)[0]

def _img2np(img_t):
    """Accepts [C,H,W] or [1,H,W] or [H,W] torch tensor → 2D numpy array."""
    x = img_t.detach().cpu()
    if x.ndim == 4:           # [T/C,1,H,W] or similar – take first plane
        x = x[0]
    if x.ndim == 3:
        x = x[0] if x.shape[0] > 1 else x.squeeze(0)
    return x.numpy()

def plot_before_after_rt(raw_imgs, raw_fns, rt_imgs, rt_fns, tag='rt50',
                         outdir='./classifier/debug_rt_before_after', per_page=24):
    os.makedirs(outdir, exist_ok=True)
    # index by base name (PSZ2G…)
    bmap = { _to_base_name(fn): i for i, fn in enumerate(raw_fns or []) }
    amap = { _to_base_name(fn): i for i, fn in enumerate(rt_fns  or []) }
    common = sorted(set(bmap) & set(amap))
    if not common:
        print("[rt-debug] no overlap between RAW and RT filename sets.")
        return

    # chunk into pages to avoid a gigantic single figure
    for page in range(0, len(common), per_page):
        chunk = common[page:page+per_page]
        n = len(chunk)
        fig, axes = plt.subplots(n, 2, figsize=(6.4, 3.0*n))
        if n == 1: axes = np.array([axes])  # keep 2D indexing
        for r, name in enumerate(chunk):
            i_raw = bmap[name]; i_rt = amap[name]
            im_raw = _img2np(raw_imgs[i_raw])
            im_rt  = _img2np(rt_imgs[i_rt])
            axL, axR = axes[r, 0], axes[r, 1]
            axL.imshow(im_raw, origin='lower', cmap='viridis'); axL.set_title(f"{name} — RAW"); axL.axis('off')
            axR.imshow(im_rt,  origin='lower', cmap='viridis'); axR.set_title(f"{name} — {tag}"); axR.axis('off')
        fig.tight_layout()
        out = os.path.join(outdir, f"test_before_after_{tag}_p{page//per_page+1:02d}.png")
        fig.savefig(out, dpi=160, bbox_inches='tight'); plt.close(fig)
        print(f"[rt-debug] wrote {out}")


def shuffle_with_filenames(images, labels, filenames=None):
    perm = torch.randperm(images.size(0))
    images, labels = images[perm], labels[perm]
    if filenames is not None:
        filenames = [filenames[i] for i in perm.tolist()]
    return images, labels, filenames

def vis_compare_quicklook(name, T, RT, hdr_img, hdr_beam, tag, outdir='.', nbins=36):
    """
    Compact figure comparing visibilities of TARGET image (T) and RAW→TARGET (RT).
    Adds:
      • annulus-wise amplitude mask for Δϕ map,
      • optional coherence mask on the ratio curve to suppress noisy spikes.
    """
    # FFTs
    U,V,FT,AT = image_to_vis(T,  hdr_img, beam_hdr=hdr_beam)
    _,_,FR,AR = image_to_vis(RT, hdr_img, beam_hdr=hdr_beam)

    # Differences
    dphi = np.angle(FT * np.conj(FR))

    # Radial profiles
    r, aT = _radial_bin(U,V,AT, nbins=nbins, stat='median')
    _, aR = _radial_bin(U,V,AR, nbins=nbins, stat='median')
    ratio = aT / (aR + 1e-12)

    # Coherence and cross-phase per annulus
    Rgrid = np.sqrt(U*U + V*V)
    edges = np.geomspace(np.nanpercentile(Rgrid,1.0), np.nanmax(Rgrid), nbins+1)

    num_r, den1_r, den2_r, ph_r = [], [], [], []
    for i in range(nbins):
        m = (Rgrid>=edges[i]) & (Rgrid<edges[i+1])
        if not np.any(m):
            num_r.append(np.nan); den1_r.append(np.nan); den2_r.append(np.nan); ph_r.append(np.nan); continue
        num  = np.nanmean(FT[m] * np.conj(FR[m]))
        den1 = np.nanmean(np.abs(FT[m])**2)
        den2 = np.nanmean(np.abs(FR[m])**2)
        num_r.append(np.abs(num)); den1_r.append(den1); den2_r.append(den2); ph_r.append(np.angle(num))
    num_r = np.asarray(num_r); den1_r = np.asarray(den1_r); den2_r = np.asarray(den2_r)
    coh   = num_r / np.sqrt(den1_r * den2_r)

    # Δϕ map: keep only uv pixels above a local amplitude threshold
    amp   = 0.5*(AT + AR)
    keep  = np.zeros_like(amp, dtype=bool)
    for i in range(nbins):
        m = (Rgrid>=edges[i]) & (Rgrid<edges[i+1])
        if not np.any(m): 
            continue
        thr = np.nanpercentile(amp[m], 35.0)
        keep |= (m & (amp > thr))
    dphi_deg_map = np.where(keep, np.rad2deg(dphi), np.nan)

    # ---- figure ----
    fig = plt.figure(figsize=(12.2, 6.6))
    gs  = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.55],
                           height_ratios=[1.0, 0.42], wspace=0.26, hspace=0.28)
    gs_maps = gs[0, 0:2].subgridspec(2, 2, wspace=0.08, hspace=0.12)

    # |F_T| (log10)
    ax00 = fig.add_subplot(gs_maps[0,0])
    ax00.imshow(np.log10(AT + 1e-12), origin='lower', aspect='equal')
    ax00.set_title('|F_T| (log10)'); ax00.set_axis_off()

    # |F_RT| (log10)
    ax01 = fig.add_subplot(gs_maps[0,1])
    ax01.imshow(np.log10(AR + 1e-12), origin='lower', aspect='equal')
    ax01.set_title('|F_RT| (log10)'); ax01.set_axis_off()

    # Δ|F|  <-- MISSING AXIS WAS HERE
    ax10 = fig.add_subplot(gs_maps[1,0])   # <— add this line
    dA_abs = np.abs(FT) - np.abs(FR)
    dA_rel = dA_abs / (0.5*(np.abs(FT)+np.abs(FR)) + 1e-12)
    dA_plot = np.where(keep, dA_rel, np.nan)
    finite = np.isfinite(dA_plot)
    R = (np.nanpercentile(np.abs(dA_plot[finite]), 99.5) if np.any(finite) else 1.0)
    norm_dA = TwoSlopeNorm(vmin=-R, vcenter=0.0, vmax=+R)
    im10 = ax10.imshow(dA_plot, origin='lower', cmap='RdBu_r', norm=norm_dA, aspect='equal')
    ax10.set_title('Δ|F| (relative)'); ax10.set_axis_off()
    div10 = make_axes_locatable(ax10); cax10 = div10.append_axes("right", size="4.5%", pad=0.04)
    fig.colorbar(im10, cax=cax10, label='Δ|F| / ⟨|F|⟩')

    # Δϕ (deg)
    ax11 = fig.add_subplot(gs_maps[1,1])
    im11 = ax11.imshow(dphi_deg_map, origin='lower', cmap='twilight', vmin=-180, vmax=180, aspect='equal')
    ax11.set_title('Δϕ (deg)'); ax11.set_axis_off()
    div11 = make_axes_locatable(ax11); cax11 = div11.append_axes("right", size="4.5%", pad=0.04)
    fig.colorbar(im11, cax=cax11, label='Δϕ (deg)')

    # Right column: radial |F| + ratio/coherence
    axA = fig.add_subplot(gs[:, 2])
    kλ = r/1e3
    axA.plot(kλ, aT, lw=1.5, label='|F_T| (median)')
    axA.plot(kλ, aR, lw=1.5, ls='--', label='|F_RT| (median)')
    axA.set_xscale('log'); axA.set_yscale('log')
    axA.set_xlabel('baseline (kλ)'); axA.set_ylabel('median |F| (Jy)')
    axA.grid(True, which='both', ls=':', alpha=0.4)

    # Suppress meaningless ratio spikes at very low coherence
    good = (coh > 0.6)
    # Add a tiny floor on both medians
    thrT = np.nanpercentile(aT, 5.0)
    thrR = np.nanpercentile(aR, 5.0)
    good &= (aT > thrT) & (aR > thrR)
    ratio_plot = np.where(good, ratio, np.nan)

    axA2 = axA.twinx()
    axA2.plot(kλ, ratio_plot, lw=1.0, label='|F_T| / |F_RT|')
    axA2.plot(kλ, coh,        lw=1.0, label='coherence')
    axA2.set_ylabel('ratio / coherence')
    ymax = np.nanmax([np.nanmax(coh), np.nanmax(ratio_plot), 1.0])
    axA2.set_ylim(0, max(1.05, ymax))

    lines, labels = axA.get_legend_handles_labels()
    lines2, labels2 = axA2.get_legend_handles_labels()
    axA.legend(lines + lines2, labels + labels2, loc='lower left', fontsize=9)

    # Bottom: phase stats vs baseline (amplitude-weighted circular stats)
    axP = fig.add_subplot(gs[1, 0:2])
    phi_bar_deg, sigma_circ_deg, rphi = [], [], []
    for i in range(nbins):
        m = (Rgrid >= edges[i]) & (Rgrid < edges[i+1]) & keep & np.isfinite(dphi)
        if not np.any(m):
            phi_bar_deg.append(np.nan); sigma_circ_deg.append(np.nan); rphi.append(np.nan); continue
        w = AT[m] * AR[m]
        z = np.exp(1j * dphi[m])
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
        if np.sum(w) <= 0: w = np.ones_like(w)
        C = np.sum(w * z) / np.sum(w)
        Rlen = np.abs(C)
        phi_bar_deg.append(np.rad2deg(np.angle(C)))
        sigma_circ_deg.append(np.rad2deg(np.sqrt(np.maximum(0.0, -2.0*np.log(Rlen)))))
        rphi.append(0.5*(edges[i] + edges[i+1]))
    rphi = np.asarray(rphi); phi_bar_deg = np.asarray(phi_bar_deg); sigma_circ_deg = np.asarray(sigma_circ_deg)
    axP.plot(rphi/1e3, np.abs(phi_bar_deg), lw=1.2, label='|mean Δϕ|')
    axP.plot(rphi/1e3, sigma_circ_deg, lw=1.2, ls='--', label='circular σ(Δϕ)')
    axP.set_xscale('log'); axP.set_xlabel('baseline (kλ)'); axP.set_ylabel('phase (deg)')
    axP.grid(True, which='both', ls=':', alpha=0.4); axP.legend(loc='upper left', fontsize=9)

    fig.suptitle(f'{name}  –  {tag}', y=0.98, fontsize=12)
    save_dir = os.path.join(outdir, 'uvmaps'); os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, f'uvcmp_{name}_{tag}.pdf')
    fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f'Wrote {out}')

def _first(pattern: str):
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else None

def _to_strs(x): 
    return [str(y) for y in x] if isinstance(x, (list, tuple)) else list(map(str, x))

def _filter_to_keep_set(imgs, labels, fns, keep_set):
    """Keep only samples whose filename is in keep_set; returns (imgs, labels, fns)."""
    if fns is None:
        raise RuntimeError("Filenames are required to filter by anchor T*kpc. Set PRINTFILENAMES=True.")
    keep_idx = [i for i, fn in enumerate(_to_strs(fns)) if str(fn) in keep_set]
    if not keep_idx:  # produce empty tensors with correct shape/device
        return imgs[:0], labels[:0], []
    imgs = imgs[keep_idx]
    labels = labels[keep_idx]
    fns = [fns[i] for i in keep_idx]
    return imgs, labels, fns

def _pixscale_arcsec(hdr):
    if 'CDELT1' in hdr:  # deg/pix
        return abs(hdr['CDELT1']) * 3600.0
    cd11 = hdr.get('CD1_1'); cd12 = hdr.get('CD1_2', 0.0)
    if cd11 is not None:
        return float(np.hypot(cd11, cd12)) * 3600.0
    raise KeyError("No CDELT* or CD* keywords in FITS header")

def collapse_logits(logits, num_classes, multilabel):
    # [B,C,H,W] → [B,C]
    if logits.ndim == 4:
        logits = torch.nn.functional.adaptive_avg_pool2d(logits, (1, 1)).squeeze(-1).squeeze(-1)
    elif logits.ndim == 3:
        logits = logits.mean(dim=-1)
    # ensure [B,C]
    if logits.ndim == 1:
        logits = logits.unsqueeze(1)
    if not multilabel and logits.shape[1] == 1 and num_classes == 2:
        logits = torch.cat([-logits, logits], dim=1)
    return logits

def compute_classification_metrics(y_true, y_pred, multilabel, num_classes):
    acc = accuracy_score(y_true, y_pred)
    if multilabel:
        avg = 'macro'
        return acc, precision_score(y_true, y_pred, average=avg, zero_division=0), \
                     recall_score(y_true, y_pred, average=avg, zero_division=0), \
                     f1_score(y_true, y_pred, average=avg, zero_division=0)
    if num_classes == 2:
        return acc, precision_score(y_true, y_pred, average='binary', zero_division=0), \
                     recall_score(y_true, y_pred, average='binary', zero_division=0), \
                     f1_score(y_true, y_pred, average='binary', zero_division=0)
    avg = 'macro'
    return acc, precision_score(y_true, y_pred, average=avg, zero_division=0), \
                 recall_score(y_true, y_pred, average=avg, zero_division=0), \
                 f1_score(y_true, y_pred, average=avg, zero_division=0)

def kernel_from_beams(raw_hdr, targ_hdr, fudge_scale=1.0):
    """
    Gaussian2DKernel that turns the RAW restoring beam into the TARGET beam
    on the RAW pixel grid (handles rotation & anisotropy with full WCS).

    Steps:
      • Beam covariances in world coords (radians)
      • C_ker = C_tgt - C_raw  (with optional broadening via fudge_scale)
      • Map to pixel coords via the 2×2 WCS Jacobian (CD/PC)
      • Build Gaussian2DKernel in pixel units
    """  

    ARCSEC = np.deg2rad(1/3600.0)

    def _sigma_from_fwhm_arcsec(theta_as):
        return (theta_as * ARCSEC) / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    def _beam_cov_world(bmaj_as, bmin_as, pa_deg):
        sx = _sigma_from_fwhm_arcsec(bmaj_as)
        sy = _sigma_from_fwhm_arcsec(bmin_as)
        th = np.deg2rad(pa_deg)
        R  = np.array([[ np.cos(th), -np.sin(th)],
                        [ np.sin(th),  np.cos(th)]], float)
        S  = np.diag([sx*sx, sy*sy])
        return R @ S @ R.T

    def _jacobian_rad_per_pix(hdr):
        if 'CD1_1' in hdr:
            CD = np.array([[hdr['CD1_1'], hdr.get('CD1_2', 0.0)],
                            [hdr.get('CD2_1', 0.0), hdr['CD2_2']]], float)
        else:
            pc11 = hdr.get('PC1_1', 1.0); pc12 = hdr.get('PC1_2', 0.0)
            pc21 = hdr.get('PC2_1', 0.0); pc22 = hdr.get('PC2_2', 1.0)
            cd1  = hdr.get('CDELT1', 1.0); cd2  = hdr.get('CDELT2', 1.0)
            CD   = np.array([[pc11,pc12],[pc21,pc22]], float) @ np.diag([cd1, cd2])
        return CD * (np.pi/180.0)

    # Beams in arcsec (+ PA in deg)
    bmaj_r = float(raw_hdr['BMAJ'])*3600.0
    bmin_r = float(raw_hdr['BMIN'])*3600.0
    pa_r   = float(raw_hdr.get('BPA', 0.0))
    bmaj_t = float(targ_hdr['BMAJ'])*3600.0
    bmin_t = float(targ_hdr['BMIN'])*3600.0
    pa_t   = float(targ_hdr.get('BPA', pa_r))

    # World covariances
    C_raw = _beam_cov_world(bmaj_r, bmin_r, pa_r)
    C_tgt = _beam_cov_world(bmaj_t, bmin_t, pa_t) * (fudge_scale**2)
    C_ker = C_tgt - C_raw
    w, V  = np.linalg.eigh(C_ker)
    w     = np.clip(w, 0.0, None)
    C_ker = (V * w) @ V.T
    if not np.any(w > 0):
        return None

    # Map to pixel coords
    J    = _jacobian_rad_per_pix(raw_hdr)
    Jinv = np.linalg.inv(J)
    Cpix = Jinv @ C_ker @ Jinv.T
    w_pix, V_pix = np.linalg.eigh(Cpix)
    w_pix = np.clip(w_pix, 0.0, None)
    if not np.any(w_pix > 0):
        return None
    s_major = float(np.sqrt(w_pix[1]))
    s_minor = float(np.sqrt(w_pix[0]))
    theta   = float(np.arctan2(V_pix[1,1], V_pix[0,1]))

    # Guard against nearly identical beams
    eps = 1e-9
    s_major = max(s_major, eps)
    s_minor = max(s_minor, eps)

    # Reasonable explicit kernel size
    nker = int(np.ceil(8.0 * max(s_major, s_minor))) | 1
    return Gaussian2DKernel(x_stddev=s_major, y_stddev=s_minor, theta=theta,
                            x_size=nker, y_size=nker)

_KER_CACHE = {}
def kernel_from_beams_cached(raw_hdr, targ_hdr, fudge_scale=1.0):
    key = (
        round(float(raw_hdr['BMAJ']),9), round(float(raw_hdr['BMIN']),9), round(float(raw_hdr.get('BPA',0.0)),6),
        round(float(targ_hdr['BMAJ']),9), round(float(targ_hdr['BMIN']),9), round(float(targ_hdr.get('BPA', raw_hdr.get('BPA',0.0))),6),
        tuple(round(float(x),12) for x in (
            raw_hdr.get('CD1_1', raw_hdr.get('CDELT1', 0.0)),
            raw_hdr.get('CD1_2', 0.0),
            raw_hdr.get('CD2_1', 0.0),
            raw_hdr.get('CD2_2', raw_hdr.get('CDELT2', 0.0)),
        )),
        round(float(fudge_scale),6),
    )
    ker = _KER_CACHE.get(key)
    if ker is None:
        ker = kernel_from_beams(raw_hdr, targ_hdr, fudge_scale=fudge_scale)
        _KER_CACHE[key] = ker
    return ker

def kpc_to_arcsec(z, L_kpc):
    L = (L_kpc * u.kpc)
    DA = COSMO.angular_diameter_distance(float(z)).to(u.kpc)  # ensure same length unit
    theta = (L / DA) * u.rad                                  # ratio → radians
    return theta.to(u.arcsec).value


def _beam_solid_angle_sr(hdr):
    """Gaussian beam solid angle in steradians; BMAJ/BMIN in degrees."""
    bmaj = float(hdr['BMAJ']) * (np.pi/180.0)
    bmin = float(hdr['BMIN']) * (np.pi/180.0)
    return (np.pi / (4.0*np.log(2.0))) * bmaj * bmin


@lru_cache(maxsize=None)
def _headers_for_name(base_name: str):
    """
    Return (raw_hdr, t25_hdr, t50_hdr, t100_hdr, pix_native_arcsec, raw_fits_path)
    Builds synthetic T*kpc headers from z if needed/possible.
    """
    base_dir = _first(f"{PSZ2_ROOT}/fits/{base_name}*") or f"{PSZ2_ROOT}/fits/{base_name}"
    raw_path = _first(f"{base_dir}/{os.path.basename(base_dir)}.fits") \
            or _first(f"{base_dir}/{os.path.basename(base_dir)}*.fits")
    if raw_path != _first(f"{base_dir}/{os.path.basename(base_dir)}.fits"):
        print("[rt50 DEBUG] RAW picked:", raw_path)
    if raw_path is None:
        raise FileNotFoundError(f"RAW FITS not found under {base_dir}")

    t25_path  = _first(f"{base_dir}/{base_name}T25kpc*.fits") \
             or _first(f"{PSZ2_ROOT}/classified/T25kpc/*/{base_name}.fits") \
             or _first(f"{PSZ2_ROOT}/classified/T25kpcSUB/*/{base_name}.fits")
    t50_path  = _first(f"{base_dir}/{base_name}T50kpc*.fits") \
             or _first(f"{PSZ2_ROOT}/classified/T50kpc/*/{base_name}.fits") \
             or _first(f"{PSZ2_ROOT}/classified/T50kpcSUB/*/{base_name}.fits")
    t100_path = _first(f"{base_dir}/{base_name}T100kpc*.fits") \
             or _first(f"{PSZ2_ROOT}/classified/T100kpc/*/{base_name}.fits") \
             or _first(f"{PSZ2_ROOT}/classified/T100kpcSUB/*/{base_name}.fits")

    raw_hdr = fits.getheader(raw_path)

    def _hdr_or_synth(tpath, kpc, ref_hdr=None, kpc_ref=None):
        if tpath:
            return fits.getheader(tpath)
        try:
            z = get_z(base_name, raw_hdr)
            return synth_taper_header_from_kpc(raw_hdr, z, kpc, mode="keep_ratio")
        except Exception:
            if (ref_hdr is not None) and (kpc_ref is not None):
                return synth_taper_header_from_ref(raw_hdr, ref_hdr, kpc, kpc_ref, mode="keep_ratio")
            return None

    # Prefer scaling from T50 if only one fixed exists
    t50_hdr  = _hdr_or_synth(t50_path,  50)
    t25_hdr  = _hdr_or_synth(t25_path,  25, ref_hdr=t50_hdr,  kpc_ref=50.0)
    t100_hdr = _hdr_or_synth(t100_path, 100, ref_hdr=t50_hdr, kpc_ref=50.0)

    pix_native = _pixscale_arcsec(raw_hdr)
    return raw_hdr, t25_hdr, t50_hdr, t100_hdr, pix_native, raw_path

def _has_anchors(fn: str, anchor_versions):
    base = _name_base_from_fn(fn)
    try:
        raw_hdr, t25_hdr, t50_hdr, t100_hdr, *_ = _headers_for_name(base)
    except Exception:
        return False
    need = {v.lower() for v in anchor_versions}
    def ok(v):
        if v == "t25kpc":  return t25_hdr  is not None
        if v == "t50kpc":  return t50_hdr  is not None
        if v == "t100kpc": return t100_hdr is not None
        if v == "raw":     return True
        return True
    return all(ok(v) for v in need)

def _has_rt_support(fn: str) -> bool:
    """
    True if we can make rt* at runtime. Prefer a real redshift; otherwise
    fall back to existing fixed anchors.
    """
    base = _name_base_from_fn(fn)
    try:
        raw_hdr, t25_hdr, t50_hdr, t100_hdr, *_ = _headers_for_name(base)
    except Exception:
        return False
    try:
        _ = get_z(base, raw_hdr)
        return True
    except Exception:
        return _has_anchors(fn, _anchor_versions)



def permute_like(x, perm):
    if x is None: return None
    idx = perm.cpu().tolist()
    if isinstance(x, torch.Tensor): return x[perm]
    if isinstance(x, np.ndarray):   return x[idx]
    if isinstance(x, (list, tuple)): return [x[i] for i in idx]
    return x

def relabel(y):
    """
    Convert raw single-class ids to 2-bit multi-label targets [RH, RR].
    RH (52) -> [1,0]
    RR (53) -> [0,1]
    If you ever have 'both', set both bits to 1 *upstream*.
    """
    y = y.long()
    out = torch.zeros((y.shape[0], 2), dtype=torch.float32, device=y.device)
    out[:, 0] = (y == 52).float()  # RH
    out[:, 1] = (y == 53).float()  # RR
    return out    

def _background_ring_mask(h, w, inner=64, pad=8):
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    cy, cx = h//2, w//2
    half = inner//2
    # guard band
    mask_c = (yy >= cy-(half+pad)) & (yy <= cy+(half+pad)) & (xx >= cx-(half+pad)) & (xx <= cx+(half+pad))
    return ~mask_c

def _radial_bin(u, v, values, nbins=36, rmin=None, rmax=None, logspace=True, stat='median'):
    """Radial bin a 2D field in uv. values can be real/abs/complex; returns r_centers, stat(values)."""
    R = np.sqrt(u*u + v*v)
    m = np.isfinite(values)
    if rmin is None: rmin = np.nanpercentile(R[m], 1.0)
    if rmax is None: rmax = np.nanmax(R[m])
    edges = (np.geomspace(rmin, rmax, nbins+1) if logspace else np.linspace(rmin, rmax, nbins+1))
    idx = np.digitize(R.ravel(), edges) - 1
    out = []
    for i in range(nbins):
        sel = (idx == i) & m.ravel()
        if not np.any(sel):
            out.append(np.nan); continue
        vals = values.ravel()[sel]
        if stat == 'median':
            out.append(np.nanmedian(vals))
        elif stat == 'mean':
            out.append(np.nanmean(vals))
        elif stat == 'abs-mean':
            out.append(np.nanmean(np.abs(vals)))
        else:
            out.append(np.nanmedian(vals))
    rc = 0.5*(edges[:-1] + edges[1:])
    return rc, np.asarray(out)

def _robust_sigma(x2d):
    x = torch.as_tensor(x2d, dtype=torch.float32)
    med = x.median()
    return 1.4826 * (x - med).abs().median()

def _per_image_percentile_stretch(x2d, lo=60, hi=95):
    t = torch.as_tensor(x2d, dtype=torch.float32)
    pl = torch.quantile(t.reshape(-1), lo/100.0)
    ph = torch.quantile(t.reshape(-1), hi/100.0)
    y = (t - pl) / (ph - pl + 1e-6)
    return y.clamp(0, 1)

def as_index_labels(y: torch.Tensor) -> torch.Tensor:
    return y.argmax(dim=1) if y.ndim > 1 else y

def convolve_to_target(raw_arr, raw_hdr, target_hdr, fudge_scale=1.0):
    """
    Convolve RAW image (on RAW grid) to TARGET restoring beam on the RAW grid,
    then convert Jy/beam_native → Jy/beam_target.
    """

    ker = kernel_from_beams_cached(raw_hdr, target_hdr, fudge_scale=fudge_scale)
    if ker is None:
        out = np.asarray(raw_arr, float).copy()
    else:
        out = convolve_fft(np.asarray(raw_arr, float), ker,
                           boundary='fill', fill_value=np.nan,
                           nan_treatment='interpolate', normalize_kernel=True,
                           psf_pad=True, fft_pad=True, allow_huge=True)
    try:
        out *= (_beam_solid_angle_sr(target_hdr) / _beam_solid_angle_sr(raw_hdr))
    except Exception:
        pass
    return out

def radial_tukey(ny, nx, alpha=0.35):
    y = np.linspace(-1, 1, ny); x = np.linspace(-1, 1, nx)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X*X + Y*Y)
    w = np.ones_like(r)
    r0 = 1.0 - alpha
    m  = (r > r0) & (r < 1.0)
    w[m]  = 0.5 * (1.0 + np.cos(np.pi*(r[m]-r0)/max(alpha,1e-9)))
    w[r>=1.0] = 0.0
    return w

def radial_hann(ny, nx):
    y = np.linspace(-1, 1, ny); x = np.linspace(-1, 1, nx)
    X, Y = np.meshgrid(x, y); r = np.sqrt(X*X + Y*Y)
    w = np.zeros_like(r); m = (r <= 1.0)
    w[m] = 0.5 * (1.0 + np.cos(np.pi * r[m]))
    return w


def collapse_logits(logits, num_classes, multilabel):
    if logits.ndim == 4:
        logits = torch.nn.functional.adaptive_avg_pool2d(logits, (1, 1)).squeeze(-1).squeeze(-1)
    elif logits.ndim == 3:
        logits = logits.mean(dim=-1)
    if logits.ndim == 1:
        logits = logits.unsqueeze(1)
    if not multilabel and logits.shape[1] == 1 and num_classes == 2:
        logits = torch.cat([-logits, logits], dim=1)
    return logits

def reproject_like(arr, src_hdr, dst_hdr):
    """
    Reproject a 2-D image from src_hdr WCS to dst_hdr WCS.

    If the 'reproject' package is available and both headers contain a valid
    2-D celestial WCS, use bilinear interpolation. Otherwise fall back to a
    center-alignment translation (keeps shape, best-effort alignment).
    """
    try:
        from reproject import reproject_interp
        HAVE_REPROJECT = True
    except Exception:
        HAVE_REPROJECT = False
    try:
        from scipy.ndimage import shift as _imgshift
    except Exception:
        _imgshift = None

    if arr is None or src_hdr is None or dst_hdr is None:
        return None

    # FAST PATH: identical grid → return as-is
    def _same_pixel_grid(h1, h2, atol=1e-12):
        for k in ('NAXIS1','NAXIS2','CTYPE1','CTYPE2'):
            if (h1.get(k) != h2.get(k)):
                return False
        def _cd(h):
            if 'CD1_1' in h:
                return (float(h['CD1_1']), float(h.get('CD1_2',0.0)),
                        float(h.get('CD2_1',0.0)), float(h['CD2_2']))
            pc11=h.get('PC1_1',1.0); pc12=h.get('PC1_2',0.0)
            pc21=h.get('PC2_1',0.0); pc22=h.get('PC2_2',1.0)
            c1=h.get('CDELT1',-1.0); c2=h.get('CDELT2', 1.0)
            M = np.array([[pc11,pc12],[pc21,pc22]],float) @ np.diag([c1,c2])
            return (float(M[0,0]), float(M[0,1]), float(M[1,0]), float(M[1,1]))
        if not np.allclose(_cd(h1), _cd(h2), atol=atol): return False
        if not np.allclose([h1.get('CRPIX1'),h1.get('CRPIX2')],
                            [h2.get('CRPIX1'),h2.get('CRPIX2')], atol=1e-9): return False
        if not np.allclose([h1.get('CRVAL1'),h1.get('CRVAL2')],
                            [h2.get('CRVAL1'),h2.get('CRVAL2')], atol=1e-9): return False
        return True

    if _same_pixel_grid(src_hdr, dst_hdr):
        return np.asarray(arr, float)

    # Try full WCS reprojection
    try:
        w_src = WCS(src_hdr).celestial
        w_dst = WCS(dst_hdr).celestial
    except Exception:
        w_src = w_dst = None

    if HAVE_REPROJECT and (w_src is not None) and (w_dst is not None):
        ny_out = int(dst_hdr['NAXIS2']); nx_out = int(dst_hdr['NAXIS1'])
        reproj, _ = reproject_interp((arr, w_src), w_dst,
                                     shape_out=(ny_out, nx_out),
                                     order='bilinear')
        return reproj.astype(float)

    # Fallback: align image centers via subpixel shift
    if (w_src is None) or (w_dst is None) or (_imgshift is None):
        return np.asarray(arr, float)

    ny, nx = arr.shape
    (ra, dec) = w_src.wcs_pix2world([[nx/2.0, ny/2.0]], 0)[0]
    (x_dst, y_dst) = w_dst.wcs_world2pix([[ra, dec]], 0)[0]
    dx = (float(dst_hdr['NAXIS1'])/2.0) - x_dst
    dy = (float(dst_hdr['NAXIS2'])/2.0) - y_dst
    return _imgshift(arr, shift=(dy, dx), order=1, mode="nearest").astype(float)

def auto_fudge_scale(raw_img, raw_hdr, targ_hdr, T_img, s_grid=None, nbins=36):
    if s_grid is None:
        s_grid = np.linspace(1.00, 1.20, 11)
    U,V,FT,AT = image_to_vis(T_img, targ_hdr, beam_hdr=targ_hdr)
    best_s, best_cost = 1.0, np.inf
    for s in s_grid:
        RT_native = convolve_to_target(raw_img, raw_hdr, targ_hdr, fudge_scale=s)
        RT_on_t   = reproject_like(RT_native, raw_hdr, targ_hdr)
        _,_,FR,AR = image_to_vis(RT_on_t, targ_hdr, beam_hdr=targ_hdr)
        # radial medians
        r, aT = _radial_bin(U,V,AT, nbins=nbins, stat='median')
        _, aR = _radial_bin(U,V,AR, nbins=nbins, stat='median')
        # simple coherence mask
        Rgrid = np.sqrt(U*U + V*V)
        edges = np.geomspace(np.nanpercentile(Rgrid,1.0), np.nanmax(Rgrid), nbins+1)
        coh = []
        for i in range(nbins):
            m = (Rgrid>=edges[i]) & (Rgrid<edges[i+1])
            if not np.any(m): coh.append(np.nan); continue
            num  = np.nanmean(FT[m]*np.conj(FR[m]))
            den1 = np.nanmean(np.abs(FT[m])**2)
            den2 = np.nanmean(np.abs(FR[m])**2)
            coh.append(np.abs(num)/np.sqrt(den1*den2))
        coh = np.asarray(coh)
        good = (coh > 0.6) & np.isfinite(aT) & np.isfinite(aR)
        if not np.any(good): 
            continue
        ratio = aT[good] / (aR[good] + 1e-12)
        cost = np.nanmedian(np.abs(np.log(ratio)))
        if cost < best_cost:
            best_cost, best_s = float(cost), float(s)
    return best_s


def apply_taper_to_tensor(
    imgs, mode, filenames,
    crop_size=(1,512,512), downsample_size=(1,128,128),
    percentile_lo=60, percentile_hi=95,
    do_stretch=True, use_asinh=True,
    require_fixed_header=False,
    ref_sigma_map=None, bg_inner=64,
    debug_dir=None
):
    """
    Build runtime-tapered planes (rtXX) by:
      RAW → (PSF-match to target restoring beam on RAW grid) → (optional uv-taper)
      → reproject onto target grid → crop/resize → percentile stretch → asinh.
    Returns (stack[B,1,H,W], keep_mask[B], kept_fns[list], skipped[list]).
    """

    mode = str(mode).lower()
    m = re.fullmatch(r'rt(\d+)', mode)
    want_kpc = int(m.group(1)) if m else None
    if want_kpc is None:
        # nothing special requested: just ensure [B,1,H,W]
        keep_mask = torch.ones(len(filenames), dtype=torch.bool)
        return (imgs if imgs.dim()==4 else imgs.unsqueeze(1)), keep_mask, list(map(str, filenames)), []

    device = imgs.device if torch.is_tensor(imgs) else torch.device('cpu')
    dtype  = imgs.dtype  if torch.is_tensor(imgs) else torch.float32
    Hout, Wout = downsample_size[-2], downsample_size[-1]

    out, kept_fns, kept_flags, skipped = [], [], [], []

    for base in map(str, filenames):
        try:
            raw_hdr, t25_hdr, t50_hdr, t100_hdr, pix_native_as, raw_path = _headers_for_name(base)
        except Exception as e:
            skipped.append(base); kept_flags.append(False); continue

        # Choose or synthesize the target header
        targ_hdr = None
        if   want_kpc == 25:   targ_hdr = t25_hdr
        elif want_kpc == 50:   targ_hdr = t50_hdr
        elif want_kpc == 100:  targ_hdr = t100_hdr
        else:
            # interpolate between existing fixed tapers if possible
            def _interp_hdr(k_lo, h_lo, k_hi, h_hi, k_want):
                if h_lo is None and h_hi is None:
                    return None
                if h_lo is None or h_hi is None:
                    return (h_lo or h_hi).copy()
                w = (k_want - k_lo) / float(k_hi - k_lo)
                outH = h_lo.copy()
                for key in ("BMAJ", "BMIN"):
                    v_lo = float(h_lo[key]); v_hi = float(h_hi[key])
                    outH[key] = v_lo*(1.0-w) + v_hi*w
                bpa_lo = float(h_lo.get("BPA", h_hi.get("BPA", 0.0)))
                bpa_hi = float(h_hi.get("BPA", h_lo.get("BPA", 0.0)))
                outH["BPA"] = bpa_lo if (k_want - k_lo) <= (k_hi - k_want) else bpa_hi
                # keep the RAW WCS so reproject_like(raw→targ) can be a no-op when grids match
                for k in ('CTYPE1','CTYPE2','CRVAL1','CRVAL2','CRPIX1','CRPIX2',
                          'CDELT1','CDELT2','CD1_1','CD1_2','CD2_1','CD2_2',
                          'PC1_1','PC1_2','PC2_1','PC2_2','NAXIS1','NAXIS2'):
                    if k in raw_hdr: outH[k] = raw_hdr[k]
                return outH
            if want_kpc < 50:
                targ_hdr = _interp_hdr(25, t25_hdr, 50, t50_hdr, want_kpc)
            elif want_kpc < 100:
                targ_hdr = _interp_hdr(50, t50_hdr, 100, t100_hdr, want_kpc)
            else:
                targ_hdr = _interp_hdr(50, t50_hdr, 100, t100_hdr, want_kpc)

        # try final synthesis if needed (from redshift or scaled ref header)
        if targ_hdr is None:
            try:
                z_here  = get_z(base, raw_hdr)
                targ_hdr = synth_taper_header_from_kpc(raw_hdr, z_here, want_kpc, mode="keep_ratio")
            except Exception:
                # scale from T50 if available
                if t50_hdr is not None:
                    try:
                        targ_hdr = synth_taper_header_from_ref(raw_hdr, t50_hdr, want_kpc, kpc_ref=50.0, mode="keep_ratio")
                    except Exception:
                        targ_hdr = None

        # Enforce parity (optional)
        if require_fixed_header and want_kpc in (25, 50, 100) and targ_hdr is None:
            skipped.append(base); kept_flags.append(False); continue
        if (t25_hdr is None) and (t50_hdr is None) and (t100_hdr is None) and (targ_hdr is None):
            skipped.append(base); kept_flags.append(False); continue

        # 1) Load RAW map (native grid)
        try:
            raw_native = np.squeeze(fits.getdata(raw_path)).astype(float)
        except Exception:
            skipped.append(base); kept_flags.append(False); continue
        raw_native = np.nan_to_num(raw_native, copy=False)

        # 2) Anti-alias (only if we’ll shrink to Hout×Wout)
        try:
            ds = int(round(crop_size[-2] / float(Hout)))
            if ds > 1:
                raw_pref = gaussian_filter(raw_native, sigma=0.5*ds, mode='nearest')
            else:
                raw_pref = raw_native
        except Exception:
            raw_pref = raw_native

        # 3) PSF-match RAW → TARGET beam on RAW grid, with optional global fudge
        targ_hdr_eff = targ_hdr
        fudge = float(FUDGE_GLOBAL)
        if os.getenv("RT_AUTO_FUDGE", "1") == "1" and (targ_hdr_eff is not None):
            try:
                tpath = _first(f"{os.path.dirname(raw_path)}/{_name_base_from_fn(base)}T{want_kpc}kpc*.fits")
                if tpath:
                    T_img = np.squeeze(fits.getdata(tpath)).astype(float)
                    fudge = auto_fudge_scale(
                        raw_pref, raw_hdr, targ_hdr_eff, T_img,
                        s_grid=np.linspace(1.00, 1.20, 11), nbins=48
                    )
            except Exception:
                pass
        matched_native = convolve_to_target(raw_pref, raw_hdr, targ_hdr_eff or raw_hdr, fudge_scale=fudge)

        # 4) Optional uv-taper (disabled by default)
        if APPLY_UV_TAPER and (UV_TAPER_FRAC > 0):
            try:
                z_here = get_z(base, raw_hdr)
                theta_as = kpc_to_arcsec(z_here, float(want_kpc))  # arcsec
                matched_native = apply_uv_gaussian_taper(matched_native, raw_hdr, theta_as * UV_TAPER_FRAC, pad_factor=2)
            except Exception:
                pass

        # 5) Reproject RAW→TARGET (image we just convolved) onto TARGET grid
        matched_on_t = reproject_like(matched_native, raw_hdr, targ_hdr_eff or raw_hdr)

        # 6) Crop+resize from TARGET grid → (Hout,Wout) using the same pipeline tool
        t = torch.from_numpy(np.nan_to_num(matched_on_t, copy=False)).float().unsqueeze(0)  # [1,H,W]
        crop_eff = _effective_crop_on_raw(raw_hdr, targ_hdr_eff, crop_size)
        formatted = apply_formatting(t, crop_size=crop_eff,
                                     downsample_size=(1, Hout, Wout)).squeeze(0)   # [1,Hout,Wout]

        # 7) Percentile stretch + optional asinh
        if do_stretch:
            stretched = _per_image_percentile_stretch(formatted.squeeze(0), percentile_lo, percentile_hi).unsqueeze(0)
        else:
            stretched = formatted
        if use_asinh:
            stretched = torch.asinh(10.0 * stretched) / math.asinh(10.0)

        # 8) Optional noise match to a reference sigma map
        if (ref_sigma_map is not None) and (base in ref_sigma_map):
            mask = _background_ring_mask(Hout, Wout, inner=bg_inner)
            sig_fake = _robust_sigma(stretched.squeeze(0)[mask])
            sig_want = torch.tensor(ref_sigma_map[base], dtype=stretched.dtype)
            add = torch.clamp(sig_want - sig_fake, min=0.0)
            if add > 1e-8:
                noise = torch.randn_like(stretched)
                s = stretched.clone()
                s[:, 0][mask] = (s[:, 0][mask] + noise[:, 0][mask]*add).clamp(0, 1)
                stretched = s

        out.append(stretched)
        kept_fns.append(base)
        kept_flags.append(True)

        # Optional debug figure per source
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            fig, ax = plt.subplots(1,2, figsize=(6,3))
            ax[0].imshow(formatted.squeeze(0).cpu().numpy(), cmap='viridis', origin='lower'); ax[0].axis('off'); ax[0].set_title('PSF-matched')
            ax[1].imshow(stretched.squeeze(0).cpu().numpy(), cmap='viridis', origin='lower'); ax[1].axis('off'); ax[1].set_title('final')
            fig.suptitle(base); fig.tight_layout()
            tag = f"rt{want_kpc}"
            fig.savefig(os.path.join(debug_dir, f"{base}_{tag}{versions}.png"), dpi=140)
            plt.close(fig)

    keep_mask = torch.tensor(kept_flags, dtype=torch.bool)
    out = torch.stack(out, dim=0).to(device=device, dtype=dtype) if out else torch.empty((0,1,Hout,Wout), device=device, dtype=dtype)
    return out, keep_mask, kept_fns, skipped

def replicate_list(x, n):
    return [v for v in x for _ in range(int(n))]

def late_augment(images, labels, filenames=None, *, st_aug=False):
    """
    Apply your normal augmentations AFTER tapering.
    Returns (imgs_aug, labels_aug, filenames_aug).
    If images is empty, this is a no-op.
    """
    if images is None or (isinstance(images, torch.Tensor) and images.numel() == 0):
        return images, labels, filenames
    imgs_aug, labels_aug = augment_images(images, labels, ST_augmentation=st_aug)
    n_aug = imgs_aug.size(0) // max(1, images.size(0))
    if filenames is not None:
        filenames = replicate_list(filenames, n_aug)
    return imgs_aug, labels_aug, filenames


def _name_base_from_fn(fn):
    stem = Path(str(fn)).stem
    return stem.split('T', 1)[0]

def _z_from_meta(name):                 # base like "PSZ2G192.18+56.12"
    return CLUSTER_META.get(name)

# put near your other helpers
def _ensure_4d(x: torch.Tensor) -> torch.Tensor:
    """Make images [B,C,H,W]. If [B,T,1,H,W], fold T into C."""
    if x is None:
        return x
    if x.dim() == 5:
        # [B, T, 1, H, W]  ->  [B, T, H, W] -> treat T as channels
        return x.flatten(1, 2)  # fold_T_axis does the same; this is inline & fast
    if x.dim() == 3:
        return x.unsqueeze(1)
    return x  # already [B,C,H,W]

def _coerce_float(v):
    try:
        x = float(v); 
        return x if np.isfinite(x) else None
    except Exception:
        pass
    if isinstance(v, (bytes, bytearray)):
        try: v = v.decode('utf-8', 'ignore')
        except Exception: return None
    if isinstance(v, str):
        m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', v)
        if m:
            try: return float(m.group(0))
            except Exception: return None
    return None

def get_z(name, hdr_primary):
    """CSV → header keys → broad scan for redshift; raises if none usable."""
    base = _name_base_from_fn(name)
    z = _z_from_meta(base)
    if z is not None: 
        return float(z)
    CAND_KEYS = ('REDSHIFT','Z','Z_CL','ZSPEC','Z_PHOT','Z_BEST','Z_MEAN','ZPLANCK','Z_PSZ2')
    for k in CAND_KEYS:
        if k in hdr_primary:
            z = _coerce_float(hdr_primary[k])
            if z is not None and 0.0 < z < 5.0: 
                return z
    for k,v in hdr_primary.items():
        ku = str(k).upper().replace('HIERARCH ', '')
        if 'ZERO' in ku or ku.startswith('DZ'):
            continue
        if ('REDSHIFT' in ku) or ('Z' == ku) or ku.startswith('Z_') or ku.endswith('_Z'):
            z = _coerce_float(v)
            if z is not None and 0.0 < z < 5.0: 
                return z
    raise KeyError(f"No usable redshift for {_name_base_from_fn(name)}")

def kpc_to_arcsec(z, L_kpc):
    return ((L_kpc * u.kpc) / COSMO.angular_diameter_distance(float(z))).to(u.arcsec).value

def synth_taper_header_from_kpc(raw_hdr, z, L_kpc, mode="keep_ratio"):
    """Target header on RAW grid for desired FWHM (kpc) at redshift z."""
    phi_as = float(kpc_to_arcsec(z, L_kpc))  # desired geometric-mean FWHM [arcsec]
    bmaj_r = abs(float(raw_hdr['BMAJ'])) * 3600.0
    bmin_r = abs(float(raw_hdr['BMIN'])) * 3600.0
    pa_r   = float(raw_hdr.get('BPA', 0.0))
    phi_as = max(phi_as, np.sqrt(bmaj_r * bmin_r))  # never sharpen

    if mode == "circular":
        bmaj_t = bmin_t = phi_as; pa_t = 0.0
    else:
        r = bmaj_r / bmin_r
        bmin_t = phi_as / np.sqrt(r)
        bmaj_t = phi_as * np.sqrt(r)
        pa_t   = pa_r

    thdr = fits.Header()
    thdr['BMAJ'] = bmaj_t / 3600.0; thdr['BMIN'] = bmin_t / 3600.0; thdr['BPA'] = pa_t
    for k in ('CTYPE1','CTYPE2','CRVAL1','CRVAL2','CRPIX1','CRPIX2',
              'CDELT1','CDELT2','CD1_1','CD1_2','CD2_1','CD2_2',
              'PC1_1','PC1_2','PC2_1','PC2_2','NAXIS1','NAXIS2'):
        if k in raw_hdr: thdr[k] = raw_hdr[k]
    return thdr

def synth_taper_header_from_ref(raw_hdr, ref_hdr, kpc_target, kpc_ref=50.0, mode="keep_ratio"):
    """Scale an existing fixed header (e.g., T50) to a new kpc target."""
    scale = float(kpc_target) / float(kpc_ref)
    bmaj = abs(float(ref_hdr['BMAJ'])) * 3600.0
    bmin = abs(float(ref_hdr['BMIN'])) * 3600.0
    pa   = float(ref_hdr.get('BPA', raw_hdr.get('BPA', 0.0)))
    phi  = np.sqrt(bmaj*bmin)
    phi_t = scale * phi
    if mode == "circular":
        bmaj_t = bmin_t = phi_t; pa_t = 0.0
    else:
        r = bmaj / bmin; bmin_t = phi_t/np.sqrt(r); bmaj_t = phi_t*np.sqrt(r); pa_t = pa
    phi_raw = np.sqrt((abs(float(raw_hdr['BMAJ']))*3600.0)*(abs(float(raw_hdr['BMIN']))*3600.0))
    if np.sqrt(bmaj_t*bmin_t) < phi_raw:
        f = phi_raw/np.sqrt(bmaj_t*bmin_t); bmaj_t *= f; bmin_t *= f
    thdr = fits.Header()
    thdr['BMAJ'] = bmaj_t/3600.0; thdr['BMIN'] = bmin_t/3600.0; thdr['BPA'] = pa_t
    for k in ('CTYPE1','CTYPE2','CRVAL1','CRVAL2','CRPIX1','CRPIX2',
              'CDELT1','CDELT2','CD1_1','CD1_2','CD2_1','CD2_2',
              'PC1_1','PC1_2','PC2_1','PC2_2','NAXIS1','NAXIS2'):
        if k in raw_hdr: thdr[k] = raw_hdr[k]
    return thdr

def _sigma_from_fwhm_arcsec(theta_as):
    return (theta_as * ARCSEC) / (2.0 * np.sqrt(2.0 * np.log(2.0)))

def _make_uv_gaussian_weight(nx, ny, dx, dy, theta_as):
    sigma_th = _sigma_from_fwhm_arcsec(theta_as)
    u = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    v = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    U, V = np.meshgrid(u, v)
    return np.exp(-2.0 * (np.pi**2) * (sigma_th**2) * (U*U + V*V))

def _cdelt_deg(hdr, axis: int) -> float:
    key = f"CDELT{axis}"
    if key in hdr and np.isfinite(hdr[key]): return float(abs(hdr[key]))
    if 'CD1_1' in hdr:
        row = (hdr['CD1_1'], hdr.get('CD1_2',0.0)) if axis==1 else (hdr.get('CD2_1',0.0), hdr['CD2_2'])
        return float(np.hypot(*row))
    pc11,pc12 = hdr.get('PC1_1',1.0), hdr.get('PC1_2',0.0)
    pc21,pc22 = hdr.get('PC2_1',0.0), hdr.get('PC2_2',1.0)
    cd1,cd2   = hdr.get('CDELT1',1.0), hdr.get('CDELT2',1.0)
    M = np.array([[pc11,pc12],[pc21,pc22]], float) @ np.diag([cd1,cd2])
    row = M[0] if axis==1 else M[1]
    return float(np.hypot(row[0], row[1]))

def _effective_crop_on_raw(raw_hdr, targ_hdr, crop_size):
    Hc_target, Wc_target = crop_size[-2], crop_size[-1]
    try:
        s_raw = _cdelt_deg(raw_hdr, 1)          # deg/pix on RAW grid
        s_tgt = _cdelt_deg(targ_hdr, 1) if targ_hdr is not None else s_raw
        # ⬇⬇ flip the ratio so RAW collects enough pixels to cover the same angular FoV
        scale = (s_tgt / s_raw) if (s_raw and np.isfinite(s_raw)) else 1.0
    except Exception:
        scale = 1.0
    Hc = max(16, int(round(Hc_target * scale)))
    Wc = max(16, int(round(Wc_target * scale)))
    return (1, Hc, Wc)

def apply_uv_gaussian_taper(img, hdr_img, theta_as, pad_factor=2):
    A = np.where(np.isfinite(img), img, 0.0).astype(float)
    ny0, nx0 = A.shape
    if pad_factor > 1:
        ny, nx = int(ny0*pad_factor), int(nx0*pad_factor)
        py = (ny-ny0)//2; px = (nx-nx0)//2
        A = np.pad(A, ((py,ny-ny0-py),(px,nx-nx0-px)), mode='constant', constant_values=0.0)
        crop = (slice(py,py+ny0), slice(px,px+nx0))
    else:
        ny, nx = ny0, nx0
        crop = (slice(0,ny0), slice(0,nx0))
    dx = abs(_cdelt_deg(hdr_img, 1)) * np.pi/180.0
    dy = abs(_cdelt_deg(hdr_img, 2)) * np.pi/180.0
    F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(A))) * (dx * dy)
    W = _make_uv_gaussian_weight(nx, ny, dx, dy, theta_as)
    At = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(F*W))).real / (dx*dy)
    return At[crop]