"""Metrologia ótica: luz estruturada (franjas com desfasamento), desembrulhamento de fase,
deflectometria, varrimento vertical de luz branca (profundidade por foco) e speckle.
Implementações próprias de técnicas publicadas na literatura."""
import numpy as np
from scipy import ndimage


# --------------------------------------------------------------------------
# Padrões
# --------------------------------------------------------------------------
def fringe_patterns(width, height, pitch, steps=4, direction="vertical", phase_offset_deg=0.0,
                    amplitude=0.5, background=0.5):
    if pitch < 2:
        raise ValueError("O passo (pitch) tem de ser >= 2.")
    if not 1 <= steps <= 360:
        raise ValueError("O número de passos tem de estar em [1, 360].")
    x = np.arange(width) if direction == "vertical" else np.arange(height)
    imgs = []
    for k in range(steps):
        d = 2 * np.pi * k / steps + np.radians(phase_offset_deg)
        line = background + amplitude * np.cos(2 * np.pi * x / pitch + d)
        img = np.tile(line, (height, 1)) if direction == "vertical" else np.tile(line[:, None], (1, width))
        imgs.append(np.clip(img * 255, 0, 255).astype(np.uint8))
    return imgs


def gray_code_patterns(width, height, bits=None):
    bits = bits or int(np.ceil(np.log2(width)))
    x = np.arange(width)
    g = x ^ (x >> 1)
    imgs = []
    for b in range(bits - 1, -1, -1):
        line = ((g >> b) & 1).astype(np.uint8) * 255
        imgs.append(np.tile(line, (height, 1)))
    return imgs


# --------------------------------------------------------------------------
# Fase embrulhada
# --------------------------------------------------------------------------
def wrapped_phase(images, shifts_deg=None):
    """N passos: devolve (fase embrulhada [-π,π], modulação, fundo)."""
    I = np.stack([np.asarray(i, float) for i in images])
    if I.ndim == 4:
        I = I.mean(axis=3)
    N = len(I)
    if N < 3:
        raise ValueError("São precisas pelo menos 3 imagens com desfasamento.")
    d = np.radians(shifts_deg) if shifts_deg is not None else 2 * np.pi * np.arange(N) / N
    S = np.tensordot(np.sin(d), I, axes=1)
    C = np.tensordot(np.cos(d), I, axes=1)
    phi = np.arctan2(-S, C)
    mod = 2.0 / N * np.sqrt(S ** 2 + C ** 2)
    bg = I.mean(axis=0)
    return phi, mod, bg


def modulation_mask(mod, threshold_ratio=0.1):
    return mod > threshold_ratio * mod.max()


# --------------------------------------------------------------------------
# Desembrulhamento
# --------------------------------------------------------------------------
def unwrap_left_to_right(phi, mask=None):
    """Desembrulhamento sequencial: primeira coluna de cima para baixo e depois cada linha."""
    p = phi.copy()
    p[:, 0] = np.unwrap(p[:, 0])
    p = np.unwrap(p, axis=1)
    if mask is not None:
        p = np.where(mask, p, np.nan)
    return p


def unwrap_quality_guided(phi, mask=None):
    """Guiado por qualidade (ordenação por fiabilidade das arestas)."""
    from skimage.restoration import unwrap_phase
    if mask is not None:
        m = np.ma.array(phi, mask=~mask)
        out = unwrap_phase(m)
        return np.where(mask, out.filled(np.nan), np.nan)
    return unwrap_phase(phi)


def unwrap_multifrequency(phases, pitches):
    """Desembrulhamento temporal hierárquico.
    phases: fases embrulhadas por ordem de passo DECRESCENTE (a primeira deve ter ≤1 período no campo)."""
    order = np.argsort(pitches)[::-1]
    phases = [phases[i] for i in order]
    pitches = [pitches[i] for i in order]
    Phi = np.mod(phases[0], 2 * np.pi)
    for k in range(1, len(phases)):
        ratio = pitches[k - 1] / pitches[k]
        est = Phi * ratio
        n = np.round((est - phases[k]) / (2 * np.pi))
        Phi = phases[k] + 2 * np.pi * n
    return Phi


def unwrap_gray_code(phi, gray_images, threshold=None):
    """Combina fase embrulhada com código de Gray (ordem de franja)."""
    bits = [np.asarray(g, float) for g in gray_images]
    thr = threshold if threshold is not None else [b.mean() for b in bits]
    if np.isscalar(thr):
        thr = [thr] * len(bits)
    g = np.zeros(phi.shape, int)
    for b, t in zip(bits, thr):
        g = (g << 1) | (b > t)
    # Gray → binário
    n = g.copy()
    shift = g >> 1
    while shift.any():
        n ^= shift
        shift >>= 1
    return phi + 2 * np.pi * n


# --------------------------------------------------------------------------
# Fase → altura / 3D
# --------------------------------------------------------------------------
def phase_to_height(Phi, Phi_ref=None, scale_mm_per_rad=1.0, remove_tilt=True, mask=None):
    D = Phi - (Phi_ref if Phi_ref is not None else 0)
    if remove_tilt:
        D = remove_plane(D, mask)
    return D * scale_mm_per_rad


def remove_plane(Z, mask=None):
    rows, cols = Z.shape
    Y, X = np.mgrid[:rows, :cols]
    m = np.isfinite(Z) if mask is None else (mask & np.isfinite(Z))
    if m.sum() < 3:
        return Z
    A = np.column_stack([X[m], Y[m], np.ones(m.sum())])
    coef, *_ = np.linalg.lstsq(A, Z[m], rcond=None)
    return Z - (coef[0] * X + coef[1] * Y + coef[2])


def depth_to_points(Z, px_mm=1.0, mask=None, color=None):
    rows, cols = Z.shape
    Y, X = np.mgrid[:rows, :cols]
    m = np.isfinite(Z) if mask is None else (mask & np.isfinite(Z))
    pts = np.column_stack([X[m] * px_mm, (rows - Y[m]) * px_mm, Z[m]])
    cols_ = None if color is None else np.asarray(color)[m]
    return pts, cols_


def save_ply(path, pts, colors=None):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i, p in enumerate(pts):
            if colors is not None:
                c = colors[i] if np.ndim(colors[i]) else (colors[i],) * 3
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
            else:
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")


# --------------------------------------------------------------------------
# Deflectometria: fases X e Y → declives → integração (Frankot–Chellappa)
# --------------------------------------------------------------------------
def frankot_chellappa(p, q):
    rows, cols = p.shape
    wx, wy = np.meshgrid(np.fft.fftfreq(cols) * 2 * np.pi, np.fft.fftfreq(rows) * 2 * np.pi)
    P, Q = np.fft.fft2(np.nan_to_num(p)), np.fft.fft2(np.nan_to_num(q))
    denom = wx ** 2 + wy ** 2
    denom[0, 0] = 1
    Z = (-1j * wx * P - 1j * wy * Q) / denom
    Z[0, 0] = 0
    return np.real(np.fft.ifft2(Z))


def deflectometry(phase_x, phase_y, gain=1.0):
    # o declive da superfície é proporcional à variação da fase observada
    p = remove_plane(phase_x) * gain
    q = remove_plane(phase_y) * gain
    return frankot_chellappa(p, q)


# --------------------------------------------------------------------------
# Varrimento vertical de luz branca / profundidade por foco
# --------------------------------------------------------------------------
def depth_from_focus(stack, z_positions, window=7):
    I = np.stack([np.asarray(i, float) if np.ndim(i) == 2 else np.asarray(i, float).mean(2) for i in stack])
    F = np.stack([ndimage.uniform_filter(ndimage.laplace(ndimage.gaussian_filter(im, 1)) ** 2, window)
                  for im in I])
    k = np.argmax(F, axis=0)
    z = np.asarray(z_positions, float)
    # interpolação parabólica em torno do máximo
    k0 = np.clip(k, 1, len(z) - 2)
    r, c = np.indices(k.shape)
    f0, f1, f2 = F[k0 - 1, r, c], F[k0, r, c], F[k0 + 1, r, c]
    den = f0 - 2 * f1 + f2
    off = np.where(np.abs(den) > 1e-12, 0.5 * (f0 - f2) / den, 0)
    off = np.clip(off, -1, 1)
    dz = np.gradient(z)[k0]
    Z = z[k0] + off * dz
    conf = F.max(axis=0)
    return Z, conf


def coherence_scan(stack, z_positions):
    """Luz branca (interferometria de coerência): pico do envelope por pixel."""
    I = np.stack([np.asarray(i, float) for i in stack])
    I = I - I.mean(axis=0)
    from scipy.signal import hilbert
    env = np.abs(hilbert(I, axis=0))
    k = np.argmax(env, axis=0)
    return np.asarray(z_positions)[k], env.max(axis=0)


# --------------------------------------------------------------------------
# Speckle
# --------------------------------------------------------------------------
def speckle_contrast(img, window=7):
    I = np.asarray(img, float)
    m = ndimage.uniform_filter(I, window)
    m2 = ndimage.uniform_filter(I ** 2, window)
    sd = np.sqrt(np.maximum(m2 - m ** 2, 0))
    return sd / np.maximum(m, 1e-9)


def speckle_reduce(img, method="median", size=5):
    I = np.asarray(img)
    if method == "median":
        return ndimage.median_filter(I, size)
    if method == "lee":
        I = I.astype(float)
        m = ndimage.uniform_filter(I, size)
        v = ndimage.uniform_filter(I ** 2, size) - m ** 2
        noise = v.mean()
        k = v / np.maximum(v + noise, 1e-9)
        return m + k * (I - m)
    import cv2
    u8 = np.clip(I, 0, 255).astype(np.uint8)
    return cv2.fastNlMeansDenoising(u8, None, h=size * 2)


# --------------------------------------------------------------------------
# Dados de exemplo (simulação) para testar sem câmara/projetor
# --------------------------------------------------------------------------
def demo_object(w=320, h=240):
    Y, X = np.mgrid[:h, :w]
    Z = 18 * np.exp(-(((X - w * 0.4) / (w * 0.15)) ** 2 + ((Y - h * 0.5) / (h * 0.2)) ** 2))
    Z += 8 * ((np.abs(X - w * 0.75) < w * 0.08) & (np.abs(Y - h * 0.5) < h * 0.25))
    return Z


def simulate_capture(Z, pitch=16, steps=4, k_rad_per_mm=0.35, noise=2.0, rng=None):
    rng = rng or np.random.default_rng(0)
    h, w = Z.shape
    X = np.arange(w)[None, :].repeat(h, 0)
    imgs = []
    for k in range(steps):
        d = 2 * np.pi * k / steps
        I = 128 + 100 * np.cos(2 * np.pi * X / pitch + k_rad_per_mm * Z + d)
        imgs.append(np.clip(I + rng.normal(0, noise, I.shape), 0, 255).astype(np.uint8))
    return imgs


def simulate_focus_stack(Z, z_positions, texture=None, rng=None):
    rng = rng or np.random.default_rng(1)
    tex = texture if texture is not None else ndimage.gaussian_filter(rng.random(Z.shape) * 255, 0.7)
    out = []
    for z in z_positions:
        blur = np.abs(Z - z) * 0.6
        levels = np.unique(np.round(blur, 1))
        img = np.zeros_like(tex)
        for b in levels:
            m = np.round(blur, 1) == b
            img[m] = ndimage.gaussian_filter(tex, max(b, 0.01))[m]
        out.append(img.astype(np.uint8))
    return out
