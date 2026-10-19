"""Tomografia (CT) e volumes 3D: Radon, retroprojeção (filtrada), reconstrução por fatias,
cortes, projeções, isosuperfícies e voxelização."""
import os
import numpy as np
from scipy import ndimage

FILTERS = ["ramp", "shepp-logan", "cosine", "hamming", "hann", "sem filtro"]


def radon(img, n_angles=180, max_angle=180.0):
    from skimage.transform import radon as _r
    a = np.asarray(img, float)
    if a.ndim == 3:
        a = a.mean(2)
    theta = np.linspace(0, max_angle, n_angles, endpoint=False)
    return _r(a, theta=theta, circle=False), theta


def backproject(sino, max_angle=180.0, filt="ramp", size=None):
    from skimage.transform import iradon
    s = np.asarray(sino, float)
    theta = np.linspace(0, max_angle, s.shape[1], endpoint=False)
    return iradon(s, theta=theta, filter_name=None if filt == "sem filtro" else filt, output_size=size, circle=False)


def sart(sino, max_angle=180.0, iterations=2, size=None):
    """Reconstrução algébrica iterativa (SART); útil com poucos ângulos."""
    from skimage.transform import iradon_sart
    s = np.asarray(sino, float)
    theta = np.linspace(0, max_angle, s.shape[1], endpoint=False)
    rec = None
    for _ in range(iterations):
        rec = iradon_sart(s, theta=theta, image=rec)
    if size:
        c = (rec.shape[0] - size) // 2
        rec = rec[c:c + size, c:c + size]
    return rec


def projections_to_volume(projections, max_angle=180.0, filt="ramp", progress=None):
    """Projeções de feixe paralelo (uma imagem por ângulo; linhas = altura Z) → volume (z, y, x)."""
    P = np.stack([np.asarray(p, float) if np.ndim(p) == 2 else np.asarray(p, float).mean(2) for p in projections])
    n_ang, nz, nx = P.shape
    vol = []
    for z in range(nz):
        sino = P[:, z, :].T  # (detetor, ângulo)
        vol.append(backproject(sino, max_angle, filt, size=nx))
        if progress:
            progress(z, nz)
    # convenção: rotação anti-horária vista de cima, y da imagem para baixo
    return np.stack(vol)[:, ::-1, :]


def forward_project_volume(vol, n_angles=90, max_angle=180.0):
    """Projeções paralelas sintéticas de um volume (z, y, x) a rodar em torno de Z."""
    V = np.asarray(vol, float)
    out = []
    for a in np.linspace(0, max_angle, n_angles, endpoint=False):
        R = ndimage.rotate(V, a, axes=(1, 2), reshape=False, order=1)
        out.append(R.sum(axis=1))  # soma ao longo de y → (z, x)
    return out


def phantom2d(size=256):
    from skimage.data import shepp_logan_phantom
    from skimage.transform import resize
    return resize(shepp_logan_phantom(), (size, size))


def phantom3d(n=64):
    z, y, x = np.mgrid[-1:1:n * 1j, -1:1:n * 1j, -1:1:n * 1j]
    V = np.zeros((n, n, n))
    for (cz, cy, cx, az, ay, ax, val) in [(0, 0, 0, .9, .7, .6, 1.0), (0, 0, 0, .8, .62, .52, -0.6),
                                          (.2, .2, .1, .25, .15, .15, 0.5), (-.3, -.2, -.15, .2, .2, .12, 0.4),
                                          (0, -.35, .2, .1, .1, .1, 0.8)]:
        V[((z - cz) / az) ** 2 + ((y - cy) / ay) ** 2 + ((x - cx) / ax) ** 2 <= 1] += val
    return V


def slice_volume(vol, axis, index):
    V = np.asarray(vol)
    index = int(np.clip(index, 0, V.shape[axis] - 1))
    return np.take(V, index, axis=axis)


def project(vol, axis=0, kind="máxima"):
    V = np.asarray(vol, float)
    return {"máxima": V.max(axis), "média": V.mean(axis), "soma": V.sum(axis), "mínima": V.min(axis)}[kind]


def isosurface(vol, level=None, spacing=(1.0, 1.0, 1.0), step=1):
    import trimesh
    from skimage import measure
    V = np.asarray(vol, float)
    lo, hi = np.nanmin(V), np.nanmax(V)
    level = (lo + hi) / 2 if level is None else level
    if not lo < level < hi:
        raise ValueError(f"O nível tem de estar entre {lo:.4g} e {hi:.4g}.")
    Vp = np.pad(V, 1, constant_values=lo)
    verts, faces, normals, _ = measure.marching_cubes(Vp, level, spacing=spacing, step_size=step)
    verts -= np.array(spacing)
    # (z, y, x) → (x, y, z)
    m = trimesh.Trimesh(verts[:, ::-1], faces[:, ::-1], process=True)
    if m.volume < 0:
        m.invert()
    return m


def voxelize(mesh, pitch=1.0):
    vox = mesh.voxelized(pitch).fill()
    M = vox.matrix.astype(np.uint8)  # (x, y, z)
    return np.transpose(M, (2, 1, 0)) * 255  # → (z, y, x)


def volume_stats(vol, threshold=None, spacing=1.0):
    V = np.asarray(vol, float)
    s = {"dimensões (z, y, x)": V.shape, "mín": float(V.min()), "máx": float(V.max()), "média": float(V.mean()),
         "desvio-padrão": float(V.std())}
    if threshold is not None:
        n = int((V > threshold).sum())
        s["voxels acima do limiar"] = n
        s["volume acima do limiar"] = n * spacing ** 3
    return s


def load_image_stack(folder):
    from .imageops import load, to_gray
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".png", ".tif", ".tiff", ".bmp", ".jpg", ".dcm")))
    if not files:
        raise ValueError("A pasta não tem imagens.")
    sl = []
    for f in files:
        p = os.path.join(folder, f)
        if f.lower().endswith(".dcm"):
            from .imageops2 import load_dicom
            a = load_dicom(p)
        else:
            a = load(p)
            a = to_gray(a) if a.ndim == 3 else a
        sl.append(np.asarray(a, float))
    shp = sl[0].shape
    return np.stack([s for s in sl if s.shape == shp])


def save_im3(path, vol):
    """Formato simples .im3: cabeçalho (3 × int32: largura, altura, profundidade) + float32."""
    V = np.asarray(vol, np.float32)
    with open(path, "wb") as f:
        np.array([V.shape[2], V.shape[1], V.shape[0]], np.int32).tofile(f)
        V.tofile(f)


def load_im3(path):
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        w, h, d = (int(x) for x in np.fromfile(f, np.int32, 3))
        if min(w, h, d) <= 0 or 12 + 4 * w * h * d != size:
            raise ValueError("Não é um ficheiro .im3 válido (cabeçalho não coincide com o tamanho).")
        return np.fromfile(f, np.float32, int(w) * int(h) * int(d)).reshape(d, h, w)
