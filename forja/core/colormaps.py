"""Escalas de cor da vista 3D (pontos, arestas, superfície): True color, Gray, Rainbow 1, Rainbow 2,
Hot metal 1, Hot metal 2, RGB, High contrast, Solid color. A cor segue a altura (Z) normalizada."""
import numpy as np

SCHEMES = ["True color", "Gray", "Rainbow 1", "Rainbow 2", "Hot metal 1", "Hot metal 2", "RGB", "High contrast", "Solid color"]
PT = {"True color": "Cor verdadeira", "Gray": "Cinzento", "Rainbow 1": "Arco-íris 1", "Rainbow 2": "Arco-íris 2",
      "Hot metal 1": "Metal quente 1", "Hot metal 2": "Metal quente 2", "RGB": "RGB", "High contrast": "Alto contraste",
      "Solid color": "Cor sólida"}
MARKERS = ["Dot", "x", "+", "Circle", "Vertical diamond", "Horizontal diamond", "Upward triangle", "Downward triangle", "Star"]
MARKERS_PT = {"Dot": "Ponto", "x": "x", "+": "+", "Circle": "Círculo", "Vertical diamond": "Losango vertical",
              "Horizontal diamond": "Losango horizontal", "Upward triangle": "Triângulo para cima",
              "Downward triangle": "Triângulo para baixo", "Star": "Estrela"}
DASHES = ["Solid line", "Dotted line", "Long dash", "Short dash", "Dot dash"]
DASHES_PT = {"Solid line": "Contínua", "Dotted line": "Pontilhada", "Long dash": "Traço longo", "Short dash": "Traço curto",
             "Dot dash": "Traço-ponto"}
# padrões (comprimentos alternados traço/espaço, em frações do comprimento de referência)
DASH_PATTERN = {"Dotted line": [0.15, 0.35], "Long dash": [1.0, 0.4], "Short dash": [0.45, 0.35], "Dot dash": [0.9, 0.3, 0.15, 0.3]}


def _interp(t, stops):
    stops = np.asarray(stops, float)
    xs = np.linspace(0, 1, len(stops))
    return np.stack([np.interp(t, xs, stops[:, i]) for i in range(3)], -1)


def colors(t, scheme, solid=(0.8, 0.8, 0.8), alpha=1.0):
    """t em [0, 1] (N,) → cores RGBA (N, 4)."""
    t = np.clip(np.nan_to_num(np.asarray(t, float)), 0, 1)
    if scheme == "Gray":
        c = np.stack([0.15 + 0.8 * t] * 3, -1)
    elif scheme == "Rainbow 1":   # azul → ciano → verde → amarelo → vermelho
        c = _interp(t, [(0, 0, 0.6), (0, 0.4, 1), (0, 1, 1), (0.2, 1, 0.2), (1, 1, 0), (1, 0.45, 0), (0.75, 0, 0)])
    elif scheme == "Rainbow 2":   # roda de cores completa (HSV)
        h = (1 - t) * 0.8
        i = np.floor(h * 6).astype(int) % 6; f = h * 6 - np.floor(h * 6)
        p, q, r_ = np.zeros_like(t), 1 - f, f
        one = np.ones_like(t)
        c = np.select([i[:, None] == k for k in range(6)],
                      [np.stack(x, -1) for x in ((one, r_, p), (q, one, p), (p, one, r_), (p, q, one), (r_, p, one), (one, p, q))])
    elif scheme == "Hot metal 1":  # preto → vermelho → amarelo → branco
        c = _interp(t, [(0, 0, 0), (0.6, 0, 0), (1, 0.3, 0), (1, 0.8, 0), (1, 1, 1)])
    elif scheme == "Hot metal 2":  # castanho → laranja → branco
        c = _interp(t, [(0.15, 0.05, 0), (0.55, 0.2, 0.02), (0.9, 0.5, 0.1), (1, 0.85, 0.5), (1, 1, 0.95)])
    elif scheme == "RGB":
        c = _interp(t, [(0, 0, 1), (0, 1, 0), (1, 0, 0)])
    elif scheme == "High contrast":
        bands = (np.floor(t * 10) % 2).astype(float)
        c = np.stack([bands, bands * 0.9, 1 - bands * 0.6], -1) * 0.9 + 0.05
    else:  # Solid color / True color sem cores próprias
        c = np.tile(np.asarray(solid, float)[:3], (len(t), 1))
    return np.c_[c, np.full(len(t), alpha)].astype(np.float32)


def marker_segments(points, marker, size):
    """Marcadores (x, +, losangos, triângulos, estrela, círculo) desenhados com segmentos no plano XY + vertical,
    com 'size' em mm. Devolve um array (2M, 3) de pares de pontos para GL "lines"."""
    P = np.asarray(points, float)
    s = size / 2
    if marker == "x":
        offs = [((-s, -s, 0), (s, s, 0)), ((-s, s, 0), (s, -s, 0))]
    elif marker == "+":
        offs = [((-s, 0, 0), (s, 0, 0)), ((0, -s, 0), (0, s, 0)), ((0, 0, -s), (0, 0, s))]
    elif marker in ("Vertical diamond", "Horizontal diamond"):
        a, b = (s * 0.6, s) if marker == "Vertical diamond" else (s, s * 0.6)
        pts = [(a, 0, 0), (0, 0, b), (-a, 0, 0), (0, 0, -b)]
        offs = [(pts[i], pts[(i + 1) % 4]) for i in range(4)]
    elif marker in ("Upward triangle", "Downward triangle"):
        k = 1 if marker == "Upward triangle" else -1
        pts = [(-s, 0, -k * s * 0.6), (s, 0, -k * s * 0.6), (0, 0, k * s)]
        offs = [(pts[i], pts[(i + 1) % 3]) for i in range(3)]
    elif marker == "Star":
        pts = []
        for i in range(10):
            r = s if i % 2 == 0 else s * 0.45
            a = np.pi / 2 + i * np.pi / 5
            pts.append((r * np.cos(a), 0, r * np.sin(a)))
        offs = [(pts[i], pts[(i + 1) % 10]) for i in range(10)]
    elif marker == "Circle":
        pts = [(s * np.cos(a), 0, s * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 13)[:-1]]
        offs = [(pts[i], pts[(i + 1) % 12]) for i in range(12)]
    else:
        return None
    O = np.asarray(offs, float)  # (K, 2, 3)
    segs = P[:, None, None, :] + O[None]  # (N, K, 2, 3)
    return segs.reshape(-1, 3)


def dashed_segments(A, B, dash, ref):
    """Divide as arestas A→B em traços segundo o tipo de linha (ref = comprimento de referência em mm)."""
    pat = DASH_PATTERN.get(dash)
    if not pat:
        return np.stack([A, B], 1).reshape(-1, 3)
    out = []
    d = B - A
    L = np.linalg.norm(d, axis=1)
    period = sum(pat) * ref
    for a, v, l in zip(A, d, L):
        if l < 1e-9:
            continue
        u = v / l
        t = 0.0; k = 0
        while t < l:
            seg = pat[k % len(pat)] * ref
            if k % 2 == 0:
                out.append(a + u * t); out.append(a + u * min(t + seg, l))
            t += seg; k += 1
            if len(out) > 2_000_000:
                break
    return np.asarray(out, float) if out else np.zeros((0, 3))
