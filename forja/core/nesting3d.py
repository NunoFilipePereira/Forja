"""Nesting 3D: arruma peças dentro de caixas (bins) X × Y × Z com folgas peça-caixa e
peça-peça, pela ordem escolhida (Default, Area descending, Volume descending). Usa as caixas envolventes das peças,
com rotação de 90° em Z quando ajuda, e o método dos "pontos extremos" (mais baixo, depois mais atrás, mais à esquerda)."""
import numpy as np

SEQUENCES = ["Default", "Area descending", "Volume descending"]


def _fits(p, s, placed, B, gap_bin, gap):
    if np.any(p < gap_bin - 1e-9) or np.any(p + s > B - gap_bin + 1e-9):
        return False
    for q, t in placed:
        if np.all(p < q + t + gap - 1e-9) and np.all(q < p + s + gap - 1e-9):
            return False
    return True


def pack(sizes, bin_size, gap_bin=0.0, gap=0.0, sequence="Volume descending", rotate=True):
    """sizes: lista de (dx, dy, dz). Devolve [(índice_da_caixa, posição_mínima xyz, rodado_90)] pela ordem original."""
    B = np.asarray(bin_size, float)
    n = len(sizes)
    S = np.asarray(sizes, float)
    order = list(range(n))
    if sequence == "Area descending":
        order.sort(key=lambda i: -S[i][0] * S[i][1])
    elif sequence == "Volume descending":
        order.sort(key=lambda i: -np.prod(S[i]))
    bins = []  # cada caixa: {"placed": [(p, s)], "points": [p]}
    out = [None] * n
    for i in order:
        s0 = S[i]
        options = [(s0, False)] + ([(s0[[1, 0, 2]], True)] if rotate and abs(s0[0] - s0[1]) > 1e-6 else [])
        done = False
        for b, bn in enumerate(bins + [None]):
            if bn is None:
                bn = {"placed": [], "points": [np.full(3, gap_bin)]}
                new = True
            else:
                new = False
            best = None
            for p in sorted(bn["points"], key=lambda q: (q[2], q[1], q[0])):
                for s, r in options:
                    if _fits(p, s, bn["placed"], B, gap_bin, gap):
                        key = (p[2], p[1], p[0])
                        if best is None or key < best[0]:
                            best = (key, p.copy(), s, r)
                if best is not None:
                    break
            if best is None:
                if new:
                    out[i] = (-1, None, False)  # não cabe numa caixa vazia
                    done = True
                    break
                continue
            _, p, s, r = best
            bn["placed"].append((p, s))
            bn["points"] = [q for q in bn["points"] if not np.allclose(q, p)]
            for d in range(3):
                e = p.copy(); e[d] += s[d] + gap
                bn["points"].append(e)
            if new:
                bins.append(bn)
                b = len(bins) - 1
            out[i] = (b, p, r)
            done = True
            break
        if not done:
            out[i] = (-1, None, False)
    return out, len(bins)
