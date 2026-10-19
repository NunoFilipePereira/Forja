"""Operações sobre malhas 3D (implementação própria sobre trimesh/manifold3d)."""
import numpy as np
import trimesh
import manifold3d as mf
from scipy import ndimage
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import polylabel, unary_union


# --------------------------------------------------------------------------
# Conversões e booleanas
# --------------------------------------------------------------------------
def to_manifold(mesh: trimesh.Trimesh) -> mf.Manifold:
    if not len(getattr(mesh, "faces", [])):
        raise ValueError("O objeto é uma nuvem de pontos (não tem faces): não dá para fazer operações de sólidos.")
    m = mf.Mesh(vert_properties=np.asarray(mesh.vertices, np.float32),
                tri_verts=np.asarray(mesh.faces, np.uint32))
    out = mf.Manifold(m)
    if out.status() != mf.Error.NoError:
        # tenta reparar antes de desistir
        rep = repair(mesh.copy())[0]
        out = mf.Manifold(mf.Mesh(vert_properties=np.asarray(rep.vertices, np.float32),
                                  tri_verts=np.asarray(rep.faces, np.uint32)))
        if out.status() != mf.Error.NoError:
            # última tentativa: une as partes que estão fechadas (ex.: conjuntos de tubos ou peças sobrepostas)
            parts = []
            for c in mesh.split(only_watertight=False):
                mc = mf.Manifold(mf.Mesh(vert_properties=np.asarray(c.vertices, np.float32), tri_verts=np.asarray(c.faces, np.uint32)))
                if mc.status() == mf.Error.NoError and not mc.is_empty():
                    parts.append(mc)
            if parts and sum(len(p.to_mesh().tri_verts) for p in parts) >= 0.9 * len(mesh.faces):
                return mf.Manifold.batch_boolean(parts, mf.OpType.Add) if len(parts) > 1 else parts[0]
            raise ValueError("A malha não é fechada (watertight). Use Malha ▸ Reparar primeiro.")
    return out


def from_manifold(m: mf.Manifold) -> trimesh.Trimesh:
    # sem fundir vértices por posição: o manifold já partilha os vértices e a fusão pode juntar superfícies quase
    # encostadas, deixando a malha aberta
    mm = m.to_mesh()
    return trimesh.Trimesh(np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts), process=False)


def boolean(a, b, op: str):
    A, B = to_manifold(a), to_manifold(b)
    r = {"union": A + B, "difference": A - B, "intersection": A ^ B}[op]
    return from_manifold(r)


def union_all(meshes):
    ms = [to_manifold(m) for m in meshes]
    acc = ms[0]
    for m in ms[1:]:
        acc = acc + m
    return from_manifold(acc)


# --------------------------------------------------------------------------
# Primitivas
# --------------------------------------------------------------------------
def primitive(kind, **p):
    if kind == "box":
        m = trimesh.creation.box(extents=(p["x"], p["y"], p["z"]))
    elif kind == "sphere":
        m = trimesh.creation.icosphere(subdivisions=4, radius=p["r"])
    elif kind == "cylinder":
        m = trimesh.creation.cylinder(radius=p["r"], height=p["h"], sections=p.get("seg", 64))
    elif kind == "cone":
        m = trimesh.creation.cone(radius=p["r"], height=p["h"], sections=p.get("seg", 64))
    elif kind == "torus":
        m = trimesh.creation.torus(major_radius=p["R"], minor_radius=p["r"])
    elif kind == "capsule":
        m = trimesh.creation.capsule(height=p["h"], radius=p["r"])
    else:
        raise ValueError(kind)
    return drop_to_floor(m)


# --------------------------------------------------------------------------
# Transformações
# --------------------------------------------------------------------------
def translate(mesh, dx, dy, dz):
    mesh.apply_translation((dx, dy, dz))
    return mesh


def rotate(mesh, ax, ay, az, about_center=True):
    c = mesh.bounds.mean(axis=0) if about_center else np.zeros(3)
    R = trimesh.transformations.euler_matrix(np.radians(ax), np.radians(ay), np.radians(az), "sxyz")
    T = trimesh.transformations.translation_matrix(c) @ R @ trimesh.transformations.translation_matrix(-c)
    mesh.apply_transform(T)
    return mesh


def scale(mesh, sx, sy, sz):
    c = mesh.bounds.mean(axis=0)
    T = trimesh.transformations.translation_matrix(c) @ np.diag([sx, sy, sz, 1.0]) \
        @ trimesh.transformations.translation_matrix(-c)
    mesh.apply_transform(T)
    return mesh


def scale_to_size(mesh, size, axis=None):
    ext = mesh.extents
    k = size / (ext.max() if axis is None else ext[axis])
    return scale(mesh, k, k, k)


def mirror(mesh, axis):
    M = np.eye(4)
    M[axis, axis] = -1
    c = mesh.bounds.mean(axis=0)
    mesh.apply_translation(-c)
    mesh.apply_transform(M)
    mesh.apply_translation(c)
    mesh.invert()  # o espelho troca a orientação das faces
    return mesh


def drop_to_floor(mesh):
    mesh.apply_translation((0, 0, -mesh.bounds[0][2]))
    return mesh


def center_on_plate(mesh):
    b = mesh.bounds
    mesh.apply_translation((-(b[0][0] + b[1][0]) / 2, -(b[0][1] + b[1][1]) / 2, -b[0][2]))
    return mesh


def lay_flat(mesh):
    """Roda para assentar na maior face plana (orientação estável)."""
    transforms, probs = trimesh.poses.compute_stable_poses(mesh, n_samples=1)
    mesh.apply_transform(transforms[0])
    return center_on_plate(mesh)


# --------------------------------------------------------------------------
# Corte e partição
# --------------------------------------------------------------------------
def cut_plane(mesh, origin, normal):
    """Corta com um plano e fecha as secções. Devolve (lado +, lado -)."""
    normal = np.asarray(normal, float)
    normal /= np.linalg.norm(normal)
    try:
        M = to_manifold(mesh)
        pos, neg = M.split_by_plane(normal.tolist(), float(np.dot(normal, origin)))
        return from_manifold(pos), from_manifold(neg)
    except ValueError:
        pos = trimesh.intersections.slice_mesh_plane(mesh, normal, origin, cap=True)
        neg = trimesh.intersections.slice_mesh_plane(mesh, -normal, origin, cap=True)
        return pos, neg


def _box_mesh(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    b = trimesh.creation.box(extents=hi - lo)
    b.apply_translation((lo + hi) / 2)
    return b


def _pin(center, axis, radius, length, seg=32):
    c = trimesh.creation.cylinder(radius=radius, height=length, sections=seg)
    z = np.array([0, 0, 1.0])
    a = np.zeros(3); a[axis] = 1
    if axis != 2:
        R = trimesh.geometry.align_vectors(z, a)
        c.apply_transform(R)
    c.apply_translation(center)
    return c


def split_grid(mesh, nx=2, ny=1, nz=1, connectors=True, pin_radius=3.0, pin_length=10.0,
               clearance=0.2, pins_per_face=1):
    """Divide o modelo numa grelha nx×ny×nz de peças imprimíveis.
    Opcionalmente adiciona pinos de encaixe (macho numa peça, furo com folga na vizinha)."""
    lo, hi = mesh.bounds
    pad = 1.0
    xs = np.linspace(lo[0], hi[0], nx + 1); ys = np.linspace(lo[1], hi[1], ny + 1)
    zs = np.linspace(lo[2], hi[2], nz + 1)
    edges = [xs, ys, zs]
    M = to_manifold(mesh)
    cells = {}
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                a = [xs[i], ys[j], zs[k]]; b = [xs[i + 1], ys[j + 1], zs[k + 1]]
                a = [v - (pad if idx == 0 else 0) for v, idx in zip(a, (i, j, k))]
                b = [v + (pad if idx == n - 1 else 0) for v, idx, n in zip(b, (i, j, k), (nx, ny, nz))]
                part = M ^ to_manifold(_box_mesh(a, b))
                cells[(i, j, k)] = part
    if connectors:
        dims = (nx, ny, nz)
        for (i, j, k), part in list(cells.items()):
            for axis in range(3):
                idx = [i, j, k]
                if idx[axis] + 1 >= dims[axis]:
                    continue
                nb = idx.copy(); nb[axis] += 1; nb = tuple(nb)
                plane = edges[axis][idx[axis] + 1]
                normal = np.zeros(3); normal[axis] = 1
                sec = mesh.section(plane_origin=normal * plane, plane_normal=normal)
                if sec is None:
                    continue
                planar, to3d = sec.to_2D()
                polys = [p for p in planar.polygons_full if p.area > (pin_radius * 3) ** 2]
                # restringe à face partilhada entre as duas células
                other = [a for a in range(3) if a != axis]
                cell_lo = np.array([edges[a][idx[a]] for a in range(3)])
                cell_hi = np.array([edges[a][idx[a] + 1] for a in range(3)])
                placed = 0
                for poly in sorted(polys, key=lambda p: -p.area):
                    if placed >= pins_per_face:
                        break
                    pt2 = polylabel(poly, tolerance=0.5)
                    if poly.exterior.distance(pt2) < pin_radius * 1.8:
                        continue
                    p3 = (to3d @ np.array([pt2.x, pt2.y, 0, 1.0]))[:3]
                    p3[axis] = plane
                    if not all(cell_lo[a] <= p3[a] <= cell_hi[a] for a in other):
                        continue
                    # pino macho: metade dentro desta peça, metade a sair para a vizinha
                    male = _pin(p3, axis, pin_radius, pin_length)
                    hole = _pin(p3, axis, pin_radius + clearance, pin_length + 2 * clearance)
                    cells[(i, j, k)] = cells[(i, j, k)] + to_manifold(male)
                    cells[nb] = cells[nb] - to_manifold(hole)
                    placed += 1
    parts = []
    for key, part in cells.items():
        m = from_manifold(part)
        if len(m.faces):
            parts.append((key, m))
    return parts


def split_by_planes(mesh, axis, positions):
    pieces = [mesh]
    normal = np.zeros(3); normal[axis] = 1
    for p in sorted(positions):
        nxt = []
        for pc in pieces:
            if pc.bounds[0][axis] < p < pc.bounds[1][axis]:
                pos, neg = cut_plane(pc, normal * p, normal)
                nxt += [m for m in (neg, pos) if len(m.faces)]
            else:
                nxt.append(pc)
        pieces = nxt
    return pieces


# --------------------------------------------------------------------------
# Oco (hollow)
# --------------------------------------------------------------------------
def hollow(mesh, thickness=2.0, resolution=120, drain_radius=0.0):
    """Esvazia o interior deixando paredes com a espessura indicada (via campo de distância)."""
    from skimage import measure
    pitch = mesh.extents.max() / resolution
    pad = 3
    lo = mesh.bounds[0] - pad * pitch
    shape = np.ceil((mesh.extents + 2 * pad * pitch) / pitch).astype(int) + 1
    vox = mesh.voxelized(pitch).fill()
    solid = np.zeros(shape, bool)
    pts = vox.points
    idx = np.round((pts - lo) / pitch).astype(int)
    ok = np.all((idx >= 0) & (idx < shape), axis=1)
    solid[tuple(idx[ok].T)] = True
    dist = ndimage.distance_transform_edt(solid) * pitch
    level = thickness
    if dist.max() <= level + pitch:
        raise ValueError("Espessura demasiado grande para este modelo.")
    verts, faces, _, _ = measure.marching_cubes(dist, level=level)
    inner = trimesh.Trimesh(verts * pitch + lo, faces[:, ::-1], process=True)
    inner = trimesh.smoothing.filter_taubin(inner, iterations=20)
    if inner.volume < 0:
        inner.invert()
    out = to_manifold(mesh) - to_manifold(repair(inner)[0])
    result = from_manifold(out)
    if drain_radius > 0:
        c = mesh.bounds.mean(axis=0)
        drain = _pin([c[0], c[1], mesh.bounds[0][2] + thickness * 1.5], 2, drain_radius, thickness * 4)
        result = boolean(result, drain, "difference")
    return result


def shell_offset(mesh, distance):
    """Inflar/encolher: desloca os vértices ao longo das normais."""
    m = mesh.copy()
    m.vertices = m.vertices + m.vertex_normals * distance
    return m


# --------------------------------------------------------------------------
# Reparação e simplificação
# --------------------------------------------------------------------------
def repair(mesh):
    rep = []
    n0 = len(mesh.faces)
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    rep.append(f"Faces degeneradas/duplicadas removidas: {n0 - len(mesh.faces)}")
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_inversion(mesh)
    trimesh.repair.fix_normals(mesh)
    holes_before = not mesh.is_watertight
    trimesh.repair.fill_holes(mesh)
    if holes_before:
        rep.append("Buracos preenchidos" + ("" if mesh.is_watertight else " (parcialmente)"))
    rep.append("Malha fechada (watertight): " + ("sim" if mesh.is_watertight else "não"))
    return mesh, rep


def simplify(mesh, target_faces):
    target_faces = int(max(4, target_faces))
    if target_faces >= len(mesh.faces):
        return mesh
    if len(mesh.faces) < 50:
        raise ValueError(f"A malha já é muito simples ({len(mesh.faces)} faces); não há nada a simplificar.")
    return mesh.simplify_quadric_decimation(face_count=target_faces)


def densify(mesh, max_edge):
    v, f = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=max_edge, max_iter=20)
    return trimesh.Trimesh(v, f)


def smooth(mesh, iterations=10):
    m = mesh.copy()
    trimesh.smoothing.filter_taubin(m, iterations=iterations)
    return m


def separate(mesh):
    return list(mesh.split(only_watertight=False))


# --------------------------------------------------------------------------
# Superfícies a partir de mapas de altura
# --------------------------------------------------------------------------
def heightmap_to_solid(H, size_x, size_y, base=1.0, mask=None):
    """Transforma uma matriz de alturas (mm) num sólido fechado: topo + paredes + fundo."""
    H = np.asarray(H, float)
    rows, cols = H.shape
    xs = np.linspace(0, size_x, cols); ys = np.linspace(size_y, 0, rows)
    X, Y = np.meshgrid(xs, ys)
    top = np.column_stack([X.ravel(), Y.ravel(), (H + base).ravel()])
    bot = np.column_stack([X.ravel(), Y.ravel(), np.zeros(H.size)])
    n = H.size
    idx = np.arange(n).reshape(rows, cols)
    a = idx[:-1, :-1].ravel(); b = idx[:-1, 1:].ravel(); c = idx[1:, :-1].ravel(); d = idx[1:, 1:].ravel()
    ft = np.vstack([np.column_stack([a, c, b]), np.column_stack([b, c, d])])
    fb = ft[:, ::-1] + n
    border = np.concatenate([idx[0, :], idx[1:, -1], idx[-1, -2::-1], idx[-2:0:-1, 0]])
    nb = np.roll(border, -1)
    fs = np.vstack([np.column_stack([border, nb, nb + n]), np.column_stack([border, nb + n, border + n])])
    m = trimesh.Trimesh(np.vstack([top, bot]), np.vstack([ft, fb, fs]), process=True)
    if m.volume < 0:
        m.invert()
    return m


def heightmap_to_surface(H, size_x, size_y, mask=None):
    """Superfície aberta (para nuvens/mapas de profundidade). Ignora pixels fora da máscara/NaN."""
    H = np.asarray(H, float)
    rows, cols = H.shape
    valid = np.isfinite(H) if mask is None else (mask & np.isfinite(H))
    xs = np.linspace(0, size_x, cols); ys = np.linspace(size_y, 0, rows)
    X, Y = np.meshgrid(xs, ys)
    V = np.column_stack([X.ravel(), Y.ravel(), np.nan_to_num(H).ravel()])
    idx = np.arange(H.size).reshape(rows, cols)
    a = idx[:-1, :-1]; b = idx[:-1, 1:]; c = idx[1:, :-1]; d = idx[1:, 1:]
    va = valid[:-1, :-1]; vb = valid[:-1, 1:]; vc = valid[1:, :-1]; vd = valid[1:, 1:]
    t1 = va & vb & vc; t2 = vb & vc & vd
    F = np.vstack([np.column_stack([a[t1], c[t1], b[t1]]), np.column_stack([b[t2], c[t2], d[t2]])])
    m = trimesh.Trimesh(V, F, process=False)
    m.remove_unreferenced_vertices()
    return m


def image_to_heightmap(gray, height, invert=False, blur=0.0, max_px=400):
    g = np.asarray(gray, float)
    if g.max() > 1:
        g = g / 255.0
    s = max(g.shape) / max_px
    if s > 1:
        g = ndimage.zoom(g, 1 / s, order=1)
    if blur > 0:
        g = ndimage.gaussian_filter(g, blur)
    if not invert:
        g = 1 - g  # escuro = alto (litofania)
    return g * height


def lithophane(gray, width, thickness_max=3.0, thickness_min=0.6):
    H = image_to_heightmap(gray, thickness_max - thickness_min)
    rows, cols = H.shape
    return heightmap_to_solid(H, width, width * rows / cols, base=thickness_min)


# --------------------------------------------------------------------------
# Fatiamento (camadas 2D, p.ex. para laser/stack de chapas)
# --------------------------------------------------------------------------
def slice_layers(mesh, layer_height):
    z0, z1 = mesh.bounds[0][2], mesh.bounds[1][2]
    heights = np.arange(z0 + layer_height / 2, z1, layer_height)
    layers = []
    for z in heights:
        sec = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        if sec is None:
            continue
        planar, T = sec.to_2D(to_2D=np.eye(4))
        polys = unary_union(list(planar.polygons_full))
        layers.append((z, polys))
    return layers


def section_polygons(mesh, origin, normal):
    sec = mesh.section(plane_origin=origin, plane_normal=normal)
    if sec is None:
        return None
    planar, _ = sec.to_2D()
    return unary_union(list(planar.polygons_full))


# --------------------------------------------------------------------------
# Medições
# --------------------------------------------------------------------------
def properties(mesh):
    v = np.asarray(mesh.vertices)
    ext = (v.max(0) - v.min(0)) if len(v) else np.zeros(3)
    if not len(mesh.faces):
        return {"Vértices / pontos": len(v), "Faces": 0, "Dimensões X×Y×Z (mm)": f"{ext[0]:.2f} × {ext[1]:.2f} × {ext[2]:.2f}"}
    info = {
        "Vértices": len(mesh.vertices),
        "Faces": len(mesh.faces),
        "Dimensões X×Y×Z (mm)": f"{ext[0]:.2f} × {ext[1]:.2f} × {ext[2]:.2f}",
        "Área de superfície (mm²)": f"{mesh.area:.2f}",
        "Fechada (watertight)": "sim" if mesh.is_watertight else "não",
        "Componentes": len(mesh.split(only_watertight=False)) if len(mesh.faces) < 400000 else "?",
    }
    if mesh.is_watertight:
        info["Volume (mm³)"] = f"{abs(mesh.volume):.2f}"
        info["Volume (cm³)"] = f"{abs(mesh.volume) / 1000:.3f}"
        info["Peso PLA 1.24 g/cm³ (g)"] = f"{abs(mesh.volume) / 1000 * 1.24:.1f}"
        info["Centro de massa"] = ", ".join(f"{v:.2f}" for v in mesh.center_mass)
    return info


def overhang_faces(mesh, angle_deg=45):
    """Faces que precisam de suporte (normal a apontar para baixo além do ângulo)."""
    nz = mesh.face_normals[:, 2]
    return nz < -np.cos(np.radians(angle_deg))
