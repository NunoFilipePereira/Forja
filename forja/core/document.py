"""Documento: lista de objetos 3D (malhas) com nome, cor e visibilidade."""
import itertools
import numpy as np
import trimesh

PALETTE = [
    (0.85, 0.55, 0.25), (0.35, 0.65, 0.85), (0.55, 0.80, 0.40), (0.85, 0.40, 0.45),
    (0.70, 0.55, 0.85), (0.90, 0.80, 0.35), (0.40, 0.80, 0.75), (0.75, 0.75, 0.75),
]


def mbounds(mesh):
    """Caixa envolvente que também funciona para nuvens de pontos (malhas sem faces)."""
    v = np.asarray(mesh.vertices)
    if not len(v):
        return np.zeros((2, 3))
    return np.array([v.min(0), v.max(0)])


class MeshObject:
    _ids = itertools.count(1)

    def __init__(self, mesh: trimesh.Trimesh, name: str, color=None):
        self.id = next(MeshObject._ids)
        self.mesh = mesh
        self.name = name
        self.color = color or PALETTE[(self.id - 1) % len(PALETTE)]
        self.visible = True
        self.alpha = 1.0
        self.vcolors = None  # (id da malha, cores RGBA por vértice)
        self.gen = None      # gerador paramétrico: {'id', 'values', 'src', 'group', 'part'}


class Document:
    def __init__(self):
        self.objects: list[MeshObject] = []
        self.listeners = []
        self.undo_stack = []

    # --- notificação -------------------------------------------------------
    def changed(self):
        for cb in self.listeners:
            cb()

    def snapshot(self):
        self.undo_stack.append([(o, o.mesh.copy()) for o in self.objects])
        self.undo_stack = self.undo_stack[-20:]

    def undo(self):
        if not self.undo_stack:
            return False
        state = self.undo_stack.pop()
        self.objects = []
        for o, m in state:
            o.mesh = m
            self.objects.append(o)
        self.changed()
        return True

    # --- edição ------------------------------------------------------------
    def add(self, mesh, name, color=None):
        obj = MeshObject(mesh, name, color)
        self.objects.append(obj)
        self.changed()
        return obj

    def remove(self, objs):
        ids = {o.id for o in objs}
        self.objects = [o for o in self.objects if o.id not in ids]
        self.changed()

    def clear(self):
        self.objects = []
        self.undo_stack = []
        self.changed()

    def bounds(self):
        vis = [o.mesh for o in self.objects if o.visible and len(o.mesh.vertices)]
        if not vis:
            return np.array([[-50, -50, 0], [50, 50, 50]], float)
        b = np.array([mbounds(m) for m in vis])
        return np.array([b[:, 0].min(axis=0), b[:, 1].max(axis=0)])
