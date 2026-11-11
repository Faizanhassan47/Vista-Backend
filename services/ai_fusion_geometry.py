"""Pure-Python level-0 geometry helpers for AI Fusion spatial correlation."""

from __future__ import annotations

from typing import Any, Iterable, Optional


BBox = tuple[float, float, float, float]  # min_x, min_y, max_x, max_y


def geometry_bbox(geometry: Optional[dict]) -> Optional[BBox]:
    if not isinstance(geometry, dict):
        return None
    gtype = geometry.get("type")
    coords = geometry.get("coordinates")
    if gtype == "Point" and isinstance(coords, (list, tuple)) and len(coords) >= 2:
        x, y = float(coords[0]), float(coords[1])
        return (x, y, x, y)
    ring = None
    if gtype == "Polygon" and isinstance(coords, list) and coords:
        ring = coords[0]
    elif gtype == "MultiPolygon" and isinstance(coords, list) and coords and coords[0]:
        ring = coords[0][0]
    if not isinstance(ring, list) or not ring:
        return None
    xs, ys = [], []
    for pt in ring:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            xs.append(float(pt[0]))
            ys.append(float(pt[1]))
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def expand_bbox(bbox: BBox, pad: float) -> BBox:
    return (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)


def bbox_intersects(a: BBox, b: BBox) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def bbox_union(boxes: Iterable[BBox]) -> Optional[BBox]:
    items = list(boxes)
    if not items:
        return None
    return (
        min(b[0] for b in items),
        min(b[1] for b in items),
        max(b[2] for b in items),
        max(b[3] for b in items),
    )


def bbox_as_polygon(bbox: BBox) -> dict:
    min_x, min_y, max_x, max_y = bbox
    return {
        "type": "Polygon",
        "coordinates": [[
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y],
            [min_x, min_y],
        ]],
    }


def _point_in_ring(x: float, y: float, ring: list) -> bool:
    """Ray casting; ring is closed or open list of [x,y]."""
    inside = False
    n = len(ring)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = float(ring[i][0]), float(ring[i][1])
        xj, yj = float(ring[j][0]), float(ring[j][1])
        intersect = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        )
        if intersect:
            inside = not inside
        j = i
    return inside


def point_in_geometry(x: float, y: float, geometry: dict) -> bool:
    gtype = geometry.get("type")
    coords = geometry.get("coordinates")
    if gtype == "Point":
        if not isinstance(coords, (list, tuple)) or len(coords) < 2:
            return False
        return abs(float(coords[0]) - x) < 1e-6 and abs(float(coords[1]) - y) < 1e-6
    if gtype == "Polygon" and isinstance(coords, list) and coords:
        return _point_in_ring(x, y, coords[0])
    if gtype == "MultiPolygon" and isinstance(coords, list):
        for poly in coords:
            if poly and _point_in_ring(x, y, poly[0]):
                return True
    return False


def _rings(geometry: dict) -> list[list]:
    gtype = geometry.get("type")
    coords = geometry.get("coordinates")
    if gtype == "Polygon" and isinstance(coords, list) and coords:
        return [coords[0]]
    if gtype == "MultiPolygon" and isinstance(coords, list):
        return [poly[0] for poly in coords if poly]
    return []


def geometries_overlap(a: dict, b: dict, *, proximity_px: float = 0.0) -> bool:
    """True if geometries intersect, one contains a vertex of the other, or bboxes are within proximity."""
    ba = geometry_bbox(a)
    bb = geometry_bbox(b)
    if ba is None or bb is None:
        return False
    pad = max(0.0, float(proximity_px))
    if not bbox_intersects(expand_bbox(ba, pad), expand_bbox(bb, pad)):
        return False

    # Point vs anything
    if a.get("type") == "Point":
        x, y = float(a["coordinates"][0]), float(a["coordinates"][1])
        if point_in_geometry(x, y, b):
            return True
        if pad > 0 and bbox_intersects(expand_bbox(ba, pad), bb):
            # proximity to polygon bbox / point
            return True
        return False
    if b.get("type") == "Point":
        return geometries_overlap(b, a, proximity_px=proximity_px)

    # Polygon / MultiPolygon: vertex-in-other or bbox intersection after pad=0
    for ring in _rings(a):
        for pt in ring:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                if point_in_geometry(float(pt[0]), float(pt[1]), b):
                    return True
    for ring in _rings(b):
        for pt in ring:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                if point_in_geometry(float(pt[0]), float(pt[1]), a):
                    return True
    # Conservative: overlapping bboxes for large patches after vertex checks
    return bbox_intersects(ba, bb)


def grid_bins(bbox: BBox, cell_size: float) -> list[tuple[int, int]]:
    if cell_size <= 0:
        cell_size = 256.0
    min_x, min_y, max_x, max_y = bbox
    x0 = int(min_x // cell_size)
    y0 = int(min_y // cell_size)
    x1 = int(max_x // cell_size)
    y1 = int(max_y // cell_size)
    return [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]


class UnionFind:
    def __init__(self, ids: Iterable[Any]):
        self.parent = {i: i for i in ids}
        self.rank = {i: 0 for i in ids}

    def find(self, item: Any) -> Any:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a: Any, b: Any) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

    def groups(self) -> dict[Any, list[Any]]:
        out: dict[Any, list[Any]] = {}
        for item in self.parent:
            root = self.find(item)
            out.setdefault(root, []).append(item)
        return out
