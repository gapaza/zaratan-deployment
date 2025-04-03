from thermoelastic2d.utils import get_res_bounds


nelx = 64
nely = 64
lci, tri, rci, bri = get_res_bounds(nelx + 1, nely + 1)
base_conditions = frozenset(
    {
        ("nelx", 64),
        ("nely", 64),
        ("fixed_elements", (tri[21], tri[32], tri[43])),
        ("force_elements_x", (bri[31])),
        ("force_elements_y", (bri[31])),
        ("heatsink_elements", (rci[31], rci[32], rci[33])),
        ("volfrac", 0.3),
        ("rmin", 1.1),
        ("weight", 0.5),  # 1.0 for pure structural, 0.0 for pure thermal
    }
)