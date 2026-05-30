import sys
import tempfile
import os
from swmm_api import read_inp_file
from pyswmm import Simulation, Nodes, Links

BASE_INP = "data/networks/chico_hydro-qx1/SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp"

factor = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
print(f"=== Simulación Qx{factor} ===\n")

# Modifica Mfactor en [INFLOWS] usando swmm_api
inp = read_inp_file(BASE_INP)
for inflow in inp["INFLOWS"].values():
    inflow.scale_factor = factor

# Escribe archivo temporal y corre con pyswmm
with tempfile.NamedTemporaryFile(mode="w", suffix=".inp", delete=False) as tmp:
    tmp_path = tmp.name
inp.write_file(tmp_path)

try:
    with Simulation(tmp_path) as sim:
        for step in sim:
            pass

        print(f"{'Nodo':<15} {'Profundidad':>12} {'Flujo':>12} {'Vol. Inund. (m3)':>18}")
        print("-" * 60)
        for node in Nodes(sim):
            flood_vol = node.statistics["flooding_volume"]
            print(f"{node.nodeid:<15} {node.depth:>12.4f} {node.total_inflow:>12.4f} {flood_vol:>18.4f}")

        print(f"\n{'Enlace':<15} {'Flujo':>12} {'Velocidad':>12}")
        print("-" * 40)
        for link in Links(sim):
            area = link.ds_xsection_area or 0
            velocidad = link.flow / area if area > 0 else 0.0
            print(f"{link.linkid:<15} {link.flow:>12.4f} {velocidad:>12.4f}")
finally:
    os.unlink(tmp_path)
