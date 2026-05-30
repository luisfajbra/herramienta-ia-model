#se hacen pruebas para poder ver los resultados de 
import pyswmm as py
file = "data/networks/chico_hydro-qx1/SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp"

with py.Simulation(file) as sim:
    for step in sim:
        print(f"Tiempo: {sim.current_time}")
        for node in sim.nodes:
            print(f"Nodo: {node.id}, Profundidad: {node.depth}, Flujo: {node.flow}")
        for link in sim.links:
            print(f"Enlace: {link.id}, Flujo: {link.flow}, Velocidad: {link.velocity}")