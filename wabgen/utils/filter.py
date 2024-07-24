
"""."""

import itertools as it
import numpy as np
import pandas as pd
import networkx as nx
import ase.io
from ase.data import covalent_radii
from ase.neighborlist import NeighborList
from func_timeout import func_set_timeout
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.alchemy.filters import RemoveDuplicatesFilter
from collections import defaultdict
from pymatgen.core import Structure
import queue
import wabgen.core


# Maximum execution time of the function (s).
time_limit = 30


def detect_ligand(atoms, indexs):
    """Return dict composed of indices belonging to the ligands of the system."""
    numbers = np.array(atoms.get_atomic_numbers())
    radii = np.array([covalent_radii[n] for n in numbers])
    nl_ligands = NeighborList(
        cutoffs=radii,
        bothways=True,
        self_interaction=False
    )

    nl_ligands.update(atoms)

    connectivity = {}
    conn = nx.DiGraph()
    for atom in atoms:
        # atoms list connected
        i1, _ = nl_ligands.get_neighbors(atom.index)
        connectivity[atom.index] = list(i1)
        conn.add_node(atom.index)

    # Add edges like bonds
    for i in connectivity:
        for ai, aj in it.product([i], connectivity[i]):
            conn.add_edge(ai, aj)
            conn.add_edge(aj, ai)

    atoms_MOL = nx.weakly_connected_components(conn)
    # dict imol : Natoms, indexs
    bulk = dict()
    ipol = 0
    for mol in atoms_MOL:
        mol = list(sorted(mol))
        bulk[ipol] = dict()
        bulk[ipol]["Natoms"] = len(mol)
        bulk[ipol]["index"] = list(np.array(indexs)[mol])
        ipol += 1

    return bulk


def get_neighbor_info(index, atoms, indices, offsets, cutoff, dist_min):
    """Return distances."""
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    distances = []
    box_offsets = []
    directions = []
    neighbor_position_list = []
    for neighbor_index, offset in zip(indices, offsets):
        neighbor_position = positions[neighbor_index] + np.dot(offset, cell)
        displacement_vector = neighbor_position - positions[index]
        direction = displacement_vector / np.linalg.norm(displacement_vector)
        # print(atoms[index].symbol, atoms[neighbor_index].symbol, direction)
        # print(classify_directions(direction, neighbor_index, threshold_angle=10))
        distance = np.linalg.norm(positions[index] - neighbor_position)
        if dist_min < distance <= cutoff:
            distances.append(distance)
            box_offsets.append(tuple(offset))
            directions.append(direction)
            neighbor_position_list.append(round(np.dot(neighbor_position, neighbor_position), 3))

    return np.array(sorted(distances)), box_offsets, directions, neighbor_position_list


def angle_matrix(vectors):
    """
    Construye una matriz de ángulos entre una lista de vectores de 3 componentes.

    Args:
        vectors (list of np.array): Lista de vectores de 3 componentes.

    Returns:
        np.array: Matriz de ángulos en radianes entre los vectores.
    """
    num_vectors = len(vectors)
    angles = np.zeros((num_vectors, num_vectors))

    for i in range(num_vectors):
        for j in range(i, num_vectors):
            angle = angle_between_vectors(vectors[i], vectors[j])
            angles[i, j] = np.rad2deg(angle)
            angles[j, i] = np.rad2deg(angle)  # La matriz es simétrica

    return angles


def angle_between_vectors(v1, v2):
    """Compute angle in rad between two vectors."""
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    # Asegurar que el valor de cos_theta esté en el rango [-1, 1] para evitar errores numéricos
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.arccos(cos_theta)
    return angle


@func_set_timeout(time_limit)
def test_mof_structure(struct, metal_center, radius, cutoff, dist_min, test_count=None):
    """Read and return if the MOF enviroment is correct from differents criterias."""
    rejected = False
    # Load structure
    mof = struct
    # select atoms from ligands
    ligands_indexs = []
    metal_index = []
    for at in mof:
        if at.symbol == metal_center:
            metal_index.append(at.index)
        elif at.symbol == "H":
            continue
        else:
            ligands_indexs.append(at.index)

    # ligands = mof[ligands_indexs]
    ligands = detect_ligand(mof[ligands_indexs], ligands_indexs)
    # Compute the neighbor list with a radius cutoff
    nl = NeighborList(
        [radius] * len(mof),
        self_interaction=False,
        bothways=True,
    )
    nl.update(mof)

    # Test connections ligands - metal
    for lig in ligands:
        symbols = mof[ligands[lig]["index"]].symbols
        print(f"Ligand:     {lig} - {symbols} (natoms: {len(symbols)})")

        pb_connections = 0
        lig_connectivity = {}
        list_offsets = []
        vectors_to_connect = []
        for lig_index in ligands[lig]["index"]:
            indices, offsets = nl.get_neighbors(lig_index)

            selection_m = []
            for i in indices:
                if i in metal_index:
                    selection_m.append(True)
                else:
                    selection_m.append(False)

            distances, box_offsets, directions, neighbor_position_list = get_neighbor_info(
                lig_index,
                mof,
                indices[selection_m],
                offsets[selection_m],
                cutoff,
                dist_min
            )
            # print(lig_index, distances, box_offsets, directions)
            lig_connectivity[lig_index] = {
                "distances": distances,
                "offsets": box_offsets,
                "metal_positions": neighbor_position_list
            }
            list_offsets += box_offsets
            vectors_to_connect += directions
            for _ in distances:
                pb_connections += 1

        angles_m = angle_matrix(vectors_to_connect)
        vectors_mean = np.array(vectors_to_connect).mean(axis=0) if np.array(vectors_to_connect).size > 0 else 0.0
        # COmprobar si estan conectaddos al mismo atomos de metal o no
        metals_neighbors = []
        for l_i in lig_connectivity:
            for val in lig_connectivity[l_i]["metal_positions"]:
                if val not in metals_neighbors:
                    metals_neighbors.append(val)
        # print(metals_neighbors)
        ligands[lig]["n_conn"] = pb_connections
        ligands[lig]["offsets"] = set(list_offsets)
        ligands[lig]["test_direction"] = np.dot(vectors_mean, vectors_mean)
        ligands[lig]["angle_mean"] = angles_m.mean() if angles_m.size > 0 else 0.0
        ligands[lig]["N_metals"] = len(metals_neighbors)

        # lignand_connect = pd.DataFrame(lig_connectivity).T
        # print(lignand_connect)

    info_ligand_connection = pd.DataFrame(ligands).T
    print("\nSummary of ligand atom connections:")
    print(info_ligand_connection)
    # Structural selection criteria
    # -----------------------------
    # I. Criteria for connectivity generated by cut-off distance.
    # At least one ligand of the system must be connected between two metal centers.
    #
    # II. Criteria for isolated ligands. Ligands not connected to anything.
    condition_I = (info_ligand_connection["N_metals"] >= 2).any()
    condition_II = (info_ligand_connection["N_metals"] == 0).any()
    print("")

    if condition_I and not condition_II:
        print("The structure has been preserved!")
        print("At least one ligand is connected to two different metal centers.")
    else:
        print("At least one ligand is not connected.")
        print("According to the selected criteria there is no metal-ligand-metal connection.")
        print("The structure will be rejected")
        print(f"{struct} --> rejected!")
        rejected = True

    return rejected


def duplicate_checker(result_queue, stop_event):
    """Verify duplicate."""
    # Initializing the StructureMatcher
    matcher = StructureMatcher(
        ltol=0.2,
        stol=0.3,
        angle_tol=5.0,
        primitive_cell=True,
        scale=True,
        attempt_supercell=True
    )

    # Create the RemoveDuplicatesFilter filter
    remove_duplcates = RemoveDuplicatesFilter(
               structure_matcher=matcher,
               symprec=1e-3
            )

    cpu = wabgen.core.get_cpu_num()
    print(f"\033[95mDuplicate checker living in cpu: {cpu}\033[0m")

    N_duplicates = 0
    while not stop_event.is_set() or not result_queue.empty():
        cpu = wabgen.core.get_cpu_num()
        try:
            generator_id, structure, response_q = result_queue.get(timeout=1)
            is_duplicate = not remove_duplcates.test(structure)
            response_q.put(is_duplicate)
            if is_duplicate:
                N_duplicates += 1
                print(f"\033[95mStructure duplicated in p({generator_id}) - N={N_duplicates} - cpu({cpu})\033[0m")
            else:
                # Structure preserved
                print(f"\033[95mStructure generated in p({generator_id}) is not a duplicate  - N={N_duplicates} - cpu({cpu})\033[0m")

        except queue.Empty:
            continue
