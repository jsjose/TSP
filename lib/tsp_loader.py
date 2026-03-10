import numpy as np
import os
import re

class TSPInstance:
    def __init__(self, name, dimension, coords, distance_matrix, optimal_cost=None):
        self.name = name
        self.dimension = dimension
        self.coords = coords            # Nx2 numpy array (x, y)
        self.distance_matrix = distance_matrix  # NxN numpy array
        self.optimal_cost = optimal_cost # Float/Int (The best known distance)

    def __repr__(self):
        return (f"TSPInstance(name='{self.name}', "
                f"dimension={self.dimension}, "
                f"optimal_cost={self.optimal_cost})")

def read_tsp_problem(filepath):
    """
    Reads a TSPLIB .tsp file and returns metadata and coordinates.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    metadata = {}
    coords = []
    edge_weights = []
    display_coords = []
    
    current_section = None
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == 'EOF':
                break

            # Detect section start
            if line.startswith('NODE_COORD_SECTION'):
                current_section = 'COORD'
                continue
            elif line.startswith('EDGE_WEIGHT_SECTION'):
                current_section = 'WEIGHTS'
                continue
            elif line.startswith('DISPLAY_DATA_SECTION'):
                current_section = 'DISPLAY'
                continue
            
            # Heuristic: Stop reading data section if we hit a metadata line (contains :)
            # This handles files where metadata might appear after data (rare but possible)
            if current_section and ':' in line and not line.split()[0].replace('.', '').isdigit():
                 # However, be careful not to trigger on data lines. 
                 # TSPLIB data usually doesn't have colons.
                 current_section = None

            if current_section == 'COORD' or current_section == 'DISPLAY':
                parts = re.split(r'\s+', line)
                if len(parts) >= 3:
                    try:
                        # TSPLIB format: ID X Y
                        x, y = float(parts[1]), float(parts[2])
                        if current_section == 'COORD':
                            coords.append([x, y])
                        else:
                            display_coords.append([x, y])
                    except ValueError:
                        continue
            elif current_section == 'WEIGHTS':
                # Parse flat numbers for the matrix
                parts = re.split(r'\s+', line)
                for p in parts:
                    try:
                        edge_weights.append(float(p))
                    except ValueError:
                        continue
            elif current_section is None:
                # Parse Metadata
                if ':' in line:
                    key, value = line.split(':', 1)
                    metadata[key.strip()] = value.strip()
    
    # If explicit coords are missing but display coords exist (common in EXPLICIT problems), use display coords
    if len(coords) == 0 and len(display_coords) > 0:
        coords = display_coords

    coords_array = np.array(coords, dtype=np.float32)
    edge_weights_array = np.array(edge_weights, dtype=np.float32)
    return metadata, coords_array, edge_weights_array

def calculate_euc_2d_matrix(coords):
    """
    Calculates Euclidean distance matrix (rounded to nearest integer).
    """
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))
    return np.round(dist_matrix).astype(int)

def convert_geo_coord(coordinate):
    """
    Converts TSPLIB GEO format (degrees.minutes) to radians.
    """
    deg = np.fix(coordinate)
    min_val = coordinate - deg
    rad = np.pi * (deg + 5.0 * min_val / 3.0) / 180.0
    return rad

def calculate_geo_matrix(coords):
    """
    Calculates distance matrix for GEO coordinates (TSPLIB format).
    """
    # coords[:, 0] is latitude, coords[:, 1] is longitude
    lats = convert_geo_coord(coords[:, 0])
    lons = convert_geo_coord(coords[:, 1])
    
    q1 = np.cos(lons[:, np.newaxis] - lons[np.newaxis, :])
    q2 = np.cos(lats[:, np.newaxis] - lats[np.newaxis, :])
    q3 = np.cos(lats[:, np.newaxis] + lats[np.newaxis, :])
    
    val = 0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)
    val = np.clip(val, -1.0, 1.0)
    
    RRR = 6378.388
    dist_matrix = (RRR * np.arccos(val) + 1.0).astype(int)
    np.fill_diagonal(dist_matrix, 0)
    
    return dist_matrix

def calculate_att_matrix(coords):
    """
    Calculates pseudo-Euclidean distance matrix for EDGE_WEIGHT_TYPE: ATT.
    d_ij = ceil( sqrt( (dx^2 + dy^2) / 10 ) )
    """
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    # Calculate sqrt( (dx^2 + dy^2) / 10 )
    dist = np.sqrt(np.sum(diff**2, axis=-1) / 10.0)
    # TSPLIB ATT rounds up to the nearest integer
    return np.ceil(dist).astype(int)

def load_solutions_file(filepath):
    """
    Parses a single file containing 'ProblemName : Cost' on each line.
    Returns a dictionary: {'a280': 2579, 'att48': 10628, ...}
    """
    solutions = {}
    if not os.path.exists(filepath):
        print(f"Warning: Solutions file not found at {filepath}")
        return solutions

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            
            # Format: name : cost
            parts = line.split(':')
            if len(parts) == 2:
                name = parts[0].strip()
                try:
                    cost = float(parts[1].strip()) # Use float to be safe, or int
                    solutions[name] = cost
                except ValueError:
                    continue
    return solutions

def load_tsp(problem_path, solutions_dict=None):
    """
    Loads a TSP problem and automatically finds its optimal cost from a provided dictionary.
    
    Args:
        problem_path (str): Path to the .tsp file.
        solutions_dict (dict): Dictionary loaded from load_solutions_file().
    """
    # 1. Read Problem
    metadata, coords, edge_weights = read_tsp_problem(problem_path)
    name = metadata.get('NAME', 'Unknown')
    dim = int(metadata.get('DIMENSION', len(coords)))
    
    # 2. Calculate Matrix
    edge_weight_type = metadata.get('EDGE_WEIGHT_TYPE', 'EUC_2D').strip()
    
    if edge_weight_type == 'EXPLICIT':
        # Handle Explicit Matrix (e.g., bayg29)
        fmt = metadata.get('EDGE_WEIGHT_FORMAT', 'FULL_MATRIX').strip()
        if fmt == 'FULL_MATRIX':
            if len(edge_weights) == dim * dim:
                dist_matrix = edge_weights.reshape((dim, dim)).astype(int)
            else:
                print(f"Warning: Explicit matrix size mismatch. Expected {dim*dim}, got {len(edge_weights)}")
                dist_matrix = np.zeros((dim, dim))
        elif fmt == 'UPPER_ROW':
            dist_matrix = np.zeros((dim, dim), dtype=int)
            k = 0
            for i in range(dim - 1):
                for j in range(i + 1, dim):
                    if k < len(edge_weights):
                        val = int(edge_weights[k])
                        dist_matrix[i, j] = val
                        dist_matrix[j, i] = val
                        k += 1
        elif fmt == 'LOWER_DIAG_ROW':
            dist_matrix = np.zeros((dim, dim), dtype=int)
            k = 0
            for i in range(dim):
                for j in range(i + 1):
                    if k < len(edge_weights):
                        val = int(edge_weights[k])
                        dist_matrix[i, j] = val
                        dist_matrix[j, i] = val
                        k += 1
        else:
            # Placeholder for UPPER_ROW, LOWER_DIAG, etc.
            print(f"Warning: Unsupported EDGE_WEIGHT_FORMAT: {fmt}")
            dist_matrix = np.zeros((dim, dim))
    elif edge_weight_type == 'GEO':
        dist_matrix = calculate_geo_matrix(coords)
    elif edge_weight_type == 'ATT':
        dist_matrix = calculate_att_matrix(coords)
    else:
        dist_matrix = calculate_euc_2d_matrix(coords)
    
    # 3. Look up solution
    opt_cost = None
    if solutions_dict and name in solutions_dict:
        opt_cost = solutions_dict[name]
    
    return TSPInstance(
        name=name,
        dimension=dim,
        coords=coords,
        distance_matrix=dist_matrix,
        optimal_cost=opt_cost
    )