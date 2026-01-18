from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
import numpy as np

class ORToolsTSPSolver:
    def __init__(self, dist_matrix):
        self.B = np.array(dist_matrix)
        self.n = len(dist_matrix)

    def solve(self, time_limit_seconds=10):
        # FIX: Change IndexManager to RoutingIndexManager
        manager = pywrapcp.RoutingIndexManager(self.n, 1, 0) 

        # 2. Create Routing Model
        routing = pywrapcp.RoutingModel(manager)

        # 3. Create and register a transit callback
        def distance_callback(from_index, to_index):
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(self.B[from_node][to_node])

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # 4. Set search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = time_limit_seconds

        # 5. Solve the problem
        solution = routing.SolveWithParameters(search_parameters)

        if solution:
            path = []
            index = routing.Start(0)
            while not routing.IsEnd(index):
                path.append(manager.IndexToNode(index))
                index = solution.Value(routing.NextVar(index))
            path.append(manager.IndexToNode(index))
            return path, int(solution.ObjectiveValue())
        return None, None