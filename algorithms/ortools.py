import numpy as np
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2


class ORToolsTSPSolver:
    """TSP solver wrapping Google OR-Tools with Guided Local Search.

    Uses PATH_CHEAPEST_ARC as the initial solution strategy and
    Guided Local Search as the improvement metaheuristic.

    Args:
        dist_matrix: NxN distance/cost matrix.
    """

    def __init__(self, dist_matrix) -> None:
        self.B = np.array(dist_matrix)
        self.n = len(dist_matrix)

    def solve(
        self, time_limit_seconds: int = 10
    ) -> tuple[list[int] | None, int | None]:
        """Runs OR-Tools routing solver.

        Args:
            time_limit_seconds: Wall-clock time limit for the solver.

        Returns:
            (path, cost): Best tour found and its integer cost,
            or (None, None) if no solution was found.
        """
        manager = pywrapcp.RoutingIndexManager(self.n, 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(self.B[from_node][to_node])

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = time_limit_seconds

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
