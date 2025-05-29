from abc import ABC, abstractmethod
import time # For time limit checking
import math # For infinity, exp
import random # For initial solution generation (optional)
import os # For creating directory
import matplotlib.pyplot as plt # Added for plotting
from data_structures import ProblemInstance, Customer, Solution # Updated Solution import
import copy # For deepcopying solutions
import collections # For deque (Tabu List)

class Solver(ABC):
    """
    Abstract base class for all VRP solvers.
    """
    def __init__(self, problem_instance: ProblemInstance, time_limit_seconds: int):
        self.problem_instance = problem_instance
        self.time_limit_seconds = time_limit_seconds
        self.best_solution: Solution | None = None
        self.convergence_data = [] 
        self.start_time = 0.0

    @abstractmethod
    def solve(self) -> Solution | None:
        """
        Solves the VRP instance and returns the best found solution.
        Populates self.convergence_data during the solving process.
        """
        self.start_time = time.time()
        pass

    def _is_time_limit_reached(self) -> bool:
        return (time.time() - self.start_time) >= self.time_limit_seconds

    def _generate_greedy_initial_solution(self) -> Solution:
        """
        Generates a greedy initial solution.
        Assigns customers to routes one by one, trying to fit them into existing routes 
        or creating new ones if necessary. This is a very basic heuristic.
        """
        solution = Solution(self.problem_instance)
        customers_to_assign = [c for c in self.problem_instance.customers if not c.is_depot]
        customers_to_assign.sort(key=lambda c: c.ready_time) 

        num_vehicles_available = self.problem_instance.num_vehicles

        for _ in range(num_vehicles_available):
            if not customers_to_assign: break 

            current_route_ids = [] 
            
            temp_solution_for_route_check = Solution(self.problem_instance)
            
            for customer in list(customers_to_assign):
                potential_new_route_ids = current_route_ids + [customer.id]
                temp_solution_for_route_check.routes = [potential_new_route_ids] 
                

                _cost, _overall_sol_feasible, route_details_list = temp_solution_for_route_check.calculate_cost_and_feasibility(update_self=False)
                
                current_route_is_intrinsically_feasible = False
                if route_details_list and len(route_details_list) == 1: 
                    details_of_current_route = route_details_list[0]
                    if not details_of_current_route.get('time_violation', False) and \
                       not details_of_current_route.get('capacity_violation', False):
                        current_route_is_intrinsically_feasible = True

                if current_route_is_intrinsically_feasible:
                    current_route_ids.append(customer.id)
                    customers_to_assign.remove(customer)
            
            if current_route_ids: 
                solution.routes.append(current_route_ids)
        
        if customers_to_assign: 
            while customers_to_assign and len(solution.routes) < num_vehicles_available:
                customer_to_add = customers_to_assign.pop(0) 
                solution.routes.append([customer_to_add.id]) 


        if customers_to_assign:
            pass 

        solution.calculate_cost_and_feasibility()
        return solution

    def _initialize_solution(self) -> Solution:
        """
        Creates an initial solution, currently using the greedy heuristic.
        """
        return self._generate_greedy_initial_solution()

    def _get_all_customer_locations(self, solution: Solution) -> list[tuple[int, int, int]]:
        """Helper to get (customer_id, route_idx, pos_in_route) for all non-depot customers in routes."""
        locations = []
        if solution and solution.routes:
            for r_idx, route in enumerate(solution.routes):
                for c_pos, cust_id in enumerate(route):
                    customer_obj = self.problem_instance.get_customer_by_id(cust_id)
                    if customer_obj and not customer_obj.is_depot:
                        locations.append((cust_id, r_idx, c_pos))
        return locations

    def _report_solution(self, solution: Solution, algorithm_name: str):
        """Prints the final solution details to the terminal."""
        print(f"\n--- {algorithm_name} - Final Solution Report ---")
        print(f"Instance: {self.problem_instance.name}")
        if solution and solution.is_feasible:
            print(f"Algorithm: {algorithm_name}")
            print(f"Total Distance: {solution.total_distance:.2f}")
            print(f"Number of Routes: {len(solution.routes)}")
            for i, route_detail in enumerate(solution.route_details):
                route_cust_ids = [self.problem_instance.depot.id] + route_detail['customer_ids'] + [self.problem_instance.depot.id]
                print(f"  Route {i+1}: {' -> '.join(map(str, route_cust_ids))}")
                print(f"    Distance: {route_detail['route_distance']:.2f}")
                print(f"    Load: {route_detail['route_load']}")
        elif solution: 
            print(f"Algorithm: {algorithm_name} - Found an INFEASIBLE solution.")
            print(f"Total Distance (raw): {solution.total_distance:.2f} (may include penalties or be sum of infeasible routes)")
            print(f"Number of Routes Attempted: {len(solution.routes)}")
            for i, route_detail in enumerate(solution.route_details):
                route_cust_ids = [self.problem_instance.depot.id] + route_detail['customer_ids'] + [self.problem_instance.depot.id]
                print(f"  Route {i+1}: {' -> '.join(map(str, route_cust_ids))}")
                print(f"    Distance: {route_detail['route_distance']:.2f}, Load: {route_detail['route_load']}")
                if route_detail.get('capacity_violation', False): print("    Capacity VIOLATED")
                if route_detail.get('time_violation', False): print("    Time VIOLATED")
        else:
            print(f"Algorithm: {algorithm_name} - No solution found.")
        print("-----------------------------------------")

    def save_solution_to_txt(self, solution: Solution, algorithm_name: str, output_folder: str = "solutions"):
        """Saves the solution details to a .txt file."""
        if not solution:
            print(f"No solution to save for {algorithm_name} on {self.problem_instance.name}.")
            return

        if not os.path.exists(output_folder):
            try:
                os.makedirs(output_folder)
            except OSError as e:
                print(f"Error creating directory {output_folder}: {e}")
                return

        filename = f"{self.problem_instance.name}_{algorithm_name}_solution.txt"
        filepath = os.path.join(output_folder, filename)

        try:
            with open(filepath, 'w') as f:
                f.write(f"Instance Name: {self.problem_instance.name}\n")
                f.write(f"Algorithm Used: {algorithm_name}\n")
                f.write(f"Total Objective Value (Distance): {solution.total_distance:.2f}\n")
                f.write(f"Solution Feasible: {solution.is_feasible}\n\n")

                if solution.routes:
                    f.write(f"Number of Routes Used: {len(solution.routes)}\n\n")
                    for i, route_detail in enumerate(solution.route_details):
                        route_cust_ids_display = [self.problem_instance.depot.id] + \
                                                 route_detail['customer_ids'] + \
                                                 [self.problem_instance.depot.id]
                        
                        f.write(f"Vehicle ID (Route No.): {i+1}\n")
                        f.write(f"  Route: {' -> '.join(map(str, route_cust_ids_display))}\n")
                        f.write(f"  Total demand served by vehicle: {route_detail['route_load']}\n")
                        f.write(f"  Total distance of vehicle's route: {route_detail['route_distance']:.2f}\n")
                        f.write(f"  Route Capacity Violation: {route_detail.get('capacity_violation', False)}\n")
                        f.write(f"  Route Time Violation: {route_detail.get('time_violation', False)}\n")
                        
                        if route_detail.get('stops_details'):
                            f.write(f"  Stop Details:\n")
                            for stop_info in route_detail['stops_details']:
                                f.write(f"    - Cust {stop_info['customer_id']}: Arrival={stop_info['arrival']:.2f}, StartServ={stop_info['service_start']:.2f}, Depart={stop_info['departure']:.2f}, LoadAfter={stop_info['current_load_after_service']}\n")
                        f.write("\n")
                else:
                    f.write("No routes found in the solution.\n")
            print(f"Solution for {algorithm_name} saved to {filepath}")
        except IOError as e:
            print(f"Error writing solution to file {filepath}: {e}")

    def plot_convergence_graph(self, algorithm_name: str, output_folder: str = "graphs"):
        """Plots the convergence graph and saves it to a file."""
        if not self.convergence_data:
            print(f"No convergence data to plot for {algorithm_name} on {self.problem_instance.name}.")
            return

        if not os.path.exists(output_folder):
            try:
                os.makedirs(output_folder)
            except OSError as e:
                print(f"Error creating directory {output_folder}: {e}")
                return

        valid_convergence_data = [(t, val) for t, val in self.convergence_data if isinstance(val, (int, float)) and math.isfinite(val)]
        if not valid_convergence_data:
            print(f"No valid (finite) objective values in convergence data for {algorithm_name} on {self.problem_instance.name}.")
            return

        times_or_iterations = [item[0] for item in valid_convergence_data]
        objective_values = [item[1] for item in valid_convergence_data]

        plt.figure(figsize=(10, 6))
        plt.plot(times_or_iterations, objective_values, marker='.', linestyle='-')
        plt.title(f"Convergence of {algorithm_name} for {self.problem_instance.name}")
        plt.xlabel("Time (seconds) or Iteration")
        plt.ylabel("Total Distance (Objective Value)")
        plt.grid(True)
        plt.tight_layout()

        filename = f"{self.problem_instance.name}_{algorithm_name}_convergence.png"
        filepath = os.path.join(output_folder, filename)
        try:
            plt.savefig(filepath)
            print(f"Convergence graph for {algorithm_name} saved to {filepath}")
        except Exception as e:
            print(f"Error saving convergence graph to {filepath}: {e}")
        plt.close() 

class SimulatedAnnealingSolver(Solver):
    def __init__(self, problem_instance: ProblemInstance, time_limit_seconds: int, 
                 initial_temp: float = 10000, cooling_rate: float = 0.998, min_temp: float = 0.01):
        super().__init__(problem_instance, time_limit_seconds)
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp

    def _generate_neighbor_sa(self, current_solution: Solution) -> Solution:
        """Generates a neighbor solution for SA by applying a random operator."""
        neighbor_solution = copy.deepcopy(current_solution)


        operators = [self._relocate_customer_sa, self._swap_customers_sa, self._two_opt_sa]
        chosen_operator = random.choice(operators)


        MAX_OPERATOR_ATTEMPTS = 10
        for _ in range(MAX_OPERATOR_ATTEMPTS):
            modified = chosen_operator(neighbor_solution) 
            if modified:
                break 
        
        neighbor_solution.calculate_cost_and_feasibility() 
        return neighbor_solution

    def _relocate_customer_sa(self, solution: Solution) -> bool:
        """Relocates a random customer to a new position (intra-route or inter-route). Modifies solution.routes.
           Returns True if a modification was made, False otherwise."""
        
        customer_locations = self._get_all_customer_locations(solution)
        if not customer_locations:
            return False 

        cust_id_to_relocate, original_route_idx, original_pos_idx = random.choice(customer_locations)

        original_route = solution.routes[original_route_idx]
        original_route.pop(original_pos_idx)

        num_routes = len(solution.routes)
        if num_routes == 0: 
            solution.routes[original_route_idx].insert(original_pos_idx, cust_id_to_relocate)
            return False
        
        target_route_idx = random.randrange(num_routes)
        target_route = solution.routes[target_route_idx]

        max_pos_in_target = len(target_route)
        target_pos_idx = random.randint(0, max_pos_in_target) 
        
        target_route.insert(target_pos_idx, cust_id_to_relocate)

        return True

    def _swap_customers_sa(self, solution: Solution) -> bool:
        """Swaps two random customers (intra-route or inter-route). Modifies solution.routes.
           Returns True if a modification was made, False otherwise."""
        customer_locations = self._get_all_customer_locations(solution)
        if len(customer_locations) < 2:
            return False 

        loc1_idx, loc2_idx = random.sample(range(len(customer_locations)), 2)
        cust1_id, route1, pos1 = customer_locations[loc1_idx]
        cust2_id, route2, pos2 = customer_locations[loc2_idx]

        solution.routes[route1][pos1], solution.routes[route2][pos2] = \
            solution.routes[route2][pos2], solution.routes[route1][pos1]
        
        return True

    def _two_opt_sa(self, solution: Solution) -> bool:
        """Performs a 2-opt move on a randomly selected route. Returns True if a modification was made."""
        candidate_routes = [route for route in solution.routes if len(route) >= 4]
        if not candidate_routes:
            return False
        route_idx = random.choice([i for i, route in enumerate(solution.routes) if len(route) >= 4])
        route = solution.routes[route_idx]
        n = len(route)
        i = random.randint(0, n - 3)
        j = random.randint(i + 2, n - 1)
        route[i:j+1] = reversed(route[i:j+1])
        return True

    def solve(self) -> Solution | None:
        super().solve() 
        print(f"Simulated Annealing started for {self.problem_instance.name} with time limit {self.time_limit_seconds}s.")
        print(f"Params: Initial Temp={self.initial_temp}, Cooling Rate={self.cooling_rate}, Min Temp={self.min_temp}")
        
        current_solution = self._initialize_solution()
        if not current_solution.routes and any(not c.is_depot for c in self.problem_instance.customers):
            print(f"SA: Initial solution for {self.problem_instance.name} is empty and there are customers to serve. Aborting SA.")
            self._report_solution(current_solution, "SimulatedAnnealing_NoInitialSol")
            return current_solution 
        
        self.best_solution = copy.deepcopy(current_solution)


        if current_solution.total_distance != float('inf'):
             self.convergence_data.append((0, current_solution.total_distance))
        
        print(f"SA Initial Solution: Cost={current_solution.total_distance:.2f}, Feasible={current_solution.is_feasible}")


        current_temp = self.initial_temp
        iteration = 0

        while current_temp > self.min_temp and not self._is_time_limit_reached():
            neighbor_solution = self._generate_neighbor_sa(current_solution)

            current_cost = current_solution.total_distance
            neighbor_cost = neighbor_solution.total_distance

            cost_delta = neighbor_cost - current_cost

            if cost_delta < 0:
                current_solution = neighbor_solution
                if neighbor_cost < self.best_solution.total_distance:
                    self.best_solution = copy.deepcopy(neighbor_solution)
                elif not self.best_solution.is_feasible and neighbor_solution.is_feasible:
                     self.best_solution = copy.deepcopy(neighbor_solution)
                elif neighbor_cost == self.best_solution.total_distance and not self.best_solution.is_feasible and neighbor_solution.is_feasible:
                     self.best_solution = copy.deepcopy(neighbor_solution)

            else:
                acceptance_probability = math.exp(-cost_delta / current_temp)
                if random.random() < acceptance_probability:
                    current_solution = neighbor_solution
            
            
            if current_solution.total_distance < self.best_solution.total_distance:
                self.best_solution = copy.deepcopy(current_solution)
            elif not self.best_solution.is_feasible and current_solution.is_feasible:
                self.best_solution = copy.deepcopy(current_solution)
            elif current_solution.total_distance == self.best_solution.total_distance and not self.best_solution.is_feasible and current_solution.is_feasible:
                self.best_solution = copy.deepcopy(current_solution)

            current_temp *= self.cooling_rate
            iteration += 1

            if iteration % 100 == 0: 
                if self.best_solution and self.best_solution.total_distance != float('inf'):
                    time_elapsed = time.time() - self.start_time
                    self.convergence_data.append((time_elapsed, self.best_solution.total_distance))

        if self.best_solution and self.best_solution.total_distance != float('inf'):
            time_elapsed = time.time() - self.start_time
            self.convergence_data.append((time_elapsed, self.best_solution.total_distance))

        self._report_solution(self.best_solution, "SimulatedAnnealing")
        self.save_solution_to_txt(self.best_solution, "SA")
        self.plot_convergence_graph("SA")
        print(f"Simulated Annealing finished. Iterations: {iteration}, Final Temp: {current_temp:.2f}")
        return self.best_solution

class TabuSearchSolver(Solver):
    def __init__(self, problem_instance: ProblemInstance, time_limit_seconds: int,
                 tabu_list_size: int = 20, max_iterations_without_improvement: int = 100):
        super().__init__(problem_instance, time_limit_seconds)
        self.tabu_list_size = tabu_list_size
        self.max_iterations_without_improvement = max_iterations_without_improvement
        self.tabu_list = collections.deque(maxlen=self.tabu_list_size)
        
    def _is_move_tabu(self, move_representation: tuple) -> bool:
        """Check if a move is in the tabu list."""
        return move_representation in self.tabu_list

    def _add_move_to_tabu_list(self, move_representation: tuple):
        """Add a move to the tabu list."""
        if move_representation not in self.tabu_list:  
            if len(self.tabu_list) >= self.tabu_list_size:
                self.tabu_list.popleft()  
            self.tabu_list.append(move_representation)

    def _calculate_move_cost(self, solution: Solution) -> float:
        """Calculate solution cost with softer penalties for infeasibility."""
        if not solution or solution.total_distance == float('inf'):
            return float('inf')
            
        base_cost = solution.total_distance
        penalty = 0
        
        if not solution.is_feasible:
            capacity_violations = 0
            time_violations = 0
            for route_detail in solution.route_details:
                if route_detail.get('capacity_violation'):
                    capacity_violations += 1
                if route_detail.get('time_violation'):
                    time_violations += 1
            
            penalty = (capacity_violations + time_violations) * 1000
            
        return base_cost + penalty

    def _generate_neighborhood_ts(self, current_solution: Solution) -> list[tuple[Solution, tuple]]:
        """Generate neighborhood with focus on maintaining feasibility."""
        neighbors = []
        if not current_solution.routes:
            return neighbors

        locations = []
        for r_idx, route in enumerate(current_solution.routes):
            for c_pos, cust_id in enumerate(route):
                if cust_id != self.problem_instance.depot.id:
                    locations.append((cust_id, r_idx, c_pos))


        num_moves = 20 if current_solution.is_feasible else 30
        
        for _ in range(num_moves):
            if not locations:
                break
                
            move_type = random.choices(['relocate', 'swap', '2opt'], 
                                     weights=[0.5, 0.3, 0.2])[0]
            
            neighbor = copy.deepcopy(current_solution)
            move_info = None
            
            if move_type == 'relocate' and locations:
                cust_id, from_route, from_pos = random.choice(locations)
                
                best_cost = float('inf')
                best_move = None
                
                possible_routes = list(range(len(neighbor.routes)))
                if len(possible_routes) > 3:
                    possible_routes = random.sample(possible_routes, 3)
                
                for to_route in possible_routes:
                    route_len = len(neighbor.routes[to_route])
                    possible_positions = list(range(route_len + 1))
                    if len(possible_positions) > 5:
                        possible_positions = random.sample(possible_positions, 5)
                    
                    for to_pos in possible_positions:
                        if to_route == from_route and (to_pos == from_pos or to_pos == from_pos + 1):
                            continue
                            
                        test_neighbor = copy.deepcopy(neighbor)
                        customer = test_neighbor.routes[from_route].pop(from_pos)
                        test_neighbor.routes[to_route].insert(to_pos, customer)
                        
                        if not test_neighbor.routes[from_route]:
                            del test_neighbor.routes[from_route]
                            if to_route > from_route:
                                to_route -= 1
                                
                        test_neighbor.calculate_cost_and_feasibility()
                        cost = self._calculate_move_cost(test_neighbor)
                        
                        if cost < best_cost:
                            best_cost = cost
                            best_move = (to_route, to_pos, test_neighbor)
                
                if best_move:
                    to_route, to_pos, neighbor = best_move
                    move_info = ('relocate', cust_id, from_route, from_pos, to_route, to_pos)
            
            elif move_type == 'swap' and len(locations) >= 2:
                idx1, idx2 = random.sample(range(len(locations)), 2)
                cust1_id, route1, pos1 = locations[idx1]
                cust2_id, route2, pos2 = locations[idx2]
                
                neighbor.routes[route1][pos1], neighbor.routes[route2][pos2] = \
                    neighbor.routes[route2][pos2], neighbor.routes[route1][pos1]
                    
                move_info = ('swap', cust1_id, route1, pos1, cust2_id, route2, pos2)
            
            elif move_type == '2opt':
                valid_routes = [i for i, r in enumerate(neighbor.routes) if len(r) >= 4]
                if valid_routes:
                    route_idx = random.choice(valid_routes)
                    route = neighbor.routes[route_idx]
                    
                    best_cost = float('inf')
                    best_2opt = None
                    
                    for _ in range(5):
                        i = random.randint(0, len(route) - 3)
                        j = random.randint(i + 2, len(route) - 1)
                        
                        test_neighbor = copy.deepcopy(neighbor)
                        test_neighbor.routes[route_idx][i:j+1] = reversed(test_neighbor.routes[route_idx][i:j+1])
                        
                        test_neighbor.calculate_cost_and_feasibility()
                        cost = self._calculate_move_cost(test_neighbor)
                        
                        if cost < best_cost:
                            best_cost = cost
                            best_2opt = (i, j, test_neighbor)
                    
                    if best_2opt:
                        i, j, neighbor = best_2opt
                        move_info = ('2opt', route_idx, i, j)
            
            if move_info:
                neighbor.calculate_cost_and_feasibility()
                neighbors.append((neighbor, move_info))
        
        return neighbors

    def solve(self) -> Solution | None:
        """Main tabu search algorithm."""
        self.start_time = time.time()
        print(f"Tabu Search started for {self.problem_instance.name} with time limit {self.time_limit_seconds}s.")
        print(f"Params: Tabu List Size={self.tabu_list_size}, Max Iter w/o Improv={self.max_iterations_without_improvement}")
        
        current_solution = self._initialize_solution()
        if not current_solution.routes:
            print(f"TS: Initial solution for {self.problem_instance.name} is empty. Aborting TS.")
            return current_solution
            
        self.best_solution = copy.deepcopy(current_solution)
        self.tabu_list.clear()
        
        current_cost = self._calculate_move_cost(current_solution)
        best_cost = current_cost
        
        print(f"TS Initial Solution: Cost={current_solution.total_distance:.2f}, Feasible={current_solution.is_feasible}")
        self.convergence_data.append((0, current_solution.total_distance))
        
        iteration = 0
        iterations_without_improvement = 0
        
        while not self._is_time_limit_reached() and iterations_without_improvement < self.max_iterations_without_improvement:
            iteration += 1
            
            neighbors = self._generate_neighborhood_ts(current_solution)
            if not neighbors:
                break
                
            best_neighbor = None
            best_neighbor_cost = float('inf')
            best_move = None
            
            for neighbor, move in neighbors:
                neighbor_cost = self._calculate_move_cost(neighbor)
                is_tabu = self._is_move_tabu(move)
                
                if not is_tabu or neighbor_cost < best_cost:
                    if neighbor_cost < best_neighbor_cost:
                        best_neighbor = neighbor
                        best_neighbor_cost = neighbor_cost
                        best_move = move
            
            if not best_neighbor:
                break
                
            current_solution = best_neighbor
            current_cost = best_neighbor_cost
            
            if current_cost < best_cost:
                self.best_solution = copy.deepcopy(current_solution)
                best_cost = current_cost
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1
            
            if best_move:
                self._add_move_to_tabu_list(best_move)
            
            if current_solution.is_feasible:
                self.convergence_data.append((time.time() - self.start_time, current_solution.total_distance))
            
            if iteration % 10 == 0:
                print(f"TS Iter: {iteration}, Current Cost: {current_solution.total_distance:.2f} "
                      f"(F:{current_solution.is_feasible}), Best Cost: {self.best_solution.total_distance:.2f} "
                      f"(F:{self.best_solution.is_feasible}), Iter w/o Improvement: {iterations_without_improvement}")
        
        if self.best_solution:
            self.convergence_data.append((time.time() - self.start_time, self.best_solution.total_distance))
        
        self._report_solution(self.best_solution, "TabuSearch")
        self.save_solution_to_txt(self.best_solution, "TS")
        self.plot_convergence_graph("TS")
        
        print(f"Tabu Search finished. Iterations: {iteration}, Iterations w/o improvement: {iterations_without_improvement}")
        return self.best_solution

class GeneticAlgorithmSolver(Solver):
    def __init__(self, problem_instance: ProblemInstance, time_limit_seconds: int,
                 population_size: int = 50, crossover_rate: float = 0.8, 
                 mutation_rate: float = 0.1, num_generations: int = 100, 
                 tournament_size: int = 5, elitism_count: int = 2):
        super().__init__(problem_instance, time_limit_seconds)
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.num_generations = num_generations 
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        if self.elitism_count >= self.population_size: 
            self.elitism_count = max(0, self.population_size -1) 

    def _calculate_fitness(self, solution: Solution) -> float:
        """Calculates fitness. Lower cost (distance) is better fitness.
           The solution.total_distance should already include penalties from calculate_cost_and_feasibility.
        """
        if solution.total_distance == float('inf'):
            return float('inf') 
        cost = solution.total_distance 
        return cost

    def _initialize_population_ga(self) -> list[Solution]:
        population = []
        for _ in range(self.population_size):
            sol = self._initialize_solution()
            population.append(sol)
        return population

    def _selection_ga(self, population: list[Solution], fitnesses: list[float]) -> list[Solution]:
        """Selects parents from the population using tournament selection."""
        selected_parents = []
        for _ in range(len(population)):
            tournament_contenders_indices = random.sample(range(len(population)), self.tournament_size)
            best_contender_idx_in_tournament = -1
            best_fitness_in_tournament = float('inf')
            
            for contender_idx in tournament_contenders_indices:
                if fitnesses[contender_idx] < best_fitness_in_tournament:
                    best_fitness_in_tournament = fitnesses[contender_idx]
                    best_contender_idx_in_tournament = contender_idx
            
            if best_contender_idx_in_tournament != -1:
                 selected_parents.append(population[best_contender_idx_in_tournament])
            else:
                 selected_parents.append(random.choice(population))
        return selected_parents

    def _repair_solution_ga(self, solution: Solution, all_customer_ids: set[int]):
        """Repairs a solution after crossover to ensure all customers are served exactly once
           and attempts to maintain feasibility. Modifies solution.routes in place.
           all_customer_ids should be a set of all non-depot customer IDs from the problem instance.
        """
        visited_in_routes = []
        customers_in_current_solution_routes = set()
        for r_idx, route in enumerate(solution.routes):
            unique_customers_in_route = []
            seen_in_this_route = set()
            for cust_id in route:
                if cust_id not in seen_in_this_route:
                    unique_customers_in_route.append(cust_id)
                    seen_in_this_route.add(cust_id)
            solution.routes[r_idx] = unique_customers_in_route 
            for cust_id in solution.routes[r_idx]:
                if cust_id in customers_in_current_solution_routes:
                    pass 
                customers_in_current_solution_routes.add(cust_id)
                visited_in_routes.append(cust_id) 
        served_counts = collections.Counter(visited_in_routes)
        unserved_customers = list(all_customer_ids - customers_in_current_solution_routes)
        multi_served_customers = {cust_id for cust_id, count in served_counts.items() if count > 1}
        if multi_served_customers:
            kept_instances_of_multiserved = set() 
            for r_idx in range(len(solution.routes) -1, -1, -1): 
                current_route = solution.routes[r_idx]
                c_idx = len(current_route) - 1
                while c_idx >= 0:
                    cust_id = current_route[c_idx]
                    if cust_id in multi_served_customers:
                        if cust_id not in kept_instances_of_multiserved:
                            kept_instances_of_multiserved.add(cust_id)
                        else:
                            current_route.pop(c_idx)
                            if cust_id not in unserved_customers:
                                pass 
                    c_idx -= 1
        current_customers_flat = {cust for route in solution.routes for cust in route}
        unserved_customers = list(all_customer_ids - current_customers_flat)
        random.shuffle(unserved_customers)
        for cust_to_insert in unserved_customers:
            best_insertion_cost = float('inf')
            best_route_idx = -1
            best_pos_idx = -1
            insertion_found = False
            for r_idx, route in enumerate(solution.routes):
                for pos in range(len(route) + 1):
                    temp_route = route[:pos] + [cust_to_insert] + route[pos:]
                    temp_sol_for_check = Solution(self.problem_instance, routes=[temp_route])
                    cost, feasible, _ = temp_sol_for_check.calculate_cost_and_feasibility(update_self=False)
                    if feasible:
                        if cost < best_insertion_cost:
                            best_insertion_cost = cost
                            best_route_idx = r_idx
                            best_pos_idx = pos
                            insertion_found = True
            if insertion_found:
                solution.routes[best_route_idx].insert(best_pos_idx, cust_to_insert)
        solution.routes = [route for route in solution.routes if route]
        if len(solution.routes) > self.problem_instance.num_vehicles:
            solution.routes = solution.routes[:self.problem_instance.num_vehicles]
        solution.calculate_cost_and_feasibility()

    def _crossover_ga(self, parent1: Solution, parent2: Solution) -> tuple[Solution, Solution]:
        """Performs crossover between two parents to produce two offspring."""
        offspring1_sol = Solution(self.problem_instance)
        offspring2_sol = Solution(self.problem_instance)
        all_customer_ids = {c.id for c in self.problem_instance.customers if not c.is_depot}
        p1_routes = copy.deepcopy(parent1.routes)
        p2_routes = copy.deepcopy(parent2.routes)
        random.shuffle(p1_routes)
        num_routes_from_p1 = random.randint(1, max(1, len(p1_routes) // 2 if len(p1_routes) > 1 else 1))
        temp_o1_routes = []
        o1_served_customers = set()
        for i in range(min(num_routes_from_p1, self.problem_instance.num_vehicles)):
            if p1_routes:
                route = p1_routes.pop(0)
                temp_o1_routes.append(route)
                for cust_id in route:
                    o1_served_customers.add(cust_id)
        random.shuffle(p2_routes)
        for p2_route in p2_routes:
            if len(temp_o1_routes) >= self.problem_instance.num_vehicles: break
            route_to_add_for_o1 = []
            can_add_this_p2_route = False
            for cust_id in p2_route:
                if cust_id not in o1_served_customers:
                    route_to_add_for_o1.append(cust_id)
                    can_add_this_p2_route = True 
            if route_to_add_for_o1 and can_add_this_p2_route:
                temp_o1_routes.append(route_to_add_for_o1)
                for cust_id in route_to_add_for_o1:
                     o1_served_customers.add(cust_id)
        offspring1_sol.routes = temp_o1_routes
        self._repair_solution_ga(offspring1_sol, all_customer_ids)
        p1_routes_copy = copy.deepcopy(parent1.routes)
        p2_routes_copy = copy.deepcopy(parent2.routes)
        random.shuffle(p2_routes_copy)
        num_routes_from_p2 = random.randint(1, max(1, len(p2_routes_copy) // 2 if len(p2_routes_copy) > 1 else 1))
        temp_o2_routes = []
        o2_served_customers = set()
        for i in range(min(num_routes_from_p2, self.problem_instance.num_vehicles)):
            if p2_routes_copy:
                route = p2_routes_copy.pop(0)
                temp_o2_routes.append(route)
                for cust_id in route:
                    o2_served_customers.add(cust_id)
        random.shuffle(p1_routes_copy)
        for p1_route in p1_routes_copy:
            if len(temp_o2_routes) >= self.problem_instance.num_vehicles: break
            route_to_add_for_o2 = []
            can_add_this_p1_route = False
            for cust_id in p1_route:
                if cust_id not in o2_served_customers:
                    route_to_add_for_o2.append(cust_id)
                    can_add_this_p1_route = True
            if route_to_add_for_o2 and can_add_this_p1_route:
                temp_o2_routes.append(route_to_add_for_o2)
                for cust_id in route_to_add_for_o2:
                    o2_served_customers.add(cust_id)
        offspring2_sol.routes = temp_o2_routes
        self._repair_solution_ga(offspring2_sol, all_customer_ids)
        return offspring1_sol, offspring2_sol

    def _mutation_ga(self, solution: Solution) -> Solution:
        """Performs mutation on a solution. Can reuse SA-like operators and now includes 2-opt."""
        mutated_solution = copy.deepcopy(solution)
        op_choice = random.random()
        customer_locations = self._get_all_customer_locations(mutated_solution)
        if op_choice < 0.33:
            if customer_locations:
                cust_id_to_relocate, original_route_idx, original_pos_idx = random.choice(customer_locations)
                mutated_solution.routes[original_route_idx].pop(original_pos_idx)
                num_routes = len(mutated_solution.routes)
                if num_routes > 0:
                    target_route_idx = random.randrange(num_routes)
                    target_route = mutated_solution.routes[target_route_idx]
                    max_pos_in_target = len(target_route)
                    target_pos_idx = random.randint(0, max_pos_in_target)
                    target_route.insert(target_pos_idx, cust_id_to_relocate)
        elif op_choice < 0.66:
            if len(customer_locations) >= 2:
                loc1_idx, loc2_idx = random.sample(range(len(customer_locations)), 2)
                cust1_id, route1_idx, pos1_idx = customer_locations[loc1_idx]
                cust2_id, route2_idx, pos2_idx = customer_locations[loc2_idx]
                mutated_solution.routes[route1_idx][pos1_idx] = cust2_id
                mutated_solution.routes[route2_idx][pos2_idx] = cust1_id
        else:
            candidate_routes = [route for route in mutated_solution.routes if len(route) >= 4]
            if candidate_routes:
                route_idx = random.choice([i for i, route in enumerate(mutated_solution.routes) if len(route) >= 4])
                route = mutated_solution.routes[route_idx]
                n = len(route)
                i = random.randint(0, n - 3)
                j = random.randint(i + 2, n - 1)
                route[i:j+1] = reversed(route[i:j+1])
        mutated_solution.calculate_cost_and_feasibility()
        return mutated_solution

    def solve(self) -> Solution | None:
        super().solve()
        print(f"Genetic Algorithm started for {self.problem_instance.name} with time limit {self.time_limit_seconds}s.")
        print(f"Params: Pop Size={self.population_size}, Crossover Rate={self.crossover_rate}, Mutation Rate={self.mutation_rate}, Generations={self.num_generations}")
        population = self._initialize_population_ga()
        if not population:
            print("GA: Failed to initialize population.")
            return None
        fitnesses = [self._calculate_fitness(sol) for sol in population]
        best_fitness_overall = float('inf')
        for i, sol in enumerate(population):
            if fitnesses[i] < best_fitness_overall:
                best_fitness_overall = fitnesses[i]
                if sol.is_feasible:
                    self.best_solution = copy.deepcopy(sol)
                elif self.best_solution is None or not self.best_solution.is_feasible:
                     self.best_solution = copy.deepcopy(sol)
            elif fitnesses[i] == best_fitness_overall and sol.is_feasible and (self.best_solution is None or not self.best_solution.is_feasible):
                 self.best_solution = copy.deepcopy(sol)
        if self.best_solution and self.best_solution.total_distance != float('inf'):
             self.convergence_data.append((0, self.best_solution.total_distance))
        else:
             if population:
                min_cost_sol = min(population, key=lambda s: s.total_distance)
                self.best_solution = copy.deepcopy(min_cost_sol)
                if min_cost_sol.total_distance != float('inf'):
                     self.convergence_data.append((0, min_cost_sol.total_distance))
        print(f"GA Initial Best Solution: Cost={(self.best_solution.total_distance if self.best_solution else float('inf')):.2f}, Feasible={(self.best_solution.is_feasible if self.best_solution else False)}")
        for generation in range(self.num_generations):
            if self._is_time_limit_reached():
                print(f"GA: Time limit reached at generation {generation}.")
                break
            parents = self._selection_ga(population, fitnesses)
            next_population = []
            if self.elitism_count > 0:
                sorted_population = sorted(zip(population, fitnesses), key=lambda x: x[1])
                for i in range(min(self.elitism_count, len(sorted_population))):
                    next_population.append(copy.deepcopy(sorted_population[i][0]))
            num_offspring_needed = self.population_size - len(next_population)
            offspring_generated_count = 0
            parent_indices = list(range(len(parents)))
            random.shuffle(parent_indices)
            idx = 0
            while offspring_generated_count < num_offspring_needed and len(parents) >=2:
                parent1 = parents[parent_indices[idx % len(parents)]]
                parent2 = parents[parent_indices[(idx + 1) % len(parents)]]
                idx += 2
                if random.random() < self.crossover_rate:
                    offspring1, offspring2 = self._crossover_ga(parent1, parent2)
                else:
                    offspring1, offspring2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
                if random.random() < self.mutation_rate:
                    offspring1 = self._mutation_ga(offspring1)
                if random.random() < self.mutation_rate:
                    offspring2 = self._mutation_ga(offspring2)
                if offspring_generated_count < num_offspring_needed:
                    next_population.append(offspring1)
                    offspring_generated_count += 1
                if offspring_generated_count < num_offspring_needed:
                    next_population.append(offspring2)
                    offspring_generated_count += 1
            while len(next_population) < self.population_size:
                if population:
                     next_population.append(copy.deepcopy(random.choice(population)))
                else:
                     next_population.append(self._initialize_solution())
            population = next_population[:self.population_size]
            fitnesses = [self._calculate_fitness(sol) for sol in population]
            for i, sol in enumerate(population):
                if fitnesses[i] < best_fitness_overall:
                    best_fitness_overall = fitnesses[i]
                    self.best_solution = copy.deepcopy(sol)
                elif fitnesses[i] == best_fitness_overall and sol.is_feasible and (self.best_solution is None or not self.best_solution.is_feasible):
                    self.best_solution = copy.deepcopy(sol)
            if generation % 10 == 0:
                if self.best_solution:
                    time_elapsed = time.time() - self.start_time
                    self.convergence_data.append((time_elapsed, self.best_solution.total_distance))
        if self.best_solution:
            time_elapsed = time.time() - self.start_time
            self.convergence_data.append((time_elapsed, self.best_solution.total_distance))
        self._report_solution(self.best_solution, "GeneticAlgorithm")
        self.save_solution_to_txt(self.best_solution, "GA")
        self.plot_convergence_graph("GA")
        print(f"Genetic Algorithm finished. Generations: {generation+1}")
        return self.best_solution

if __name__ == '__main__':
    from data_structures import Customer 
    depot_main = Customer(id=0, x=0, y=0, demand=0, ready_time=0, due_date=1000, service_time=0)
    cust1_main = Customer(id=1, x=10, y=10, demand=5, ready_time=0, due_date=100, service_time=10)
    cust2_main = Customer(id=2, x=20, y=0, demand=8, ready_time=0, due_date=100, service_time=10)
    cust3_main = Customer(id=3, x=0, y=20, demand=12, ready_time=0, due_date=100, service_time=10)
    
    customers_list_main = [depot_main, cust1_main, cust2_main, cust3_main]
    
    dummy_instance_main = ProblemInstance(
        name="TestInstance_Main",
        customers=customers_list_main,
        num_vehicles=2,
        vehicle_capacity=20 
    )
    dummy_instance_main.calculate_distance_matrix()

    print(f"Created dummy instance: {dummy_instance_main.name} with {len(dummy_instance_main.customers)} customers.")
    print(f"Depot ID: {dummy_instance_main.depot.id}")
    print(f"Vehicle Capacity: {dummy_instance_main.vehicle_capacity}")

    print("\n--- Testing Greedy Initial Solution Generation Directly ---")
    class TempSolver(Solver): 
        def solve(self):
            self.start_time = time.time()
            return self._generate_greedy_initial_solution()
    
    temp_solver = TempSolver(dummy_instance_main, 60)
    initial_greedy_solution = temp_solver.solve() 
    print(f"Generated Greedy Solution: {initial_greedy_solution}")
    if initial_greedy_solution and initial_greedy_solution.routes:
        for i, r_info in enumerate(initial_greedy_solution.route_details):
            print(f"  Route {i+1} (IDs: {r_info['customer_ids']}): Dist={r_info['route_distance']:.2f}, Load={r_info['route_load']}, Feasible={not (r_info['time_violation'] or r_info['capacity_violation'])}")
            if r_info.get('time_violation'): print(f"    Time Violation in route {i+1}")
            if r_info.get('capacity_violation'): print(f"    Capacity Violation in route {i+1}")
    elif not initial_greedy_solution.routes:
        print("Greedy solution resulted in no routes.")

    time_limit_main = 60
    print("\n--- Testing Solvers with Output & Plotting ---   [Time Limit: {time_limit_for_test}s per solver]")

    sa_solver_main = SimulatedAnnealingSolver(dummy_instance_main, time_limit_main)
    sa_solution_main = sa_solver_main.solve()

    ts_solver_main = TabuSearchSolver(dummy_instance_main, time_limit_main)
    ts_solution_main = ts_solver_main.solve()

    ga_solver_main = GeneticAlgorithmSolver(dummy_instance_main, time_limit_main)
    ga_solution_main = ga_solver_main.solve()

    print("\nCheck the 'solutions' and 'graphs' directories for output files.") 