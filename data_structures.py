from dataclasses import dataclass, field
import math

@dataclass
class Customer:
    id: int
    x: float
    y: float
    demand: float
    ready_time: float
    due_date: float
    service_time: float
    is_depot: bool = False

    def __post_init__(self):
        if self.id == 0:
            self.is_depot = True

@dataclass
class ProblemInstance:
    name: str
    customers: list[Customer]
    num_vehicles: int
    vehicle_capacity: float
    depot: Customer = field(init=False)
    distance_matrix: list[list[float]] = field(default_factory=list, init=False)

    def __post_init__(self):
        depot_customer = None
        for customer in self.customers:
            if customer.is_depot or customer.id == 0: 
                depot_customer = customer
                break
        if depot_customer is None and self.customers:
             
            for customer in self.customers:
                if customer.id == 0:
                    customer.is_depot = True 
                    depot_customer = customer
                    break
            if depot_customer is None and self.customers:
                 raise ValueError("Depot (customer 0) not found in customer list.")
        elif not self.customers:
            raise ValueError("Customer list is empty, cannot initialize ProblemInstance.")
        self.depot = depot_customer

    def calculate_distance_matrix(self):
        """Calculates the Euclidean distance matrix between all customers."""
        num_customers = len(self.customers)
        self.distance_matrix = [[0.0] * num_customers for _ in range(num_customers)]
        for i in range(num_customers):
            for j in range(num_customers):
                if i == j:
                    self.distance_matrix[i][j] = 0.0
                else:
                    cust1 = self.customers[i]
                    cust2 = self.customers[j]
                    dist = math.sqrt((cust1.x - cust2.x)**2 + (cust1.y - cust2.y)**2)
                    self.distance_matrix[i][j] = dist

    def get_customer_by_id(self, customer_id: int) -> Customer | None:
        """Returns a customer object by its ID."""
        for customer in self.customers:
            if customer.id == customer_id:
                return customer
        return None

@dataclass
class Solution:
    """
    Represents a solution to the VRPTW problem.
    """
    problem_instance: ProblemInstance
    routes: list[list[int]] = field(default_factory=list)
    total_distance: float = float('inf')
    is_feasible: bool = False
    route_details: list[dict] = field(default_factory=list)

    def __str__(self):
        return (f"Solution(Total Distance: {self.total_distance:.2f}, Feasible: {self.is_feasible}, "
                f"Num Routes: {len(self.routes)})")

    def calculate_cost_and_feasibility(self, update_self: bool = True):
        """Calculates the total distance and checks feasibility of all routes in the solution."""
        if not self.problem_instance or not hasattr(self.problem_instance, 'depot') or not self.problem_instance.depot:
            if update_self:
                self.total_distance = float('inf')
                self.is_feasible = False
                self.route_details = []
            return float('inf'), False, []
        
        depot = self.problem_instance.depot
        dist_matrix = self.problem_instance.distance_matrix

        if not dist_matrix:
            if update_self:
                self.total_distance = float('inf')
                self.is_feasible = False
                self.route_details = []
            return float('inf'), False, []

        if not self.routes:

            has_only_depot_customers = all(c.is_depot or c.demand == 0 for c in self.problem_instance.customers)
            if not has_only_depot_customers:
                pass 
            else:
                if update_self:
                    self.total_distance = 0
                    self.is_feasible = True
                    self.route_details = []
                return 0, True, []

        current_total_distance = 0.0
        solution_is_feasible = True
        all_route_details_list = []

        served_customer_ids = set()
        for route_cust_ids in self.routes:
            for cust_id in route_cust_ids:
                served_customer_ids.add(cust_id)

        all_problem_customer_ids = {c.id for c in self.problem_instance.customers if not c.is_depot}
        
        if not all_problem_customer_ids.issubset(served_customer_ids):
            solution_is_feasible = False 

        for route_customer_ids in self.routes:
            route_info = {
                'customer_ids': list(route_customer_ids), 
                'stops_details': [], 
                'route_distance': 0.0, 
                'route_load': 0.0, 
                'time_violation': False, 
                'capacity_violation': False
            }

            if not route_customer_ids: 
                all_route_details_list.append(route_info) 
                continue

            route_dist = 0.0
            current_load = 0.0
            current_time = depot.ready_time
            route_feasible_current = True
            last_customer_obj = depot
            
            for customer_id in route_customer_ids:
                customer = self.problem_instance.get_customer_by_id(customer_id)
                if not customer or customer.is_depot:
                    route_feasible_current = False; break 
                
                if not (0 <= last_customer_obj.id < len(dist_matrix) and 
                        0 <= customer.id < len(dist_matrix[last_customer_obj.id])):
                    route_feasible_current = False; break

                travel_dist = dist_matrix[last_customer_obj.id][customer.id]
                route_dist += travel_dist
                current_time += travel_dist
                arrival_time = current_time
                service_start_time = max(arrival_time, customer.ready_time)
                
                if service_start_time > customer.due_date:
                    route_feasible_current = False
                    route_info['time_violation'] = True
                
                current_time = service_start_time + customer.service_time
                departure_time = current_time

                if departure_time > customer.due_date:
                    route_feasible_current = False
                    route_info['time_violation'] = True
                
                current_load += customer.demand
                if current_load > self.problem_instance.vehicle_capacity:
                    route_feasible_current = False
                    route_info['capacity_violation'] = True
                
                route_info['stops_details'].append({
                    'customer_id': customer.id,
                    'arrival': arrival_time,
                    'service_start': service_start_time,
                    'departure': departure_time,
                    'current_load_after_service': current_load
                })
                if not route_feasible_current: break
                last_customer_obj = customer
            
            if route_feasible_current:
                if not (0 <= last_customer_obj.id < len(dist_matrix) and 
                        0 <= depot.id < len(dist_matrix[last_customer_obj.id])):
                    route_feasible_current = False
                else:
                    travel_to_depot_dist = dist_matrix[last_customer_obj.id][depot.id]
                    route_dist += travel_to_depot_dist
                    current_time += travel_to_depot_dist
                    if current_time > depot.due_date:
                        route_feasible_current = False
                        route_info['time_violation'] = True
            
            route_info['route_distance'] = route_dist
            route_info['route_load'] = current_load
            all_route_details_list.append(route_info)

            if route_feasible_current:
                current_total_distance += route_dist
            else:
                solution_is_feasible = False
                current_total_distance += route_dist 

        
        PENALTY_VALUE = 100000  

        if not solution_is_feasible:
            current_total_distance += PENALTY_VALUE
            
        if update_self:
            self.total_distance = current_total_distance 
            self.is_feasible = solution_is_feasible
            self.route_details = all_route_details_list

        return current_total_distance, solution_is_feasible, all_route_details_list

if __name__ == '__main__':
    depot_data = {'id': 0, 'x': 40, 'y': 50, 'demand': 0, 'ready_time': 0, 'due_date': 1000, 'service_time': 0}
    cust1_data = {'id': 1, 'x': 52, 'y': 75, 'demand': 10, 'ready_time': 100, 'due_date': 200, 'service_time': 90}
    cust2_data = {'id': 2, 'x': 45, 'y': 70, 'demand': 30, 'ready_time': 50, 'due_date': 150, 'service_time': 90}

    c0 = Customer(**depot_data)
    c1 = Customer(**cust1_data)
    c2 = Customer(**cust2_data)

    print(f"Depot: {c0}")
    print(f"Customer 1: {c1}")

    all_customers = [c0, c1, c2]
    
    try:
        instance = ProblemInstance(
            name="TestInstance",
            customers=all_customers,
            num_vehicles=2,
            vehicle_capacity=100
        )
        print(f"\nProblem Instance: {instance.name}")
        print(f"Depot in instance: {instance.depot}")
        print(f"Num vehicles: {instance.num_vehicles}, Capacity: {instance.vehicle_capacity}")

        instance.calculate_distance_matrix()
        print("\nDistance Matrix:")
        for row in instance.distance_matrix:
            print([round(d, 2) for d in row])
        
        retrieved_cust1 = instance.get_customer_by_id(1)
        print(f"\nRetrieved Customer 1: {retrieved_cust1}")

    except ValueError as e:
        print(f"Error creating instance: {e}")

    depot_s = Customer(id=0, x=0, y=0, demand=0, ready_time=0, due_date=1000, service_time=0)
    c1_s = Customer(id=1, x=10, y=0, demand=10, ready_time=0, due_date=100, service_time=10)
    c2_s = Customer(id=2, x=0, y=10, demand=5, ready_time=0, due_date=100, service_time=5)

    instance_s = ProblemInstance(
        name="SolTest",
        customers=[depot_s, c1_s, c2_s],
        num_vehicles=1,
        vehicle_capacity=20
    )
    instance_s.calculate_distance_matrix()

    sol = Solution(problem_instance=instance_s)
    sol.routes = [[1, 2]] 
    
    dist, feasible, details = sol.calculate_cost_and_feasibility()
    print(f"\nSolution Test: Distance={dist:.2f}, Feasible={feasible}")
    print(f"Solution Object: {sol}")
    if details:
        for i, detail in enumerate(details):
            print(f"  Route {i+1} details: dist={detail['route_distance']:.2f}, load={detail['route_load']}, time_viol={detail['time_violation']}, cap_viol={detail['capacity_violation']}")

    sol_only_depot_route = Solution(problem_instance=instance_s)
    sol_only_depot_route.routes = [[]] 
    dist_od, feasible_od, _ = sol_only_depot_route.calculate_cost_and_feasibility()
    print(f"\nSolution Test (empty route): Distance={dist_od:.2f}, Feasible={feasible_od}, Obj: {sol_only_depot_route}")
    
    sol_no_routes = Solution(problem_instance=instance_s)
    sol_no_routes.routes = []
    dist_nr, feasible_nr, _ = sol_no_routes.calculate_cost_and_feasibility()
    print(f"\nSolution Test (no routes): Distance={dist_nr:.2f}, Feasible={feasible_nr}, Obj: {sol_no_routes}") 