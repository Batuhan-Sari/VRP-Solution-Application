import re
from data_structures import Customer, ProblemInstance

def parse_vehicle_info(lines):
    """Parses vehicle information (number and capacity) from instance file lines."""
    vehicle_info = {}
    try:
        vehicle_header_index = -1
        for i, line in enumerate(lines):
            if "VEHICLE" in line:
                vehicle_header_index = i
                break
        
        if vehicle_header_index == -1 or vehicle_header_index + 2 >= len(lines):
            return None

        if "NUMBER" not in lines[vehicle_header_index + 1] or "CAPACITY" not in lines[vehicle_header_index + 1]:
            return None
            
        values_line = lines[vehicle_header_index + 2].strip()
        
        numbers = re.findall(r'\d+', values_line)
        
        if len(numbers) >= 2:
            vehicle_info['number'] = int(numbers[0])
            vehicle_info['capacity'] = int(numbers[1])
        else:
            return None
            
    except Exception as e:
        return None
    return vehicle_info

def parse_customer_data_to_objects(lines):
    """Parses customer data from instance file lines into Customer objects."""
    customers_obj_list = []
    try:
        customer_header_index = -1
        for i, line in enumerate(lines):
            if "CUSTOMER" in line:
                customer_header_index = i
                break

        if customer_header_index == -1 or customer_header_index + 2 >= len(lines):
            return []

        data_start_index = customer_header_index + 2
        
        for i in range(data_start_index, len(lines)):
            line = lines[i].strip()
            if not line: 
                continue
            
            parts = re.split(r'\s+', line) 
            if len(parts) < 7:
                continue
            
            try:
                cust_no = int(parts[0])
                xcoord = float(parts[1])
                ycoord = float(parts[2])
                demand = float(parts[3])
                ready_time = float(parts[4])
                due_date = float(parts[5])
                service_time = float(parts[6])
                
                customers_obj_list.append(Customer(
                    id=cust_no,
                    x=xcoord,
                    y=ycoord,
                    demand=demand,
                    ready_time=ready_time,
                    due_date=due_date,
                    service_time=service_time
                ))
            except ValueError as ve:
                continue
                
    except Exception as e:
        return []
    return customers_obj_list


def read_and_create_instance(file_path: str) -> ProblemInstance | None:
    """
    Reads a VRPTW instance file, parses its content, creates Customer objects,
    a ProblemInstance object, calculates the distance matrix, and returns the instance.
    
    Args:
        file_path (str): The path to the instance file.
        
    Returns:
        ProblemInstance | None: The ProblemInstance object with calculated distance matrix,
                                or None if parsing or instance creation fails.
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        return None
    except Exception as e:
        return None

    if not lines:
        return None

    instance_name = lines[0].strip()

    vehicle_info = parse_vehicle_info(lines)
    if vehicle_info is None:
        num_vehicles = 0
        vehicle_capacity = 0
    else:
        num_vehicles = vehicle_info.get('number', 0)
        vehicle_capacity = vehicle_info.get('capacity', 0)
        
    customer_objects = parse_customer_data_to_objects(lines)
    if not customer_objects:
        return None
    
    try:
        problem_instance = ProblemInstance(
            name=instance_name,
            customers=customer_objects,
            num_vehicles=num_vehicles,
            vehicle_capacity=vehicle_capacity
        )
        problem_instance.calculate_distance_matrix()
    except ValueError as ve:
        return None
    except Exception as e:
        return None
        
    return problem_instance

if __name__ == '__main__':
    test_file_path_c201 = 'instances/c201.txt' 
    
    
    print(f"--- Parsing {test_file_path_c201} ---")
    instance_c201 = read_and_create_instance(test_file_path_c201)
    
    if instance_c201:
        print(f"Successfully created instance: {instance_c201.name}")
        print(f"Number of vehicles: {instance_c201.num_vehicles}, Capacity: {instance_c201.vehicle_capacity}")
        print(f"Depot: {instance_c201.depot}")
        print(f"Number of customers (incl. depot): {len(instance_c201.customers)}")
        if instance_c201.distance_matrix:
            print(f"Distance matrix calculated. Shape: {len(instance_c201.distance_matrix)}x{len(instance_c201.distance_matrix[0]) if instance_c201.distance_matrix else 0}")
        else:
            print("Distance matrix not calculated.")
    else:
        print(f"Failed to create instance from file: {test_file_path_c201}")

    print("\n--- Parsing instances/r201.txt ---")
    test_file_path_r201 = 'instances/r201.txt'
    instance_r201 = read_and_create_instance(test_file_path_r201)
    if instance_r201:
        print(f"Successfully created instance: {instance_r201.name}")
        print(f"Number of customers (incl. depot): {len(instance_r201.customers)}")
        if instance_r201.distance_matrix and instance_r201.customers:
             print(f"Distance from depot to customer 1 ({instance_r201.customers[1].id}): {instance_r201.distance_matrix[instance_r201.depot.id][instance_r201.customers[1].id]:.2f}") 
    else:
        print(f"Failed to create instance from file: {test_file_path_r201}") 