# Import all necessary libraries
import pandas as pd
import numpy as np
import copy
import random
import logging
import time
import matplotlib.pyplot as plt

# Set logging                   
logger = logging.getLogger(name='paintshop')
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(message)s',
                    handlers=[logging.FileHandler("paintshop.log")])

# Set needed values
VERYBIGNUMBER = 424242424242
random.seed(70)
#start_time = time.time()

# Import excel data
logger.info("Importing data from Excel files...")
orders_df = pd.read_excel('paintshop_september_2024.xlsx', sheet_name='Orders')
machines_df = pd.read_excel('paintshop_september_2024.xlsx', sheet_name='Machines')
setups_df = pd.read_excel('paintshop_september_2024.xlsx', sheet_name='Setups')

# Define sets
B = orders_df['order'].tolist()  # Bestellingen
M = machines_df['machine'].tolist()  # Machines
H1 = setups_df['from_colour'].tolist()  # Startkleuren
H2 = setups_df['to_colour'].tolist()  # Eindkleuren

# Starting values
current_time = {machine: 0 for machine in M}  # Tijd per machine
current_color = {machine: None for machine in M}  # Kleur per machine
scheduled_orders = {machine: [] for machine in M}  # Orders per machine
available_orders = B.copy()  # Beschikbare orders

# Functions for calculating data for mathematical model
def processing_time(surface, speed):
    """
    Calculate the processing time based on surface area and machine speed.
    """
    return surface / speed

def setup_time(from_colour, to_colour, setups_df):
    """
    Calculate setup time based on the color transition.
    """
    if from_colour == to_colour or from_colour is None:
        return 0
    else:
        return setups_df[(setups_df['from_colour'] == from_colour) & (setups_df['to_colour'] == to_colour)]['setup_time'].values[0]

def calculate_total_penalty(current_time, deadline, penalty):
    """
    Calculate total penalty based on the delay from the deadline.
    """
    delay = max(0, current_time - deadline)
    return delay * penalty

def calculate_penalty_for_schedule(scheduled_orders):
    """Calculate the total penalty for the entire schedule."""
    total_penalty = 0
    for machine, orders in scheduled_orders.items():
        for order in orders:
            completion_time = order['end_time']
            order_info = orders_df[orders_df['order'] == order['order']].iloc[0]
            penalty = order_info['penalty']
            deadline = order_info['deadline']
            total_penalty += calculate_total_penalty(completion_time, deadline, penalty)
    return total_penalty

def update_schedule(scheduled_orders, current_time, current_colour):
    """Update the machine schedules and recalculate times and setups."""
    for machine, orders in scheduled_orders.items():
        current_time[machine] = 0
        current_colour[machine] = None
        for order in orders:
            order_info = orders_df[orders_df['order'] == order['order']].iloc[0]
            process_time = processing_time(order_info['surface'], 
                                           machines_df[machines_df['machine'] == machine]['speed'].values[0])
            set_time = setup_time(current_colour[machine], order_info['colour'], setups_df)

            start_time = current_time[machine]
            end_time = start_time + process_time + set_time

            order['start_time'] = start_time
            order['end_time'] = end_time
            order['setup_time'] = set_time
            current_time[machine] = end_time
            current_colour[machine] = order_info['colour']

# Function for constructive heuristics            
def greedy_paint_planner():
    """
    Greedy algorithm to assign orders to machines based on speed, setup time, and penalty.

    Returns:
        tuple: Total penalty and a dictionary of scheduled orders for each machine.
    """
    total_penalty = 0
    # Start timing
    #start_time_gpp = time.time()
    
    # Sort machines by speed (fastest first)
    sorted_machines = machines_df.sort_values(by='speed', ascending=False)['machine'].tolist()
      
    # Loop until all orders are assigned
    while available_orders:
        for machine in sorted_machines:
            if not available_orders:
                break  # Stop if all orders are assigned
            
            best_order = None
            min_setup_time = VERYBIGNUMBER
            max_penalty = -VERYBIGNUMBER
            
            # Find the best order for the current machine
            for order in available_orders:
                order_info = orders_df[orders_df['order'] == order].iloc[0]
                surface = order_info['surface']
                colour = order_info['colour']
                deadline = order_info['deadline']
                penalty = order_info['penalty']
                
                # Calculate processing time and setup time
                process_time = processing_time(surface, machines_df[machines_df['machine'] == machine]['speed'].values[0])
                set_time = setup_time(current_color[machine], colour, setups_df)
                completion_time = current_time[machine] + process_time + set_time
                
                # Check if there is a penalty
                current_penalty = calculate_total_penalty(completion_time, deadline, penalty)
                
                # Select the order with the lowest setup time and highest penalty
                if set_time < min_setup_time or (set_time == min_setup_time and current_penalty > max_penalty):
                    best_order = order
                    min_setup_time = set_time
                    max_penalty = current_penalty
            
            # If a best order is found, assign it to the machine
            if best_order is not None:
                order_info = orders_df[orders_df['order'] == best_order].iloc[0]
                process_time = processing_time(order_info['surface'], machines_df[machines_df['machine'] == machine]['speed'].values[0])
                set_time = setup_time(current_color[machine], order_info['colour'], setups_df)
                
                # Update machine status
                start_time = current_time[machine]
                end_time = current_time[machine] + process_time + set_time
                scheduled_orders[machine].append({
                    'order': best_order,
                    'start_time': start_time,
                    'end_time': end_time,
                    'setup_time': set_time,
                    'colour': order_info['colour'],
                })
                current_time[machine] = end_time
                current_color[machine] = order_info['colour']
                
                # Remove the assigned order from the list of available orders
                available_orders.remove(best_order)
                total_penalty += max_penalty  # Add penalty
                          
    # Log elapsed time for greedy algorithm
    #end_time_gpp = time.time()
    #logger.info(msg=f'Elapsed time Constructive Heuristics (gpp): {end_time_gpp - start_time_gpp:.6f}')
    return total_penalty, scheduled_orders

# Function for discrete improving search
def two_exchange():
    """
    Discrete improving search algorithm for optimising the schedule.

    Returns:
        int: Total penalty after optimisation.
        dict: Optimised schedule with orders per machine.
    """
    # Start timing
    #start_time_te = time.time()
    
    # Start with the feasible solution from the greedy planner
    total_penalty, scheduled_orders = greedy_paint_planner()

    # Variables to keep track of the current status
    current_time = {machine: 0 for machine in M}
    current_colour = {machine: None for machine in M}
    
    # Helper function to update the schedule after an exchange
    def update_schedule(scheduled_orders, current_time, current_colour):
        for machine, orders in scheduled_orders.items():
            current_time[machine] = 0
            current_colour[machine] = None
            for order in orders:
                order_info = orders_df[orders_df['order'] == order['order']].iloc[0]
                process_time = processing_time(order_info['surface'], machines_df[machines_df['machine'] == machine]['speed'].values[0])
                set_time = setup_time(current_colour[machine], order_info['colour'], setups_df)

                start_time = current_time[machine]
                end_time = start_time + process_time + set_time

                order['start_time'] = start_time
                order['end_time'] = end_time
                order['setup_time'] = set_time
                current_time[machine] = end_time
                current_colour[machine] = order_info['colour']

    # Initially update the schedule
    update_schedule(scheduled_orders, current_time, current_colour)

    improved = True
    penalty_history = []  # List to store the penalties at each iteration

    while improved:
        improved = False
        best_penalty = calculate_penalty_for_schedule(scheduled_orders)

        # Append the current penalty to the history list
        penalty_history.append(best_penalty)

        # Try swapping each pair of orders on different machines
        for machine1 in scheduled_orders:
            for order1 in scheduled_orders[machine1]:
                for machine2 in scheduled_orders:
                    if machine1 == machine2:
                        continue
                    for order2 in scheduled_orders[machine2]:
                        # Make a temporary copy of the schedule
                        temp_scheduled_orders = copy.deepcopy(scheduled_orders)

                        # Exchange the orders directly here
                        idx1 = next(i for i, o in enumerate(temp_scheduled_orders[machine1]) if o['order'] == order1['order'])
                        idx2 = next(i for i, o in enumerate(temp_scheduled_orders[machine2]) if o['order'] == order2['order'])
                        
                        temp_scheduled_orders[machine1][idx1], temp_scheduled_orders[machine2][idx2] = (
                            temp_scheduled_orders[machine2][idx2],
                            temp_scheduled_orders[machine1][idx1]
                        )

                        # Update the schedule after the exchange
                        temp_time = copy.deepcopy(current_time)
                        temp_colour = copy.deepcopy(current_colour)
                        update_schedule(temp_scheduled_orders, temp_time, temp_colour)

                        # Calculate the total penalty for the new schedule
                        temp_penalty = calculate_penalty_for_schedule(temp_scheduled_orders)

                        # If the penalty has improved, accept the exchange
                        if temp_penalty < best_penalty:
                            scheduled_orders = temp_scheduled_orders
                            best_penalty = temp_penalty
                            improved = True
                            break  # Return to the start if an improvement is found
                    if improved:
                        break
            if improved:
                break

        # If no exchange leads to improvement, stop the algorithm
        if not improved:
            logger.info(msg="No further improvements possible.")
            break
      
    # After optimisation: return the total penalty and the schedule
    total_penalty = calculate_penalty_for_schedule(scheduled_orders)
    logger.debug(msg=f"Total penalty: {total_penalty}")

    # Append final penalty to history list (optional, depending on whether it is included already)
    penalty_history.append(total_penalty)

    # Plot the total penalty against each iteration
    plt.figure(figsize=(10, 6))
    plt.plot(penalty_history, marker='o', linestyle='-', color='b')
    plt.title('Total Penalty per Iteration using 2-exchange')
    plt.xlabel('Iteration')
    plt.ylabel('Total Penalty')
    plt.grid(True)
    plt.show()

    # Log elapsed time for 2 exchange
    #end_time_te = time.time()
    #logger.info(msg=f'Elapsed time Discrete Improving Search (te): {end_time_te - start_time_te:.6f}')
    
    return total_penalty, scheduled_orders

#Function for meta-heuristics
def simulated_annealing(max_iterations, initial_temp, cooling_rate):
    
    # Start timing
    #start_time_sa = time.time()
    
    # Start with the feasible solution from the greedy planner
    total_penalty, scheduled_orders = greedy_paint_planner()

    current_solution = copy.deepcopy(scheduled_orders)
    current_penalty = calculate_penalty_for_schedule(current_solution)

    temperature = initial_temp
    best_solution = copy.deepcopy(current_solution)
    best_penalty = current_penalty

    penalty_history = [current_penalty]

    for iteration in range(max_iterations):
        new_solution = copy.deepcopy(current_solution)
        machine1, machine2 = random.sample(M, 2)

        if new_solution[machine1] and new_solution[machine2]:
            order1_idx = random.randint(0, len(new_solution[machine1]) - 1)
            order2_idx = random.randint(0, len(new_solution[machine2]) - 1)

            # Swap two orders between different machines
            new_solution[machine1][order1_idx], new_solution[machine2][order2_idx] = (
                new_solution[machine2][order2_idx],
                new_solution[machine1][order1_idx]
            )

            # Re-initialize current_time and current_colour after the swap
            temp_time = {machine: 0 for machine in M}
            temp_colour = {machine: None for machine in M}

            # Update schedule with the new swapped solution
            update_schedule(new_solution, temp_time, temp_colour)
            new_penalty = calculate_penalty_for_schedule(new_solution)

            if new_penalty < current_penalty:
                current_solution = new_solution
                current_penalty = new_penalty
                if new_penalty < best_penalty:
                    best_solution = copy.deepcopy(new_solution)
                    best_penalty = new_penalty
            else:
                # Calculate acceptance probability based on temperature
                penalty_diff = (current_penalty - new_penalty)
                acceptance_probability = np.exp(penalty_diff / max(1, temperature))

                if random.random() < acceptance_probability:
                    current_solution = new_solution
                    current_penalty = new_penalty

        # Update temperature using cooling schedule
        temperature = initial_temp / (1 + cooling_rate * iteration)
        penalty_history.append(current_penalty)

        if iteration % 100 == 0:
            logger.debug(msg=f"Iteration {iteration}, Best Penalty: {best_penalty}")

        if temperature < 1e-8:
            break

    # Plot the penalty history
    plt.plot(penalty_history, marker='o', color='b')
    plt.title('Total Penalty per Iteration using Simulated Annealing')
    plt.xlabel('Iteration')
    plt.ylabel('Total Penalty')
    plt.grid(True)
    plt.show()
    
    # Log elapsed time for simulated annealing
    #end_time_sa = time.time()
    #logger.info(msg=f'Elapsed time Meta Heuristics (sa): {end_time_sa - start_time_sa:.6f}')

    return best_penalty, best_solution

# Function for plotting a Gantt chart of the resulting schedules
def plot_schedule(scheduled_orders, method, orders_df):
    """Plots a Gantt chart of the scheduled orders and marks late orders.

    Args:
        scheduled_orders (dict): Every order, their start time, end time, on which machine, and setup time.
        method (str): Method used to calculate the schedule.
        orders_df (pd.DataFrame): DataFrame with order details like deadlines and penalties.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = 0
    
    # Colours for visualisation
    color_map = {
        'Green': 'green',
        'Yellow': 'yellow',
        'Blue': 'blue',
        'Red': 'red'
    }

    for machine, orders in scheduled_orders.items():
        y_pos += 1  # For each machine
        for order in orders:
            order_info = orders_df[orders_df['order'] == order['order']].iloc[0]
            order_color = order['colour']
            processing_time = order['end_time'] - order['start_time'] - order['setup_time']
            setup_time = order['setup_time']
            start_time = order['start_time']
            end_time = order['end_time']
            deadline = order_info['deadline']
            
            # Check if the order is late
            is_late = end_time > deadline

            # Draw processing time
            bar_color = color_map[order_color] 
            ax.barh(y_pos, processing_time, left=start_time + setup_time, color=bar_color, edgecolor='black')
            
            # Add order number, mark it with '*' if late
            label = f"Order {order['order']}" + (' * ' if is_late else '')
            ax.text(start_time + setup_time + processing_time / 2, y_pos, label, ha='center', va='center', color='black', rotation=90, weight='bold')

            # Draw setup time
            if setup_time > 0:
                ax.barh(y_pos, setup_time, left=start_time, color='gray', edgecolor='black', hatch='//')
    
    ax.set_yticks(range(1, len(scheduled_orders) + 1))
    ax.set_yticklabels([f"Machine {m}" for m in scheduled_orders.keys()])
    ax.set_xlabel('Time')
    ax.set_ylabel('Machines')
    ax.set_title(f'Gantt Chart for Paint Shop Scheduling using {method}')
    plt.show()
    
def main():
    # Call greedy_paint_planner to get the initial schedule and plot it
    logger.info(msg="Generating schedule using Greedy Paint Planner...")
    start_time_gpp = time.time()
    greedy_penalty, greedy_schedule = greedy_paint_planner()
    logger.info(msg=f"Greedy total penalty: {greedy_penalty:.2f}")
    end_time_gpp = time.time()
    logger.info(msg=f'Elapsed time Constructive Heuristics (gpp): {end_time_gpp - start_time_gpp:.6f}')
    plot_schedule(greedy_schedule, 'Constructive Heuristics', orders_df)
    
    # Call two_exchange to optimize the schedule and plot it
    logger.info(msg="Improving schedule using 2-Exchange...")
    start_time_te = time.time()
    two_exchange_penalty, two_exchange_schedule = two_exchange()
    logger.info(msg=f"2-Exchange total penalty: {two_exchange_penalty:2f}")
    end_time_te = time.time()
    logger.info(msg=f'Elapsed time Discrete Improving Search (te): {end_time_te - start_time_te:.6f}')
    plot_schedule(two_exchange_schedule, '2-Exchange', orders_df)
    
    # Call simulated_annealing to further optimize and plot it
    logger.info(msg="Improving schedule using Simulated Annealing...")
    start_time_sa = time.time()
    max_iterations = 5000  # Set the number of iterations for Simulated Annealing
    initial_temp = 100     # Set the initial temperature
    cooling_rate = 0.98     # Set the cooling rate
    sa_penalty, sa_schedule = simulated_annealing(max_iterations, initial_temp, cooling_rate)
    logger.info(msg=f"Simulated Annealing total penalty: {sa_penalty:.2f}")
    end_time_sa = time.time()
    logger.info(msg=f'Elapsed time Meta Heuristics (sa): {end_time_sa - start_time_sa:.6f}')
    plot_schedule(sa_schedule, 'Simulated Annealing', orders_df)

if __name__ == "__main__":
    main()