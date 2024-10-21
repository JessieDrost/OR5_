# Import all necessary libraries
import pandas as pd
import numpy as np
import copy
import random
import logging
import time
import matplotlib.pyplot as plt

VERYBIGNUMBER = 424242424242

# Set logging                   
logger = logging.getLogger(name='paintshop')
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(message)s',
                    handlers=[logging.FileHandler("paintshop.log")])

# We want to be able to generate random instances
random.seed(70)

# File paths of the databases (comment/uncomment as needed)
file_path = 'paintshop_september_2024.xlsx'
# file_path = 'paintshop_november_2024.xlsx'

# Import excel data
logger.info("Importing data from Excel files...")
orders_df = pd.read_excel(file_path, sheet_name='Orders')
machines_df = pd.read_excel(file_path, sheet_name='Machines')
setups_df = pd.read_excel(file_path, sheet_name='Setups')

# Define sets
B = orders_df['order'].tolist()  # Orders
M = machines_df['machine'].tolist()  # Machines

# Define starting values
current_time = {machine: 0 for machine in M}  # Time per machine
current_color = {machine: None for machine in M}  # Colour per machine
scheduled_orders = {machine: [] for machine in M}  # Orders per machine
available_orders = B.copy()  # Available orders

# Functions for calculating data for mathematical model
def processing_time(surface, speed):
    """ Calculate the processing time based on surface area and machine speed. """
    return surface / speed

def setup_time(from_colour, to_colour, setups_df):
    """ Calculate setup time based on the color transition. """
    if from_colour == to_colour or from_colour is None:
        return 0
    return setups_df[(setups_df['from_colour'] == from_colour) & (setups_df['to_colour'] == to_colour)]['setup_time'].values[0]

def calculate_total_penalty(current_time, deadline, penalty):
    """ Calculate total penalty based on the delay from the deadline. """
    delay = max(0, current_time - deadline)
    return delay * penalty

def calculate_penalty_for_schedule(scheduled_orders):
    """ Calculate the total penalty for the entire schedule. """
    total_penalty = 0
    for machine, orders in scheduled_orders.items():
        for order in orders:
            completion_time = order['end_time']
            order_info = orders_df[orders_df['order'] == order['order']].iloc[0]
            total_penalty += calculate_total_penalty(completion_time, order_info['deadline'], order_info['penalty'])
    return total_penalty

def update_schedule(scheduled_orders, current_time, current_colour):
    """ Update the machine schedules and recalculate times and setups. """
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

# Function for constructive heuristics: Greedy Paint Planner
def greedy_paint_planner():
    """ Greedy algorithm to assign orders to machines.
        Selects the order with the lowest setup time and assigns it to the fastest available machine.

    Returns:
        total_penalty (int): The total penalty incurred after scheduling.
        scheduled_orders (dict): A dictionary containing the schedule for each machine.
    """   
    total_penalty = 0  # Initialize total penalty
    sorted_machines = machines_df.sort_values(by='speed', ascending=False)['machine'].tolist()  # Sort machines by speed (fastest machines first)

    # Continue until all orders are assigned
    while available_orders:
        for machine in sorted_machines:  # Iterate over each machine (starting with the fastest)
            if not available_orders:
                break  # Stop if all orders have been assigned

            best_order = None
            min_setup_time = VERYBIGNUMBER
            max_penalty = - VERYBIGNUMBER

            # Find the best order to assign to the current machine
            for order in available_orders:
                order_info = orders_df[orders_df['order'] == order].iloc[0]
                process_time = processing_time(order_info['surface'], machines_df[machines_df['machine'] == machine]['speed'].values[0])
                set_time = setup_time(current_color[machine], order_info['colour'], setups_df)
                completion_time = current_time[machine] + process_time + set_time
                current_penalty = calculate_total_penalty(completion_time, order_info['deadline'], order_info['penalty'])

                if set_time < min_setup_time or (set_time == min_setup_time and current_penalty > max_penalty):
                    best_order = order
                    min_setup_time = set_time
                    max_penalty = current_penalty

            # Choose the order with the lowest setup time, and in case of a tie, select the higher penalty order
            if best_order is not None:
                order_info = orders_df[orders_df['order'] == best_order].iloc[0]
                process_time = processing_time(order_info['surface'], machines_df[machines_df['machine'] == machine]['speed'].values[0])
                set_time = setup_time(current_color[machine], order_info['colour'], setups_df)

                # Schedule the order on the current machine
                start_time = current_time[machine]
                end_time = start_time + process_time + set_time
                scheduled_orders[machine].append({
                    'order': best_order,
                    'start_time': start_time,
                    'end_time': end_time,
                    'setup_time': set_time,
                    'colour': order_info['colour'],
                })
                current_time[machine] = end_time
                current_color[machine] = order_info['colour']
                
                # Remove the order from available orders and add its penalty to the total
                available_orders.remove(best_order)
                total_penalty += max_penalty

    return total_penalty, scheduled_orders

# Function for discrete improving search: 2-exchange algorithm
def two_exchange():
    """ Discrete improving search algorithm for optimizing the schedule.
        Starts with the solution obtained through greedy_paint_planner().
        Evaluates possible 2-exchange moves by swapping two operations to check for improvements.

    Returns:
        total_penalty (int): The total penalty after optimization.
        scheduled_orders (dict): The optimized schedule after the 2-exchange.
    """
    # Start with the feasible solution from the greedy planner
    total_penalty, scheduled_orders = greedy_paint_planner()

    penalty_history = []
    improved = True
    update_schedule(scheduled_orders, current_time, current_color)

    while improved:
        improved = False
        # Calculate penalty for the current schedule and add current penalty to history
        best_penalty = calculate_penalty_for_schedule(scheduled_orders)
        penalty_history.append(best_penalty)

        # Try swapping each pair of orders on different machines
        for machine1 in scheduled_orders:
            for order1 in scheduled_orders[machine1]:
                for machine2 in scheduled_orders:
                    if machine1 == machine2:
                        continue  # Skip if both orders are on the same machine
                    for order2 in scheduled_orders[machine2]:
                        temporary_scheduled_orders = copy.deepcopy(scheduled_orders)

                        # Find indices of the orders to be swapped
                        idx1 = next(i for i, o in enumerate(temporary_scheduled_orders[machine1]) if o['order'] == order1['order'])
                        idx2 = next(i for i, o in enumerate(temporary_scheduled_orders[machine2]) if o['order'] == order2['order'])

                        temporary_scheduled_orders[machine1][idx1], temporary_scheduled_orders[machine2][idx2] = (
                            temporary_scheduled_orders[machine2][idx2],
                            temporary_scheduled_orders[machine1][idx1]
                        )

                        # Update the schedule after swapping and calculate the total penalty for the new schedule
                        update_schedule(temporary_scheduled_orders, current_time, current_color)
                        temporary_penalty = calculate_penalty_for_schedule(temporary_scheduled_orders)

                        # If the penalty improves, accept the exchange
                        if temporary_penalty < best_penalty:
                            scheduled_orders = temporary_scheduled_orders
                            best_penalty = temporary_penalty
                            improved = True
                            break
                    if improved:
                        break
            if improved:
                break

        # Stop if no further improvements are found
        if not improved:
            logger.info("No further improvements possible.")
            break
        
    # Calculate the total penalty for the optimized schedule
    total_penalty = calculate_penalty_for_schedule(scheduled_orders)
    logger.info(f"Total penalty: {total_penalty}")
    penalty_history.append(total_penalty)

    # Plot the total penalty over iterations
    plt.figure(figsize=(10, 6))
    plt.plot(penalty_history, marker='o', linestyle='-', color='b')
    plt.title('Total Penalty per Iteration using 2-exchange')
    plt.xlabel('Iteration')
    plt.ylabel('Total Penalty')
    plt.grid(True)
    plt.show()

    return total_penalty, scheduled_orders

# Function for meta-heuristics: Simulated Annealing
def simulated_annealing(max_iterations, initial_temperature, cooling_rate):
    """Performs simulated annealing to optimize the paint shop schedule.

    Args:
        max_iterations (int): The maximum number of iterations for the annealing process.
        initial_temperature (float): The starting temperature for annealing.
        cooling_rate (float): The rate at which the temperature decreases per iteration.

    Returns:
        tuple: Returns the best penalty score and the best solution (optimized schedule).
    """    
    
    # Start with a feasible solution generated by the greedy planner
    total_penalty, scheduled_orders = greedy_paint_planner()

    # Deep copy of the initial solution to use in the optimization
    current_solution = copy.deepcopy(scheduled_orders)
    current_penalty = calculate_penalty_for_schedule(current_solution)

    # Initialize temperature and track the best solution and its penalty
    temperature = initial_temperature
    best_solution = copy.deepcopy(current_solution)
    best_penalty = current_penalty

    # Lists to track the progress of the optimization for plotting purposes
    penalty_history = [current_penalty]
    best_penalty_history = [best_penalty]
    temperature_history = [temperature]
    
    detonation_threshold = 300  # Threshold for resetting the solution if no improvement occurs
    no_improvement_count = 0  # Counter to track iterations with no improvement
    
    # Begin the main loop for the simulated annealing algorithm
    for iteration in range(max_iterations):
        new_solution = copy.deepcopy(current_solution)
        
        # Neighborhood search: Swap orders either within the same machine or between machines
        if random.random() < 0.5:  # Swap orders within the same machine (local move)
            machine = random.choice(M)  # Select a random machine
            if len(new_solution[machine]) > 1:  # Ensure there are at least 2 orders to swap
                order1_idx, order2_idx = random.sample(range(len(new_solution[machine])), 2)
                # Swap two orders within the selected machine
                new_solution[machine][order1_idx], new_solution[machine][order2_idx] = (
                    new_solution[machine][order2_idx],
                    new_solution[machine][order1_idx]
                )
        else:  # Swap orders between two different machines (larger move)
            machine1, machine2 = random.sample(M, 2)  # Select two random machines
            if new_solution[machine1] and new_solution[machine2]:  # Ensure both have orders to swap
                order1_idx = random.randint(0, len(new_solution[machine1]) - 1)
                order2_idx = random.randint(0, len(new_solution[machine2]) - 1)
                # Swap one order between the two machines
                new_solution[machine1][order1_idx], new_solution[machine2][order2_idx] = (
                    new_solution[machine2][order2_idx],
                    new_solution[machine1][order1_idx]
                )

        # Reset the current processing times and colors after the swap
        temp_time = {machine: 0 for machine in M}
        temp_colour = {machine: None for machine in M}

        # Update the schedule to reflect the changes made in the new solution
        update_schedule(new_solution, temp_time, temp_colour)
        
        # Calculate the penalty for the new solution
        new_penalty = calculate_penalty_for_schedule(new_solution)
        
        # Accept or reject the new solution based on penalty difference and temperature
        penalty_diff = new_penalty - current_penalty

        if penalty_diff < 0:  # Accept the new solution if it's better (lower penalty)
            current_solution = new_solution
            current_penalty = new_penalty
            no_improvement_count = 0  # Reset counter if improvement happens
            if new_penalty < best_penalty:  # Update best solution if it's the best so far
                best_solution = copy.deepcopy(new_solution)
                best_penalty = new_penalty
        else:  # Accept worse solutions probabilistically to allow exploration
            acceptance_probability = np.exp(-penalty_diff / max(temperature, 1e-8))
            if random.random() < acceptance_probability:
                current_solution = new_solution
                current_penalty = new_penalty
            else:
                no_improvement_count += 1  # Increment no improvement counter if solution is rejected

        # Cooling: Reduce the temperature after each iteration
        temperature *= cooling_rate

        # Keep track of penalties and temperatures for plotting
        penalty_history.append(current_penalty)
        best_penalty_history.append(best_penalty)
        temperature_history.append(temperature)

        # Detonation condition: If no improvement for a set number of iterations, reset
        if no_improvement_count >= detonation_threshold:
            logger.info(msg=f"Detonation at iteration {iteration}")
            # Reset to a random solution to escape local optima
            current_solution = copy.deepcopy(scheduled_orders)
            current_penalty = calculate_penalty_for_schedule(current_solution)
            no_improvement_count = 0  # Reset counter
            temperature *= 1.5 # Slightly increase the temperature 

        # Early stop condition: If the temperature is very low, stop the process
        if temperature < 1e-8:
            break

        # Log progress every 100 iterations
        if iteration % 100 == 0:
            logger.info(msg=f"Iteration {iteration}, Current Penalty: {current_penalty}, Best Penalty: {best_penalty}, Temperature: {temperature}")

    # Plot the penalty and temperature history
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Total Penalty', color='b')
    ax1.plot(penalty_history, label='Current Penalty', color='b', marker='o')
    ax1.plot(best_penalty_history, label='Best Penalty', color='g', linestyle='--')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Temperature', color='r')  # we already handled the x-label with ax1
    ax2.plot(temperature_history, label='Temperature', color='r', linestyle='-.')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.legend(loc='upper right')

    plt.title('Simulated Annealing: Penalty and Temperature over Iterations')
    fig.tight_layout()  # to ensure the plot fits within the figure
    plt.grid(True)
    plt.show()
    
    return best_penalty, best_solution  # Return the best penalty and solution found

# Function to plot a Gantt chart for the resulting schedule
def plot_schedule(scheduled_orders, method, orders_df):
    """Plots a Gantt chart of the scheduled orders and marks late orders.

    Args:
        scheduled_orders (dict): A dictionary where each machine maps to its list of orders.
        method (str): The method used to calculate the schedule (e.g., Simulated Annealing).
        orders_df (pd.DataFrame): DataFrame containing order details (e.g., deadlines, penalties).
    """
    fig, ax = plt.subplots(figsize=(10, 6))  # Create a figure for the Gantt chart
    
    y_pos = 0  # Variable to position the orders vertically by machine

    # Iterate over each machine and its scheduled orders
    for machine, orders in scheduled_orders.items():
        y_pos += 1  # Increment position for each machine
        for order in orders:
            # Retrieve detailed information about the order
            order_info = orders_df[orders_df['order'] == order['order']].iloc[0]
            order_color = order['colour']  # The color of the order
            processing_time = order['end_time'] - order['start_time'] - order['setup_time']  # Calculate processing time
            setup_time = order['setup_time']  # Setup time before the order starts
            start_time = order['start_time']
            end_time = order['end_time']
            deadline = order_info['deadline']  # Retrieve the deadline for this order
            
            # Check if the order is late based on its end time and deadline
            is_late = end_time > deadline

            # Draw the bar representing the processing time on the Gantt chart
            bar_color = order_color
            ax.barh(y_pos, processing_time, left=start_time + setup_time, color=bar_color, edgecolor='black')
            
            # Add the order label and mark it with '*' if the order is late
            label = f"Order {order['order']}" + (' * ' if is_late else '')
            ax.text(start_time + setup_time + processing_time / 2, y_pos, label, ha='center', va='center', color='black', rotation=90, weight='bold')

            # Draw the setup time portion of the bar
            if setup_time > 0:
                ax.barh(y_pos, setup_time, left=start_time, color='gray', edgecolor='black', hatch='//')
    
    # Set the labels and titles for the Gantt chart
    ax.set_yticks(range(1, len(scheduled_orders) + 1))
    ax.set_yticklabels([f"Machine {m}" for m in scheduled_orders.keys()])
    ax.set_xlabel('Time')
    ax.set_ylabel('Machines')
    ax.set_title(f'Gantt Chart for Paint Shop Scheduling using {method}. {file_path}')
    plt.show()

def export_schedule_to_excel(scheduled_orders, file_path, orders_df):
    """
    Creates a Pandas DataFrame of the schedule and exports it to an Excel file.

    Args:
        scheduled_orders (dict): The scheduled orders with start time, end time, machine, and setup time.
        file_path (str): Path to the Excel file to be created.
        orders_df (pd.DataFrame): DataFrame containing the original order details (e.g., deadlines, penalties).

    Returns:
        None
    """
    # Initialize list to store schedule data
    schedule_data = []

    # Loop through scheduled orders and extract relevant information
    for machine, orders in scheduled_orders.items():
        for order in orders:
            # Find order information in orders_df
            order_info = orders_df[orders_df['order'] == order['order']].iloc[0]

            # Collect details for each order
            schedule_data.append({
                'Machine': machine,
                'Order': order['order'],
                'Start Time': order['start_time'],
                'End Time': order['end_time'],
                'Setup Time': order['setup_time'],
                'Processing Time': order['end_time'] - order['start_time'] - order['setup_time'],
                'Color': order['colour'],
                'Deadline': order_info['deadline'],
                'Late': order['end_time'] > order_info['deadline']  # Check if the order is late
            })

    # Create a Pandas DataFrame
    schedule_df = pd.DataFrame(schedule_data)

    # Export the DataFrame to an Excel file
    schedule_df.to_excel(f'Schedule {file_path}', index=False)

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
    initial_temperature = 200     # Set the initial temperature
    cooling_rate = 0.9995     # Set the cooling rate
    logger.info(f'max iterations: {max_iterations}, initial temperature: {initial_temperature}, cooling rate: {cooling_rate}')
    sa_penalty, sa_schedule = simulated_annealing(max_iterations, initial_temperature, cooling_rate)
    logger.info(msg=f"Simulated Annealing total penalty: {sa_penalty:.2f}")
    end_time_sa = time.time()
    logger.info(msg=f'Elapsed time Meta Heuristics (sa): {end_time_sa - start_time_sa:.6f}')
    plot_schedule(sa_schedule, 'Simulated Annealing', orders_df)
    
    # Export schedule to excel
    logger.info(msg='Exporting to Excel...')
    export_schedule_to_excel(scheduled_orders, file_path, orders_df)
    logger.info(msg='------------------- END OF OPTIMIZATION -------------------')

if __name__ == "__main__":
    main()   