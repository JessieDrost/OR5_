import pandas as pd
import numpy as np
import copy
import random
import logging
import time
import matplotlib.pyplot as plt
from constructive_heuristics import greedy_paint_planner, plot_schedule

# Set up logging for debugging purposes
VERYBIGNUMBER = 424242424242

logger = logging.getLogger(name='sa-logger')
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(message)s',
                    handlers=[logging.FileHandler("sa.log")])

# Parameters for simulated annealing
initial_temperature = 100
cooling_rate = 0.998
iterations_per_temperature = 1000

# Import data from Excel
orders_df = pd.read_excel('paintshop_september_2024.xlsx', sheet_name='Orders')
machines_df = pd.read_excel('paintshop_september_2024.xlsx', sheet_name='Machines')
setups_df = pd.read_excel('paintshop_september_2024.xlsx', sheet_name='Setups')

# Define sets
B = orders_df['order'].tolist()  # Orders
M = machines_df['machine'].tolist()  # Machines

# Initial values for each machine
current_time = {machine: 0 for machine in M}
current_colour = {machine: None for machine in M}
scheduled_orders = {machine: [] for machine in M}
available_orders = B.copy()

# Helper functions
def processing_time(surface, speed):
    """Calculate the processing time based on surface area and machine speed."""
    return surface / speed

def setup_time(from_colour, to_colour, setups_df):
    """Calculate the setup time based on the colour transition."""
    if from_colour == to_colour or from_colour is None:
        return 0
    else:
        return setups_df[(setups_df['from_colour'] == from_colour) & 
                         (setups_df['to_colour'] == to_colour)]['setup_time'].values[0]

def calculate_total_penalty(current_time, deadline, penalty):
    """Calculate the total penalty based on the delay from the deadline."""
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

# Simulated Annealing
def simulated_annealing(max_iterations, initial_temp, cooling_rate):
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
            print(f"Iteration {iteration}, Current Penalty: {current_penalty}, Best Penalty: {best_penalty}")

        if temperature < 1e-8:
            break

    # Plot the penalty history
    plt.plot(penalty_history, marker='o', color='b')
    plt.title('Total Penalty per Iteration using Simulated Annealing')
    plt.xlabel('Iteration')
    plt.ylabel('Total Penalty')
    plt.grid(True)
    plt.show()

    return best_penalty, best_solution

# Call the simulated annealing function
max_iterations = 5000
total_penalty, optimised_scheduled_orders = simulated_annealing(max_iterations, initial_temperature, cooling_rate)
plot_schedule(optimised_scheduled_orders, 'Simulated Annealing', orders_df)