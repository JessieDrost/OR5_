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

random.seed(70)
logger = logging.getLogger(name='sa-logger')
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(message)s',
                    handlers=[logging.FileHandler("sa.log")])

# Parameters for simulated annealing

initial_temperature = 100
cooling_rate = 0.9995
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
def simulated_annealing(max_iterations, initial_temp, cooling_rate, detonation_threshold=300):
    # Start with the feasible solution from the greedy planner
    total_penalty, scheduled_orders = greedy_paint_planner()

    current_solution = copy.deepcopy(scheduled_orders)
    current_penalty = calculate_penalty_for_schedule(current_solution)

    temperature = initial_temp
    best_solution = copy.deepcopy(current_solution)
    best_penalty = current_penalty

    penalty_history = [current_penalty]
    best_penalty_history = [best_penalty]
    temperature_history = [temperature]

    no_improvement_count = 0  # Counter for detecting stagnation (no improvements)

    for iteration in range(max_iterations):
        new_solution = copy.deepcopy(current_solution)
        
        # Neighborhood search: Swap orders within the same machine or between machines
        if random.random() < 0.5:  # Swap within a machine (small local change)
            machine = random.choice(M)
            if len(new_solution[machine]) > 1:
                order1_idx, order2_idx = random.sample(range(len(new_solution[machine])), 2)
                new_solution[machine][order1_idx], new_solution[machine][order2_idx] = (
                    new_solution[machine][order2_idx],
                    new_solution[machine][order1_idx]
                )
        else:  # Swap orders between two machines
            machine1, machine2 = random.sample(M, 2)
            if new_solution[machine1] and new_solution[machine2]:
                order1_idx = random.randint(0, len(new_solution[machine1]) - 1)
                order2_idx = random.randint(0, len(new_solution[machine2]) - 1)
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

        # Accept or reject new solution based on penalty and temperature
        penalty_diff = new_penalty - current_penalty

        if penalty_diff < 0:
            # Accept the better solution
            current_solution = new_solution
            current_penalty = new_penalty
            no_improvement_count = 0  # Reset improvement counter
            if new_penalty < best_penalty:
                best_solution = copy.deepcopy(new_solution)
                best_penalty = new_penalty
        else:
            # Accept worse solution with a probability dependent on the temperature
            acceptance_probability = np.exp(-penalty_diff / max(temperature, 1e-8))
            if random.random() < acceptance_probability:
                current_solution = new_solution
                current_penalty = new_penalty
            else:
                no_improvement_count += 1

        # Cooling: Update temperature using the exponential schedule
        temperature *= cooling_rate

        # Track penalties and temperature
        penalty_history.append(current_penalty)
        best_penalty_history.append(best_penalty)
        temperature_history.append(temperature)

        # Detonation: if no improvement after a set number of iterations
        if no_improvement_count >= detonation_threshold:
            logger.info(f"Detonation at iteration {iteration}")
            # Reset current solution to a completely random new solution (force exploration)
            current_solution = copy.deepcopy(scheduled_orders)
            current_penalty = calculate_penalty_for_schedule(current_solution)
            no_improvement_count = 0  # Reset counter
            temperature *= 1.5  # Slightly raise the temperature to escape local optima

        # Early stop if temperature is too low
        if temperature < 1e-8:
            break

        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Current Penalty: {current_penalty}, Best Penalty: {best_penalty}, Temperature: {temperature}")

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

    return best_penalty, best_solution

# Call the simulated annealing function
max_iterations = 10000
total_penalty, optimised_scheduled_orders = simulated_annealing(max_iterations, initial_temperature, cooling_rate)
plot_schedule(optimised_scheduled_orders, 'Simulated Annealing', orders_df)
