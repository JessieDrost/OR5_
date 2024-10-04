import pandas as pd
import numpy as np
import copy
import random
import time
import matplotlib.pyplot as plt
from constructive_heuristics import greedy_paint_planner, plot_schedule

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
    return surface / speed

def setup_time(from_colour, to_colour, setups_df):
    if from_colour == to_colour or from_colour is None:
        return 0
    else:
        return setups_df[(setups_df['from_colour'] == from_colour) & (setups_df['to_colour'] == to_colour)]['setup_time'].values[0]

def calculate_total_penalty(current_time, deadline, penalty):
    delay = max(0, current_time - deadline)
    return delay * penalty

def calculate_penalty_for_schedule(scheduled_orders):
    total_penalty = 0
    for machine, orders in scheduled_orders.items():
        for order in orders:
            completion_time = order['end_time']
            order_info = orders_df[orders_df['order'] == order['order']].iloc[0]
            penalty = order_info['penalty']
            deadline = order_info['deadline']
            total_penalty += calculate_total_penalty(completion_time, deadline, penalty)
    return total_penalty

# Define update_schedule function here
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

# Two-exchange algorithm
def two_exchange():
    total_penalty, scheduled_orders = greedy_paint_planner()

    # Initially update the schedule
    update_schedule(scheduled_orders, current_time, current_colour)

    improved = True
    penalty_history = []

    while improved:
        improved = False
        best_penalty = calculate_penalty_for_schedule(scheduled_orders)
        penalty_history.append(best_penalty)

        for machine1 in scheduled_orders:
            for order1 in scheduled_orders[machine1]:
                for machine2 in scheduled_orders:
                    if machine1 == machine2:
                        continue
                    for order2 in scheduled_orders[machine2]:
                        temp_scheduled_orders = copy.deepcopy(scheduled_orders)
                        idx1 = next(i for i, o in enumerate(temp_scheduled_orders[machine1]) if o['order'] == order1['order'])
                        idx2 = next(i for i, o in enumerate(temp_scheduled_orders[machine2]) if o['order'] == order2['order'])
                        temp_scheduled_orders[machine1][idx1], temp_scheduled_orders[machine2][idx2] = (
                            temp_scheduled_orders[machine2][idx2],
                            temp_scheduled_orders[machine1][idx1]
                        )

                        temp_time = copy.deepcopy(current_time)
                        temp_colour = copy.deepcopy(current_colour)
                        update_schedule(temp_scheduled_orders, temp_time, temp_colour)
                        temp_penalty = calculate_penalty_for_schedule(temp_scheduled_orders)

                        if temp_penalty < best_penalty:
                            scheduled_orders = temp_scheduled_orders
                            best_penalty = temp_penalty
                            improved = True
                            break
                    if improved:
                        break
            if improved:
                break

        if not improved:
            break

    total_penalty = calculate_penalty_for_schedule(scheduled_orders)
    penalty_history.append(total_penalty)

    plt.plot(penalty_history, marker='o', color='b')
    plt.xlabel('Iteration')
    plt.ylabel('Total Penalty')
    plt.grid(True)
    plt.show()

    return total_penalty, scheduled_orders

# Simulated Annealing
def simulated_annealing(max_iterations=10000, initial_temp=1000, cooling_rate=0.995):
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

            new_solution[machine1][order1_idx], new_solution[machine2][order2_idx] = (
                new_solution[machine2][order2_idx],
                new_solution[machine1][order1_idx]
            )

            temp_time = copy.deepcopy(current_time)
            temp_colour = copy.deepcopy(current_colour)
            update_schedule(new_solution, temp_time, temp_colour)
            new_penalty = calculate_penalty_for_schedule(new_solution)

            if new_penalty < current_penalty:
                current_solution = new_solution
                current_penalty = new_penalty
                if new_penalty < best_penalty:
                    best_solution = copy.deepcopy(new_solution)
                    best_penalty = new_penalty
            else:
                acceptance_probability = np.exp((current_penalty - new_penalty) / temperature)
                if random.random() < acceptance_probability:
                    current_solution = new_solution
                    current_penalty = new_penalty

        temperature *= cooling_rate
        penalty_history.append(current_penalty)

        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Current Penalty: {current_penalty}, Best Penalty: {best_penalty}")

        if temperature < 1e-8:
            break

    plt.plot(penalty_history, marker='o', color='b')
    plt.xlabel('Iteration')
    plt.ylabel('Total Penalty')
    plt.grid(True)
    plt.show()

    return best_penalty, best_solution

# Uncomment the method you want to test first
total_penalty, optimised_scheduled_orders = simulated_annealing()
plot_schedule(optimised_scheduled_orders, 'Simulated Annealing', orders_df)

# total_penalty, optimised_scheduled_orders = two_exchange()
# plot_schedule(optimised_scheduled_orders, '2-exchange', orders_df)

