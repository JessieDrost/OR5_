import pandas as pd
import numpy as np
import copy
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
H1 = setups_df['from_colour'].tolist()  # Starting colours
H2 = setups_df['to_colour'].tolist()  # Ending colours

# Initial values for each machine
current_time = {machine: 0 for machine in M}  # Time per machine
current_colour = {machine: None for machine in M}  # Colour per machine
scheduled_orders = {machine: [] for machine in M}  # Orders per machine
available_orders = B.copy()  # Available orders

# Helper functions for calculations
def processing_time(surface, speed):
    """
    Calculate the processing time based on surface area and machine speed.

    Args:
        surface (float): Surface area of the order.
        speed (float): Processing speed of the machine.

    Returns:
        float: Processing time for the order.
    """
    return surface / speed

def setup_time(from_colour, to_colour, setups_df):
    """
    Calculate the setup time based on the colour transition.

    Args:
        from_colour (str): Starting colour.
        to_colour (str): Ending colour.
        setups_df (DataFrame): DataFrame containing setup times for colour transitions.

    Returns:
        int: Setup time required for the transition.
    """
    if from_colour == to_colour or from_colour is None:
        return 0
    else:
        return setups_df[(setups_df['from_colour'] == from_colour) & (setups_df['to_colour'] == to_colour)]['setup_time'].values[0]

def calculate_total_penalty(current_time, deadline, penalty):
    """
    Calculate the total penalty based on the delay from the deadline.

    Args:
        current_time (int): Completion time of the order.
        deadline (int): Deadline of the order.
        penalty (int): Penalty rate for missing the deadline.

    Returns:
        int: Penalty for the order.
    """
    delay = max(0, current_time - deadline)
    return delay * penalty

def calculate_penalty_for_schedule(scheduled_orders):
    """
    Helper function to calculate the total penalty for the entire schedule.

    Args:
        scheduled_orders (dict): Dictionary containing the scheduled orders per machine.

    Returns:
        int: Total penalty for the schedule.
    """
    total_penalty = 0
    for machine, orders in scheduled_orders.items():
        for order in orders:
            completion_time = order['end_time']
            order_info = orders_df[orders_df['order'] == order['order']].iloc[0]
            penalty = order_info['penalty']
            deadline = order_info['deadline']
            total_penalty += calculate_total_penalty(completion_time, deadline, penalty)
    return total_penalty

def two_exchange():
    """
    Discrete improving search algorithm for optimising the schedule.

    Returns:
        int: Total penalty after optimisation.
        dict: Optimised schedule with orders per machine.
    """
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
            print("No further improvements possible.")
            break

    # After optimisation: return the total penalty and the schedule
    total_penalty = calculate_penalty_for_schedule(scheduled_orders)
    print(f"Total penalty: {total_penalty}")

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

    return total_penalty, scheduled_orders



# Execute the algorithm
if __name__ == "__main__":
    total_penalty, optimised_scheduled_orders = two_exchange()
    plot_schedule(optimised_scheduled_orders, '2-exchange', orders_df)
