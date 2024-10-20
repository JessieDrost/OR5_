"""Solving the scheduling problem
    using a greedy constructive heuristic.

    Step 0: all machines are free 
    Step 1: stop if all orders have been assigned to a machine 
    Step 2: start with the fastest machine and choose from the orders with the lowest setup time the one with the highest penalty, and assign it to the machine. If assigning this order would result in a penalty, this order is assigned to the next fastest machine. If an assignment results in a penalty for every machine, the order is assigned to the fastest machine.
    Step 3: repeat steps 1 and 2 
    Calculate the start and end time of each order and the setup time between all consecutive orders, plot this with matplotlib on a Gantt chart 
    print total penalty
"""

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

VERY_BIG_NUMBER = 424242424242

# Import data from Excel
orders_df = pd.read_excel('paintshop_november_2024.xlsx', sheet_name='Orders')
machines_df = pd.read_excel('paintshop_november_2024.xlsx', sheet_name='Machines')
setups_df = pd.read_excel('paintshop_november_2024.xlsx', sheet_name='Setups')

# Define sets
B = orders_df['order'].tolist()  # Orders
M = machines_df['machine'].tolist()  # Machines
H1 = setups_df['from_colour'].tolist()  # Start colours
H2 = setups_df['to_colour'].tolist()  # End colours

# Initial values for each machine
current_time = {machine: 0 for machine in M}  # Time per machine
current_color = {machine: None for machine in M}  # Colour per machine
scheduled_orders = {machine: [] for machine in M}  # Orders per machine
available_orders = B.copy()  # Available orders

# Helper functions for calculations
def processing_time(surface, speed):
    """
    Calculate the processing time based on surface area and machine speed.

    Args:
        surface (float): The surface area of the order.
        speed (float): The speed of the machine.

    Returns:
        float: The processing time required for the order.
    """
    return surface / speed

def setup_time(from_colour, to_colour, setups_df):
    """
    Calculate setup time based on the colour transition.

    Args:
        from_colour (str): The colour to transition from.
        to_colour (str): The colour to transition to.
        setups_df (pd.DataFrame): DataFrame containing setup times.

    Returns:
        float: The setup time required for the colour transition.
    """
    if from_colour == to_colour or from_colour is None:
        return 0
    else:
        return setups_df[(setups_df['from_colour'] == from_colour) & (setups_df['to_colour'] == to_colour)]['setup_time'].values[0]

def calculate_total_penalty(current_time, deadline, penalty):
    """
    Calculate total penalty based on the delay from the deadline.

    Args:
        current_time (float): The current completion time of the order.
        deadline (float): The deadline for the order.
        penalty (float): The penalty rate for missing the deadline.

    Returns:
        float: The total penalty incurred due to delay.
    """
    delay = max(0, current_time - deadline)
    return delay * penalty

def greedy_paint_planner():
    """
    Greedy algorithm to assign orders to machines based on speed, setup time, and penalty.

    Returns:
        tuple: Total penalty and a dictionary of scheduled orders for each machine.
    """
    total_penalty = 0
    
    # Sort machines by speed (fastest first)
    sorted_machines = machines_df.sort_values(by='speed', ascending=False)['machine'].tolist()
    
    # Loop until all orders are assigned
    while available_orders:
        for machine in sorted_machines:
            if not available_orders:
                break  # Stop if all orders are assigned
            
            best_order = None
            min_setup_time = VERY_BIG_NUMBER
            max_penalty = -VERY_BIG_NUMBER
            
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

    return total_penalty, scheduled_orders

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

# Execute the algorithm
if __name__ == "__main__":
    total_penalty, scheduled_orders = greedy_paint_planner()
    print(f"Total Penalty: {total_penalty}")
    plot_schedule(scheduled_orders, 'greedy heuristics', orders_df)
