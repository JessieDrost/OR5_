# CONSTRUCTIEVE HEURISTIEK
"""
    Pseudocode:
    Step 0: Initialization. Start with all-free initual partial solution
    x^(0) = (#,...,#) and set solution index t <- 0.
    Step 1: Stopping. If all components of current solution x^(t) are fixed, stop
    and output xhat <- x^(t) as an approximate optimum.
    Step 2: Step. Choose a free componen x_p of partial solution x^(t) and a value
    for it that is plausible leads to good feasible completions. Then, advance to
    partial solution x^(t+1) identical to x^(t) except that x_p is fixed at the chosen
    value.
    Step 3: Increment. Increment t <- t+1, and return to Step 1.
"""
# WISKUNDIG MODEL

import pandas as pd
import numpy as np

# Gegevens importeren vanuit excel
orders_df = pd.read_excel('paintshop_september_2024.xlsx', sheet_name='Orders')
machines_df = pd.read_excel('paintshop_september_2024.xlsx', sheet_name='Machines')
setups_df = pd.read_excel('paintshop_september_2024.xlsx', sheet_name='Setups')

# Sets definiëren
B = orders_df['order'].tolist()  # Bestellingen
M = machines_df['machine'].tolist()  # Machines
H1 = setups_df['from_colour'].tolist()  # Startkleuren
H2 = setups_df['to_colour'].tolist()  # Eindkleuren

# Hulpfuncties voor berekeningen
def processing_time(surface, speed):
    return surface / speed

def setup_time(from_colour, to_colour, setups_df):
    if from_colour == to_colour or from_colour is None:  # Geen wisseling of initiale setup
        return 0
    else:
        return setups_df[(setups_df['from_colour'] == from_colour) & (setups_df['to_colour'] == to_colour)]['setup_time'].values[0]

# Startwaarden voor elke machine
current_time = {machine: 0 for machine in M}  # Tijd per machine
current_color = {machine: None for machine in M}  # Kleur per machine
scheduled_orders = {machine: [] for machine in M}  # Orders per machine
available_orders = B.copy()  # Beschikbare orders

# Planning functie: toewijzen van een order aan een machine
def schedule_order(order, machine, current_time, current_color):
    order_info = orders_df[orders_df['order'] == order].iloc[0]
    surface = order_info['surface']
    colour = order_info['colour']
    deadline = order_info['deadline']
    penalty = order_info['penalty']

    # Bereken verwerkingstijd en omsteltijd
    process_time = processing_time(surface, machines_df[machines_df['machine'] == machine]['speed'].values[0])
    set_time = setup_time(current_color[machine], colour, setups_df)

    # Update tijd en kleur van de machine
    current_time[machine] += process_time + set_time
    current_color[machine] = colour

    # Bereken vertraging en boete
    delay = max(0, current_time[machine] - deadline)
    total_penalty = delay * penalty

    # Plan de order
    scheduled_orders[machine].append(order)
    available_orders.remove(order)

    return total_penalty


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

VERYBIGNUMBER = 424242424242

# Colors for visualization
color_map = {
    'Green': 'green',
    'Yellow': 'yellow',
    'Blue': 'blue',
    'Red': 'red'
}

# Function to find setup time between colors
def get_setup_time(from_color, to_color, setups_df):
    setup_time = setups_df[(setups_df['from_colour'] == from_color) & 
                           (setups_df['to_colour'] == to_color)]['setup_time']
    return setup_time.values[0] if not setup_time.empty else 0

# Constructive Nearest Neighbor heuristic function
# Constructive Nearest Neighbor heuristic function with all machines being used
def nearest_neighbor_scheduling(orders_df, machines_df, setups_df):
    # Available machines and their speeds
    machines = machines_df['machine'].tolist()
    machine_speeds = machines_df['speed'].tolist()
    
    # Track current time and current color for each machine
    current_time = {m: 0 for m in machines}
    current_color = {m: None for m in machines}
    
    # Assign scheduled orders to machines
    scheduled_orders = {m: [] for m in machines}

    available_orders = orders_df.to_dict('records')
    
    # Continue until all orders are scheduled
    while available_orders:
        for machine in machines:
            if not available_orders:
                break  # Exit if no orders remain
            
            min_penalty = VERYBIGNUMBER
            best_order = None
            for order in available_orders:
                # Get setup time if needed
                if current_color[machine]:
                    setup_time = get_setup_time(current_color[machine], order['colour'], setups_df)
                else:
                    setup_time = 0  # No setup time for the first order

                # Calculate time to process the order
                processing_time = order['surface'] / machines_df.loc[machine-1, 'speed']
                total_time = setup_time + processing_time

                # Calculate delay and penalty
                finish_time = current_time[machine] + total_time
                delay = max(0, finish_time - order['deadline'])
                penalty = delay * order['penalty']
                
                # Find the order with the minimum penalty
                if penalty < min_penalty:
                    min_penalty = penalty
                    best_order = order
            
            # Assign best order to the machine
            if best_order:
                processing_time = best_order['surface'] / machines_df.loc[machine-1, 'speed']
                setup_time = get_setup_time(current_color[machine], best_order['colour'], setups_df) if current_color[machine] else 0
                finish_time = current_time[machine] + processing_time + setup_time

                # Update current time, color and append the order to the machine schedule
                current_time[machine] = finish_time
                current_color[machine] = best_order['colour']
                scheduled_orders[machine].append((best_order, current_time[machine], setup_time))

                # Remove the order from available orders
                available_orders.remove(best_order)
    
    return scheduled_orders


# Two-opt swap method (improves the schedule)
def two_opt_swap(scheduled_orders):
    # Implement the two-opt swap logic here for optimization
    pass

# Plotting function to visualize schedules
def plot_schedule_ch(scheduled_orders):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = 0
    for machine, orders in scheduled_orders.items():
        y_pos += 1  # Stacking machines on the plot
        start_time = 0
        
        for order, end_time, setup_time in orders:
            order_color = order['colour']
            processing_time = order['surface'] / machines_df.loc[machine-1, 'speed']
            
            # Draw order processing
            ax.barh(y_pos, processing_time, left=start_time + setup_time, color=color_map[order_color], edgecolor='black')
            ax.text(start_time + setup_time + processing_time / 2, y_pos, f"Order {order['order']}", ha='center', va='center', color='black', rotation=90)

            # Draw setup time (if any)
            if setup_time > 0:
                ax.barh(y_pos, setup_time, left=start_time, color='gray', edgecolor='black', hatch='//')
            
            # Update start time for the next order
            start_time += setup_time + processing_time

    ax.set_yticks(range(1, len(scheduled_orders) + 1))
    ax.set_yticklabels([f"Machine {m}" for m in scheduled_orders.keys()])
    ax.set_xlabel('Time')
    ax.set_ylabel('Machines')
    ax.set_title('Paint Shop Scheduling Using Constructive Heuristics')
    plt.show()

# Main execution
scheduled_orders = nearest_neighbor_scheduling(orders_df, machines_df, setups_df)
two_opt_swap(scheduled_orders)
plot_schedule_ch(scheduled_orders)