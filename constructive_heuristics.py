"""Solving scheduling Problem
    using greedy constructive heuristic
    
    Stap 0: alle machines zijn vrij 
    Stap 1: stop als alle orders aan een machine zijn toegewezen 
    Stap 2: begin bij de snelste machine en kies van de orders met de laagste set-up tijd degene met de hoogste penalty en wijs deze toe aan de machine. Als het toewijzen van deze order zou resulteren in een penalty, wordt deze order toegewezen aan de volgende snelste machine. Als een toewijzing bij iedere machine resulteert in een penalty, wordt de order toegewezen aan de snelste machine.
    Stap 3: herhaal stap 1 en 2 
bereken de start en eindtijd van iedere order en de set-up tijd tussen alle opeenvolgende orders, plot deze met matplotlib op een gantt schema 
print totale penalty

"""
import pandas as pd
import numpy as np
import logging
import time
import matplotlib.pyplot as plt

VERYBIGNUMBER = 424242424242

# Colors for visualization
color_map = {
    'Green': 'green',
    'Yellow': 'yellow',
    'Blue': 'blue',
    'Red': 'red'
}

# Gegevens importeren vanuit excel
orders_df = pd.read_excel('paintshop_september_2024.xlsx', sheet_name='Orders')
machines_df = pd.read_excel('paintshop_september_2024.xlsx', sheet_name='Machines')
setups_df = pd.read_excel('paintshop_september_2024.xlsx', sheet_name='Setups')

# Sets definiëren
B = orders_df['order'].tolist()  # Bestellingen
M = machines_df['machine'].tolist()  # Machines
H1 = setups_df['from_colour'].tolist()  # Startkleuren
H2 = setups_df['to_colour'].tolist()  # Eindkleuren

# Startwaarden voor elke machine
current_time = {machine: 0 for machine in M}  # Tijd per machine
current_color = {machine: None for machine in M}  # Kleur per machine
scheduled_orders = {machine: [] for machine in M}  # Orders per machine
available_orders = B.copy()  # Beschikbare orders

# Hulpfuncties voor berekeningen
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

# Greedy toewijzingsfunctie
def greedy_paint_planner():
    """
    Greedy algorithm to assign orders to machines based on speed, setup time, and penalty.
    """
    total_penalty = 0
    
    # Sorteer machines op snelheid (snelste eerst)
    sorted_machines = machines_df.sort_values(by='speed', ascending=False)['machine'].tolist()
    
    # Loop totdat alle orders zijn toegewezen
    while available_orders:
        for machine in sorted_machines:
            if not available_orders:
                break  # Stop als alle orders zijn toegewezen
            
            best_order = None
            min_setup_time = VERYBIGNUMBER
            max_penalty = -VERYBIGNUMBER
            
            # Zoek naar de beste order voor de huidige machine
            for order in available_orders:
                order_info = orders_df[orders_df['order'] == order].iloc[0]
                surface = order_info['surface']
                colour = order_info['colour']
                deadline = order_info['deadline']
                penalty = order_info['penalty']
                
                # Bereken verwerkingstijd en omsteltijd
                process_time = processing_time(surface, machines_df[machines_df['machine'] == machine]['speed'].values[0])
                set_time = setup_time(current_color[machine], colour, setups_df)
                completion_time = current_time[machine] + process_time + set_time
                
                # Controleer of er een penalty is
                current_penalty = calculate_total_penalty(completion_time, deadline, penalty)
                
                # Selecteer de order met de laagste set-up tijd en hoogste penalty
                if set_time < min_setup_time or (set_time == min_setup_time and current_penalty > max_penalty):
                    best_order = order
                    min_setup_time = set_time
                    max_penalty = current_penalty
            
            # Als er een beste order gevonden is, wijs deze toe aan de machine
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
                
                # Verwijder de toegewezen order uit de lijst van beschikbare orders
                available_orders.remove(best_order)
                total_penalty += max_penalty  # Boete optellen

    return total_penalty, scheduled_orders

def plot_schedule(scheduled_orders):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = 0
    for machine, orders in scheduled_orders.items():
        y_pos += 1  # Voor elke machine
        for order in orders:
            order_color = order['colour']
            processing_time = order['end_time'] - order['start_time'] - order['setup_time']
            setup_time = order['setup_time']
            start_time = order['start_time']
            
            # Teken verwerkingstijd
            ax.barh(y_pos, processing_time, left=start_time + setup_time, color=color_map[order_color], edgecolor='black')
            ax.text(start_time + setup_time + processing_time / 2, y_pos, f"Order {order['order']}", ha='center', va='center', color='black', rotation=90)

            # Teken setup tijd
            if setup_time > 0:
                ax.barh(y_pos, setup_time, left=start_time, color='gray', edgecolor='black', hatch='//')
    
    ax.set_yticks(range(1, len(scheduled_orders) + 1))
    ax.set_yticklabels([f"Machine {m}" for m in scheduled_orders.keys()])
    ax.set_xlabel('Time')
    ax.set_ylabel('Machines')
    ax.set_title('Gantt Chart for Paint Shop Scheduling')
    plt.show()

# Uitvoeren van het algoritme
if __name__ == "__main__":
    total_penalty, scheduled_orders = greedy_paint_planner()
    print(f"Total Penalty: {total_penalty}")
    plot_schedule(scheduled_orders)