# WISKUNDIG MODEL
"""
    Stap 0: start met de toelaatbare oplossing die volgt uit de greedy_paint_planner functie. 
    Stap 1: als geen enkele wisseling in starttijd en machine tussen twee orders resulteert in een lagere totale penalty, stop.
    Stap 2: kies twee orders waarvan het omwisselen van de starttijd en machine van deze twee orders zou resulteren in een lagere totale penalty en wissel ze om.
    Stap 3: update de planning met de nu omgewisseld de orders vast.
    Stap 4: ga terug naar stap 1.
bereken de start en eindtijd van iedere order en de set-up tijd tussen alle opeenvolgende orders, plot deze met matplotlib op een gantt schema 
print totale penalty
    """
import pandas as pd
import numpy as np
import copy
import time
import matplotlib.pyplot as plt
from constructive_heuristics import greedy_paint_planner, plot_schedule

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

def calculate_penalty_for_schedule(scheduled_orders):
    """Hulpfunctie om de totale penalty te berekenen voor het gehele schema."""
    total_penalty = 0
    for machine, orders in scheduled_orders.items():
        for order in orders:
            completion_time = order['end_time']
            order_info = orders_df[orders_df['order'] == order['order']].iloc[0]
            penalty = order_info['penalty']
            deadline = order_info['deadline']
            total_penalty += calculate_total_penalty(completion_time, deadline, penalty)
    return total_penalty

def exchange_orders(order1, machine1, order2, machine2, scheduled_orders):
    """Hulpfunctie om twee orders tussen machines te wisselen."""
    # Wissel de orders in de planning
    idx1 = next(i for i, o in enumerate(scheduled_orders[machine1]) if o['order'] == order1)
    idx2 = next(i for i, o in enumerate(scheduled_orders[machine2]) if o['order'] == order2)
    
    scheduled_orders[machine1][idx1], scheduled_orders[machine2][idx2] = (
        scheduled_orders[machine2][idx2],
        scheduled_orders[machine1][idx1]
    )

def update_schedule(scheduled_orders, current_time, current_color):
    """Hulpfunctie om het schema bij te werken na een wisseling."""
    for machine, orders in scheduled_orders.items():
        current_time[machine] = 0
        current_color[machine] = None
        for order in orders:
            order_info = orders_df[orders_df['order'] == order['order']].iloc[0]
            process_time = processing_time(order_info['surface'], machines_df[machines_df['machine'] == machine]['speed'].values[0])
            set_time = setup_time(current_color[machine], order_info['colour'], setups_df)
            
            start_time = current_time[machine]
            end_time = start_time + process_time + set_time
            
            order['start_time'] = start_time
            order['end_time'] = end_time
            order['setup_time'] = set_time
            current_time[machine] = end_time
            current_color[machine] = order_info['colour']

def discrete_improving_search():
    """Discrete improving search algoritme voor het optimaliseren van de planning."""
    # Start met de toelaatbare oplossing van de greedy planner
    total_penalty, scheduled_orders = greedy_paint_planner()
    
    # Variabelen om de huidige status bij te houden
    current_time = {machine: 0 for machine in M}
    current_color = {machine: None for machine in M}
    update_schedule(scheduled_orders, current_time, current_color)
    
    improved = True
    penalty_history = []  # List to store the penalties at each iteration
    
    while improved:
        improved = False
        best_penalty = calculate_penalty_for_schedule(scheduled_orders)
        
        # Append the current penalty to the history list
        penalty_history.append(best_penalty)
        
        # Probeer elk paar orders op verschillende machines te wisselen
        for machine1 in scheduled_orders:
            for order1 in scheduled_orders[machine1]:
                for machine2 in scheduled_orders:
                    if machine1 == machine2:
                        continue
                    for order2 in scheduled_orders[machine2]:
                        # Maak een tijdelijke kopie van de planning
                        temp_scheduled_orders = copy.deepcopy(scheduled_orders)
                        
                        # Wissel de orders
                        exchange_orders(order1['order'], machine1, order2['order'], machine2, temp_scheduled_orders)
                        
                        # Werk het schema bij na de wisseling
                        temp_time = copy.deepcopy(current_time)
                        temp_color = copy.deepcopy(current_color)
                        update_schedule(temp_scheduled_orders, temp_time, temp_color)
                        
                        # Bereken de totale penalty voor de nieuwe planning
                        temp_penalty = calculate_penalty_for_schedule(temp_scheduled_orders)
                        
                        # Als de penalty is verbeterd, accepteer de wisseling
                        if temp_penalty < best_penalty:
                            scheduled_orders = temp_scheduled_orders
                            best_penalty = temp_penalty
                            improved = True
                            break  # Ga terug naar het begin als er een verbetering is gevonden
                    if improved:
                        break
            if improved:
                break
        
        # Als geen enkele wisseling heeft geleid tot verbetering, stop het algoritme
        if not improved:
            print("Geen verdere verbeteringen mogelijk.")
            break
    
    # Na optimalisatie: geef de totale penalty en het schema
    total_penalty = calculate_penalty_for_schedule(scheduled_orders)
    print(f"Totale penalty: {total_penalty}")
    
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


# Uitvoeren van het algoritme
if __name__ == "__main__":
    total_penalty, optimized_scheduled_orders = discrete_improving_search()
    plot_schedule(optimized_scheduled_orders, '2-exchange', orders_df)