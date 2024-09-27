# META-HEURISTIEK
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

