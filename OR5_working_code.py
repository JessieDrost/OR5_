# WISKUNDIG MODEL
import pandas as pd
import numpy as np
import os
print(os.getcwd())

# Gegevens importeren vanuit excel
orders_df = pd.read_excel('paintshop_september_2024.xlsx', sheet_name='Orders')
machines_df = pd.read_excel('paintshop_september_2024.xlsx', sheet_name='Machines')
setups_df = pd.read_excel('paintshop_september_2024.xlsx', sheet_name='Setups')

# Sets definiëren
O = orders_df['order'].tolist() # Bestellingengit
M = machines_df['machine'].tolist() # Machines
H = setups_df['from_colour'].unique().tolist() # Kleuren (unieke waardes om dubbele te voorkomen want daar kan je niks mee)

#Parameters definiëren
s_o = dict(zip(orders_df['order'], orders_df['surface'])) # Oppervlakte per bestelling
k_o = dict(zip(orders_df['order'], orders_df['colour'])) # Kleur van iedere bestelling
d_o = dict(zip(orders_df['order'], orders_df['deadline'])) # Deadline van iedere bestelling
c_o = dict(zip(orders_df['order'], orders_df['penalty'])) # Boete voor iedere bestelling
v_m = dict(zip(machines_df['machine'], machines_df['speed'])) # Snelheid van iedere machine

print("Running the correct script!")
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

# DISCRETE IMPROVING SEARCH
"""
    Pseudocode: 
    Step 0: Initialization. Choose any starting feasible solution x^(0), and set
    solution index t <- 0.
    Step 1: Local Optimum. If no more delta_x in move set M is both improving 
    and feasible at current solution x^(t), stop. Point x^(t) is a local optimum.
    Step 2: Move. Choose some imporving feasible move delta_x elementof M as delta_x^(t+1).
    Step 3: Update. x^(t+1) <- x^(t) + delta_x^(t+1)
    Step 4: Increment. Increment t <- t+1, and return to Step 1.
"""
print(orders_df.head())
print("Running the correct script!")
# META-HEURISTIEK