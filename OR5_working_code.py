# WISKUNDIG MODEL

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt    
import time 
#Current time per machine is 0
#Scheduled orders empty list
#Available orders is orders
#Current colour per machine
#Def setup time(prev colour, current colour)
#Def nearest neighbour 
#For index if constraints

# Gegevens importeren vanuit excel
orders_df = pd.read_excel('paintshop_september_2024.xlsx', sheet_name='Orders')
machines_df = pd.read_excel('paintshop_september_2024.xlsx', sheet_name='Machines')
setups_df = pd.read_excel('paintshop_september_2024.xlsx', sheet_name='Setups')

# Sets definiëren
B = orders_df['order'].tolist() # Bestellingen
M = machines_df['machine'].tolist() # Machines
H1 = setups_df['from_colour'].tolist() # Start kleuren
H2 = setups_df['to_colour'].tolist() # Eind kleuren 

current_time_M1 = 0
current_time_M2 = 0
current_time_M3 = 0
current_color_M1 = None
current_color_M2 = None
current_color_M3 = None 
scheduled_orders = []
available_orders = B

#handige cijfertjes definiëren
VERYBIGNUMBER = 424242424242

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

def setup_time(previous_color, current_color):
    
    
def nearest_neighbor():
    
    
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

# META-HEURISTIEK