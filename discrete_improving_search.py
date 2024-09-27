# DISCRETE IMPROVING SEARCH
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
import random                       
import matplotlib.pyplot as plt     
import logging                      
import time                         
import mathematical_model

VERYBIGNUMBER = 4242424242

random.seed(42)                     
logger = logging.getLogger(name='paintshop-logger')
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(message)s',
                    handlers=[logging.FileHandler("paintshop.log")])

def makespan(schedule):
    """
        Calculates the makespan of the current schedule
        
        Arguments:
            schedule (dict): {machine: [(job, finish_time, setup_time), ...]}
            
        Returns:
            The maximum completion time across all machines (i.e., the makespan)
    """
    return max(max(finish_time for _, finish_time, _ in jobs) for jobs in schedule.values())

def nearest_neighbor(schedule, jobs):
    """
        Nearest neighbor heuristic based on the job processing times (surface/machine speed)
        
        Arguments:
            schedule (dict): {machine: [(job, finish_time, setup_time), ...]}
            jobs (list): List of jobs to be scheduled
        
        Returns:
            schedule (dict): Nearest neighbor-based initial schedule
    """
    logger.info("Applying Nearest Neighbor heuristic for initial schedule")
    
    # Sort jobs by processing time (ascending)
    jobs_sorted = sorted(jobs, key=lambda x: x['surface'] / x['speed'])
    
    for job in jobs_sorted:
        # Assign each job to the machine that will finish the earliest
        best_machine = min(schedule, key=lambda m: makespan({m: schedule[m]}))
        machine_jobs = schedule[best_machine]
        last_finish_time = machine_jobs[-1][1] if machine_jobs else 0
        process_time = job['surface'] / job['speed']
        setup_time = job['setup_time']
        finish_time = last_finish_time + setup_time + process_time
        schedule[best_machine].append((job, finish_time, setup_time))
    
    return schedule

def random_schedule(machines, jobs):
    """
        Generates a random schedule for the jobs
        
        Arguments:
            machines (list): List of machines
            jobs (list): List of jobs
        
        Returns:
            schedule (dict): A random schedule
    """
    schedule = {m: [] for m in machines}
    random.shuffle(jobs)
    
    for job in jobs:
        machine = random.choice(machines)
        machine_jobs = schedule[machine]
        last_finish_time = machine_jobs[-1][1] if machine_jobs else 0
        process_time = job['surface'] / job['speed']
        setup_time = job['setup_time']
        finish_time = last_finish_time + setup_time + process_time
        schedule[machine].append((job, finish_time, setup_time))
    
    return schedule

def two_opt_schedule(schedule, machines, jobs):
    """
        Apply the 2-opt heuristic to improve the current schedule by swapping jobs
        
        Arguments:
            schedule (dict): The current schedule
            machines (list): List of machines
            jobs (list): List of jobs
        
        Returns:
            The improved schedule after applying 2-opt
    """
    logger.info("Applying 2-opt to improve the schedule")
    
    best_schedule = schedule.copy()
    best_makespan = makespan(best_schedule)
    
    while True:
        found_improvement = False
        
        # Iterate through machines and attempt to swap jobs to improve the schedule
        for machine1 in machines:
            for i, (job1, finish_time1, setup_time1) in enumerate(best_schedule[machine1]):
                for machine2 in machines:
                    if machine1 == machine2:
                        continue
                    for j, (job2, finish_time2, setup_time2) in enumerate(best_schedule[machine2]):
                        # Swap job1 and job2 and recalculate the makespan
                        new_schedule = best_schedule.copy()
                        new_schedule[machine1][i], new_schedule[machine2][j] = best_schedule[machine2][j], best_schedule[machine1][i]
                        new_makespan = makespan(new_schedule)
                        
                        if new_makespan < best_makespan:
                            logger.info(f"Improved schedule with swap: makespan reduced from {best_makespan:.2f} to {new_makespan:.2f}")
                            best_schedule = new_schedule
                            best_makespan = new_makespan
                            found_improvement = True
        
        if not found_improvement:
            break
    
    return best_schedule

def plot_schedule_dis(schedule):
    """
        Plot the current schedule as a Gantt chart
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    color_map = {
        'yellow': 'yellow', 
        'blue': 'blue', 
        'red': 'red', 
        'green': 'green', 
        'gray': 'gray'
    }
    
    for machine_idx, (machine, jobs) in enumerate(schedule.items()):
        for job, finish_time, setup_time in jobs:
            color = color_map.get(job['colour'], 'gray')
            start_time = finish_time - (job['surface'] / job['speed']) - setup_time
            ax.add_patch(plt.Rectangle(
                (start_time, machine_idx),  # (x, y)
                job['surface'] / job['speed'],  # width
                0.9,  # height
                edgecolor='black',
                facecolor=color,
                hatch="/" if setup_time > 0 else None
            ))
            ax.text(
                start_time + (job['surface'] / job['speed']) / 2, 
                machine_idx + 0.45, 
                f"Order {job['order_id']}", 
                color='black', 
                ha='center', 
                va='center',
                rotation=90
            )

    ax.set_ylim(0, len(schedule))
    ax.set_xlabel('Time')
    ax.set_ylabel('Machines')
    ax.set_yticks(range(len(schedule)))
    ax.set_yticklabels([f"Machine {i+1}" for i in range(len(schedule))])
    ax.set_title('Paint Shop Scheduling Using Discrete Improving Search')
    
    plt.show()

scheduled_orders = random_schedule(orders_df, machines_df, setups_df)
two_opt_schedule(scheduled_orders)
plot_schedule_dis(scheduled_orders)