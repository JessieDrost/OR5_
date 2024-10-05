import random
import math
import matplotlib.pyplot as plt # (from my Discrete Improving Search code)
import logging
import time  

VERYBIGNUMBER = 424242424242
NUMLOCATIONS = 25

random.seed(42)                     
logger = logging.getLogger(name='sa-logger')
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(message)s',
                    handlers=[logging.FileHandler("sa.log")])

# Parameters for simulated annealing
initial_temperature = 100
cooling_rate = 0.995
iterations_per_temperature = 1000

def euclidean_distance(point1, point2):
    """
        Euclidean distance between two points (from my Discrete Improving Search code)
        
        Arguments:
            point1, point2: (x,y)-coordinates of two points
            
        Returns:
            shortest distance from point1 to point2 'as the crow flies' 
    """
    x1, y1 = point1
    x2, y2 = point2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def total_distance(locations, tour):
    """
        Calculates the total distance of a tour along all locations (from my Discrete Improving Search code)
        Replaces ChatGPT's tour_distance function

        Arguments:
            set of locations ({0,1,...,num_locations-1})
            tour (list): a permutation of the set of locations

        Returns:
            sum of all distances between two consecutive locations in tour 
    """
    totalDistance = 0
    for i in range(len(tour)):
        fromLocation = locations[tour[i]]
        toLocation = locations[tour[(i + 1) % len(tour)]]
        totalDistance += euclidean_distance(fromLocation, toLocation)

    return totalDistance

def random_tour(locations):
    """
        Generates a tour visiting all locations at random

        Arguments:
            set of locations ({0,1,...,num_locations-1})

        Returns:
            (list) a permutation of the set of locations        
    """
    num_locations = len(locations)
    rnd_tour = random.sample(range(num_locations), num_locations)

    total_distance_rnd_tour = total_distance(locations, rnd_tour)
    logger.info(msg=f"RND tour has distance {total_distance_rnd_tour:.2f}")    

    return rnd_tour

# Simulated annealing function
def simulated_annealing(locations, tour, temperature, cooling_rate, iterations_per_temp):
    iteration_log = []
    start_time = time.time()

    current_tour = tour
    best_tour = tour
    current_distance = total_distance(locations, current_tour)
    best_distance = current_distance
    iteration = 0
    log_record = (iteration,current_distance,best_distance,temperature)
    iteration_log.append(log_record)

    while temperature > 1e-3:
        for _ in range(iterations_per_temp):
            # Generate a neighboring solution by swapping two random cities
            i, j = random.sample(range(len(tour)), 2)
            neighbor_tour = current_tour[:]
            neighbor_tour[i], neighbor_tour[j] = neighbor_tour[j], neighbor_tour[i]
            neighbor_distance = total_distance(locations, neighbor_tour)

            logger.debug(msg=f"Swap locations {i} and {j} in current tour gives tour with length {neighbor_distance:6.2f}")    
            # Decide whether to accept the neighbor solution
            delta = neighbor_distance - current_distance
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_tour = neighbor_tour
                current_distance = neighbor_distance
                logger.debug(msg=f"Move has gain {delta:6.2f}, is accepted")

                # Update the best solution if necessary
                if current_distance < best_distance:
                    best_tour = current_tour
                    best_distance = current_distance
                    logger.debug(msg=f"Best solution is updated")

                print(f"Temperature {temperature:8.4f}, current distance {current_distance:6.2f}, best distance {best_distance:6.2f}", end='\r')

            log_record = (iteration+1,current_distance,best_distance,temperature)
            iteration_log.append(log_record)
            iteration += 1

        # Cool the temperature
        temperature *= cooling_rate

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("\n")
    print(f"Elapsed time: {elapsed_time:.2f}")

    # visualize iteration log
    plt.figure(0)
    x = [ilog[0] for ilog in iteration_log]
    y1 = [ilog[1] for ilog in iteration_log]
    y2 = [ilog[2] for ilog in iteration_log]
    y3 = [ilog[3] for ilog in iteration_log]

    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.plot(x, y3)
    plt.xlabel('Iteration')
    plt.ylabel('Distance')
    plt.title(f'Progress of SA heuristic applied to {len(tour)} nodes')    

    return best_tour, best_distance

def main():
    # Generate random instance in [0,100]x[0,100]
    # ----------
    num_locations = NUMLOCATIONS
    coordinates = []
    distance = {}
    for i in range(num_locations):
        x = random.uniform(0,100)
        y = random.uniform(0,100)
        coordinates.append((x,y))

    for p1 in range(num_locations):
        for p2 in range(num_locations):
            point1 = coordinates[p1]
            point2 = coordinates[p2]
            dist = euclidean_distance(point1,point2)
            keyToRetrieve = (point1,point2)
            distance[keyToRetrieve] = dist

    logger.info(msg=f"Random instance with {num_locations} locations")    

    # Generate a tour at random
    # ----------
    rnd_tour = random_tour(coordinates)
    total_distance_rnd_tour = total_distance(coordinates,rnd_tour)

    # Call the simulated annealing function
    best_tour, best_distance = simulated_annealing(coordinates, rnd_tour, initial_temperature, cooling_rate, iterations_per_temperature)

    # Print the results
    print("\n")
    print("Best Tour:", best_tour)
    print("Best Distance:", best_distance)


    # Plot the random solution and the discrete improving search solution 
    # (from my Discrete Improving Search code)
    # ---

    # Separate the x and y values into separate lists
    x = [coord[0] for coord in coordinates]
    y = [coord[1] for coord in coordinates]

    # In Figure 1, we show both the random solution and the NN solution
    plt.figure(1)
    plt.scatter(x, y)
    plt.xlabel('X-axis Label')
    plt.ylabel('Y-axis Label')

    for i in range(1, len(coordinates)):
        x1, y1 = coordinates[best_tour[i - 1]]
        x2, y2 = coordinates[best_tour[i]]
        plt.plot([x1, x2], [y1, y2], 'b-')  
    x1, y1 = coordinates[best_tour[len(coordinates)-1]]
    x2, y2 = coordinates[best_tour[0]]
    plt.plot([x1, x2], [y1, y2], 'b-')  

    for i in range(1, len(coordinates)):
        x1, y1 = coordinates[rnd_tour[i - 1]]
        x2, y2 = coordinates[rnd_tour[i]]
        plt.plot([x1, x2], [y1, y2], '--', color='grey', linewidth=0.8)  
    x1, y1 = coordinates[rnd_tour[len(coordinates)-1]]
    x2, y2 = coordinates[rnd_tour[0]]
    plt.plot([x1, x2], [y1, y2], '--', color='grey', linewidth=0.8)  

    plt.title(f'Solutions on {len(coordinates)} nodes; RND tour: {total_distance_rnd_tour:.2f}, SA tour: {best_distance:.2f}')
    plt.show()

if __name__ == "__main__":
    main()