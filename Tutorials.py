print('hoi')

"""
    Solving randomly generated instances of the
    Traveling Salesman Problem (TSP) for a given number of locations
    by applying Discrete Improving Search with 2-opt neighborhood structure

    - intermediate actions are logged in log file
    - computation time is determined 

    To compare the performance of 2-opt,  nearest neighbor is applied as well.
    
"""

VERYBIGNUMBER = 424242424242
NUMLOCATIONS = 5

import random                       
import matplotlib.pyplot as plt     
import logging                      
import time                         

random.seed(42)                     
logger = logging.getLogger(name='2opt-logger')
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(message)s',
                    handlers=[logging.FileHandler("2opt.log")])

def euclidean_distance(point1, point2):
    """
        Euclidean distance between two points
        
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
        Calculates the total distance of a tour along all locations

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

def nearest_neighbor(locations, distance):
    """
        Determines a tour according to the nearest neighbor heuristic
        (constructive search heuristic)
        
        Arguments:
            set of locations ({0,1,...,num_locations-1})
            distance (dictionary): distance between a pair of two locations
            
        Returns:
            (list) a permutation of the set of locations
        
    """
    logger.debug(msg="Apply Greedy Constructive Search on TSP (Nearest Neighbor)")        
    startTime = time.time()

    num_locations = len(locations)
    nn_tour = []
    visited = [False]*num_locations

    # Initialize nn-tour
    nn_tour.append(0) 
    visited[0] = True
    last_visited = 0
    logger.debug(msg=f"Initialize NN tour: start with location 0")    

    while visited.count(False) > 0:
        min_distance = VERYBIGNUMBER 
        for j in range(num_locations):
            from_location = locations[last_visited]
            to_location = locations[j]
            key = (from_location, to_location)
            if not(visited[j]) and (distance[key] < min_distance):
                    min_distance = distance[key]
                    nearest = j
        nn_tour.append(nearest)
        visited[nearest] = True
        last_visited = nearest
        logger.debug(msg=f"Add location {nearest} and arc ({last_visited},{nearest}) with distance {min_distance:.2f} to tour")    
    endTime = time.time()

    total_distance_nn_tour = total_distance(locations, nn_tour)
    logger.info(msg=f"NN tour has distance {total_distance_nn_tour:.2f}; time: {endTime-startTime:.6f}")    

    return nn_tour

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

def two_opt_tour_best_improvement(locations, distance, init_tour):
    """
        Improves a given initial_tour by applying 2-exchanges and best improvement, resulting in a 2-optimal tour
        (discrete improving search heuristic)

        Arguments:
            set of locations
            distance (dictionary): distance between a pair of two locations
            init_tour (list): a permutation of the set of locations

        Returns:
            (list) a permutation of the set of locations        
    """    
    logger.debug(msg="Apply Discrete Improving Search with 2-exchange neighborhood and best improvement on TSP (2-opt)")        
    startTime = time.time()

    # STEP 0: start with initial tour
    num_locations = len(locations)
    curr_tour = init_tour
    total_distance_curr_tour = total_distance(locations,curr_tour)
    logger.info(msg=f"Initial tour has distance {total_distance_curr_tour:.2f}")    

    while True:
        # determine feasible improving move and gain 
        max_gain = -VERYBIGNUMBER
        fi_move_found = False # 
        for i in range(1,num_locations-1):           # from i=1 upto and including num_locations-2
            for j in range(i+2,num_locations + 1):     # from j=i+2 upto and including num_locations
                if (i==1 and j==num_locations):      # then j=0=i-1; not a feasible tour 
                    break

                loc_imin1 = locations[curr_tour[i-1]]
                loc_i = locations[curr_tour[i]]
                loc_jmin1 = locations[curr_tour[j-1]]
                if j == num_locations:
                    loc_j = locations[curr_tour[0]]
                else:
                    loc_j = locations[curr_tour[j]]

                gain = (distance[loc_imin1,loc_i]+distance[loc_jmin1,loc_j])-(distance[loc_imin1,loc_jmin1]+distance[loc_i,loc_j])
                logger.info(msg=f"Evaluate move ({i},{j})->Remove: ({i-1},{i}) and ({j-1},{j}); gain: {gain:.2f}")  
                
                # if necessary: update best feasible improving move
                if (gain>0 and gain>max_gain):
                    max_gain = gain
                    move = (i,j)
                    fi_move_found = True
        
        # STEP 1: if no move is both improving and feasible: stop; local optimum
        if not(fi_move_found):
            break

        # STEP 2: choose improving feasible move (=best feasible improving move)
        i,j = move

        # STEP 3: update the tour (apply the improving feasible move)
        neighbor_tour = curr_tour[:]
        neighbor_tour[i:j] = reversed(curr_tour[i:j])
        total_distance_neighbor_tour = total_distance(locations,neighbor_tour)
        logger.info(msg=f"Selected feasible improving move ({i},{j}) with gain {max_gain:.2f}, resulting tour has length {total_distance_neighbor_tour:.2f}")    

        curr_tour = neighbor_tour

    endTime = time.time()

    total_distance_two_opt_tour = total_distance(locations, curr_tour)
    logger.info(msg=f"2-opt tour has distance {total_distance_two_opt_tour:.2f}; time: {endTime-startTime:.6f}")    

    return curr_tour

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

    # Apply nearest neighbor to instance
    # ----------
    nn_tour = nearest_neighbor(coordinates,distance)
    total_distance_nn_tour = total_distance(coordinates,nn_tour)

    # Generate a tour at random
    # ----------
    rnd_tour = random_tour(coordinates)
    total_distance_rnd_tour = total_distance(coordinates,rnd_tour)

    # Apply 2-opt to random tour
    # ----------
    dis_tour = two_opt_tour_best_improvement(coordinates,distance,rnd_tour)
    total_distance_dis_tour = total_distance(coordinates,dis_tour)

    # Plot the random solution and the discrete improving search solution
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
        x1, y1 = coordinates[dis_tour[i - 1]]
        x2, y2 = coordinates[dis_tour[i]]
        plt.plot([x1, x2], [y1, y2], 'b-')  
    x1, y1 = coordinates[dis_tour[len(coordinates)-1]]
    x2, y2 = coordinates[dis_tour[0]]
    plt.plot([x1, x2], [y1, y2], 'b-')  
 
    for i in range(1, len(coordinates)):
        x1, y1 = coordinates[rnd_tour[i - 1]]
        x2, y2 = coordinates[rnd_tour[i]]
        plt.plot([x1, x2], [y1, y2], '--', color='grey', linewidth=0.8)  
    x1, y1 = coordinates[rnd_tour[len(coordinates)-1]]
    x2, y2 = coordinates[rnd_tour[0]]
    plt.plot([x1, x2], [y1, y2], '--', color='grey', linewidth=0.8)  
 
    plt.title(f'Solutions on {len(coordinates)} nodes; RND tour: {total_distance_rnd_tour:.2f}, 2OPT tour: {total_distance_dis_tour:.2f}, ' \
              f'NN tour: {total_distance_nn_tour:.2f}')
    plt.show()

if __name__ == "__main__":
    main()