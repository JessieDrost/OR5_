"""Solving Traveling Salesman Problem 
    by applying Nearest Neighbor (greedy constructive heuristic)
"""

VERYBIGNUMBER = 424242424242
NUMLOCATIONS = 10

"""I want to be able to generate random instances
"""
import random                       
random.seed(42)                     

"""And to visualize my instance and tour
"""
import matplotlib.pyplot as plt     

"""And to keep track of all steps
"""
import logging                      
logger = logging.getLogger(name='nn-logger')
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(message)s',
                    handlers=[logging.FileHandler("nn.log")])

"""And, finally, to determine the computation time
"""
import time                         


def euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def total_distance(locations, tour):
    totalDistance = 0
    for i in range(len(tour)):
        fromLocation = locations[tour[i]]
        toLocation = locations[tour[(i + 1) % len(tour)]]
        totalDistance += euclidean_distance(fromLocation, toLocation)
    
    return totalDistance
        
def main():
    """Generate random instance
       ------------------------
    """
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

    """Apply nearest neighbor
       ----------------------
    """
    logger.debug(msg="Apply Nearest Neighbor (NN)")        
    startTime = time.time()

    nn_tour = []
    visited = [False]*num_locations

    """Initialize nn-tour"""
    nn_tour.append(0) 
    visited[0] = True
    last_visited = 0
    logger.debug(msg=f"Initialize NN tour: start with location 0")    

    while visited.count(False) > 0:
        min_distance = VERYBIGNUMBER # very big number
        for j in range(num_locations):
            if not(visited[j]) and (distance[coordinates[last_visited],coordinates[j]] < min_distance):
                    min_distance = distance[coordinates[last_visited],coordinates[j]]
                    nearest = j
        nn_tour.append(nearest)
        visited[nearest] = True
        last_visited = nearest
        logger.debug(msg=f"Add location {nearest} and arc ({last_visited},{nearest}) with distance {min_distance:.2f} to tour")    
    endTime = time.time()

    total_distance_nn_tour = total_distance(coordinates, nn_tour)
    logger.info(msg=f"NN tour has distance {total_distance_nn_tour:.2f}; time: {endTime-startTime:.6f}")    

    # Plot the random solution and the NN solution
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
        x1, y1 = coordinates[nn_tour[i - 1]]
        x2, y2 = coordinates[nn_tour[i]]
        plt.plot([x1, x2], [y1, y2], 'b-')  
    x1, y1 = coordinates[nn_tour[len(coordinates)-1]]
    x2, y2 = coordinates[nn_tour[0]]
    plt.plot([x1, x2], [y1, y2], 'b-')  
    plt.title(f'Solutions on {len(coordinates)} nodes; NN tour: {total_distance_nn_tour:.2f}')

    plt.show()

if __name__ == "__main__":
    main()
