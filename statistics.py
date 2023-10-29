import cp

# create a bunch of CPs and note how often the optimization succeeds
# and how many iterations it takes

# create a dictionary to store the results
# for each n, we have optimization successes, assignment successes, and trials
results = []

for n in range(1, 12):
    # make a bunch of CPs
    # for each CP, try to optimize it
    optimization_successes = 0
    assignment_successes = 0
    trials = 0
    for i in range(100):
        mycp = cp.CreasePattern()
        mycp.side = 500
        mycp.add_square_vertices()
        for i in range(n):
            mycp.add_random_vertex()
        mycp.push_to_edge(10)
        mycp.triangulate()
        mycp.evenize_vertices()
        mycp.remove_edge_folds()
        res1 = mycp.optimize()
        if res1.success == True:
            optimization_successes += 1
            res2 = mycp.assign_mv()
            if not res1 == []:
                assignment_successes += 1
        trials += 1
    results.append([optimization_successes, assignment_successes, trials])

# save results ot a csv file
import csv
with open('results.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["n", "optimization successes", "assignment successes", "trials"])
    for i in range(len(results)):
        writer.writerow([i+1, results[i][0], results[i][1], results[i][2]])

