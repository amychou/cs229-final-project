import numpy as np
import matplotlib.pyplot as plt

files = ["run_100k_rory.out", "run_100k_rory_old.out", "run_100k_amy.out", "run_100k_relu_amy.out",
    "run_200k_rory.out", "run_200k_rory_old.out", "run_200k_amy.out", "run_200k_relu_amy.out",
    "run_500k_rory.out", "run_500k_rory_old.out", "run_500k_amy.out", "run_500k_relu_amy.out",
    "run_1M_rory.out", "run_1M_rory_old.out", "run_1M_amy.out", "run_1M_relu_amy.out"]

files = ["stand_1M_relu_amy.out"]

for fileName in files:
    base = fileName[:len(fileName)-4]
    steps = np.loadtxt(base + "_steps.txt", delimiter=',')
    stats = np.loadtxt(base + "_stats.txt", delimiter=',')
    iterations = range(stats.size/4)

    plt.figure(figsize=(18, 6))
    plt.title("Rewards per iteration (Standing)", fontsize=28)
    plt.plot(iterations, stats[:,1], 'blue', label = 'Average episode reward')
    plt.plot(iterations, stats[:,3], 'r--', label = 'Maximum episode reward')
    plt.plot(iterations, stats[:,2], 'g--', label = 'Minimum episode reward')
    plt.legend(loc = 'upper right', prop={'size': 14})
    plt.xlabel("Iteration", fontsize=18)
    plt.ylabel("Reward", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig("plot_" + base + "_rewards.png")
    plt.close()

    plt.figure(figsize=(18, 6))
    plt.title("Reward per step", fontsize=28)
    plt.plot(range(steps.size), steps, 'black')
    plt.xlabel("Step", fontsize=18)
    plt.ylabel("Reward", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig("plot_" + base + "_steprewards.png")
    plt.close()

    '''
    plt.figure(figsize=(10, 12))
    plt.title("Average number of episode steps per iteration", fontsize=28)
    plt.plot(iterations, 10000./stats[:,0], 'black')
    plt.xlabel("Iteration", fontsize=18)
    plt.ylabel("Average number of steps per episode", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig("plot_" + base + "_steps.png")
    plt.close()'''
