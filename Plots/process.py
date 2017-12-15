import numpy as np

files = ["run_100k_rory.out", "run_100k_rory_old.out", "run_100k_amy.out", "run_100k_relu_amy.out",
    "run_200k_rory.out", "run_200k_rory_old.out", "run_200k_amy.out", "run_200k_relu_amy.out",
    "run_500k_rory.out", "run_500k_rory_old.out", "run_500k_amy.out", "run_500k_relu_amy.out",
    "run_1M_rory.out", "run_1M_rory_old.out", "run_1M_amy.out", "run_1M_relu_amy.out"]

# files = ["stand_1M_relu_amy.out"]

for fileName in files:
    stepRewards = []
    iterStats = []
    with open(fileName, "r") as f:
        lines = []
        for line in f:
            line = line.strip()
            if line.startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')) and ("core" not in line) and ("wheat" not in line):
                if ("episode" in line):
                    stats = line.split(" ")
                    numEpisodes = int(stats[0])
                    avgReward = float(stats[4])
                    minReward = float(stats[5][1:len(stats[5])-1])
                    maxReward = float(stats[6][:len(stats[6])-1])
                    iterStats.append([numEpisodes, avgReward, minReward, maxReward])
                else:
                    splitLine = line.split(": ")
                    stepRewards.append(float(splitLine[len(splitLine)-1]))
    # npSteps = np.asarray(stepRewards, dtype=np.float32)
    npSteps = np.array(stepRewards)
    npStats = np.array([np.array(x) for x in iterStats])
    stepsFileName = fileName[:len(fileName)-4] + "_steps.txt"
    statsFileName = fileName[:len(fileName)-4] + "_stats.txt"
    np.savetxt(stepsFileName, npSteps, delimiter = ',')
    np.savetxt(statsFileName, npStats, delimiter = ',')
