"""
This file analyzes the tensorboard file to calculate the training speed.
Intervals where evaluation is performed are filtered out by excluding outliers.

command line argument: path to tensorboard logs file
"""
import sys
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

if __name__ == '__main__':
    path = sys.argv[1]
    ea = event_accumulator.EventAccumulator(path,
            size_guidance={
                event_accumulator.SCALARS: 0,
            }) 
    ea.Reload()

    print(ea.Tags())

    timestamps = []
    steps = []
    for event in ea.Scalars('epe3'):
        timestamps.append(event.wall_time)
        steps.append(event.step)

    timestamps = np.array(timestamps)
    steps = np.array(steps)

    steps_diffs = steps[1:]-steps[:-1]
    step_size = steps_diffs[0]
    if np.any(steps_diffs != step_size):
        print("non uniform step size, aborting")
        exit(0)
    else:
        print(f"uniform step size: {step_size}")

    ts_diffs = timestamps[1:]-timestamps[:-1] 

    max_diff = np.max(ts_diffs)
    min_diff = np.min(ts_diffs)
    avg_diff = np.mean(ts_diffs)
    std_diff = np.std(ts_diffs)

    threshold = std_diff*3
    outliers = np.abs(ts_diffs - avg_diff) > threshold
    inliers = np.abs(ts_diffs - avg_diff) <= threshold 
    num_outliers = np.sum(outliers)
    num_inliers = np.sum(inliers)
    avg_diff_inliers = np.mean(ts_diffs[inliers])
    
    print(f"max {max_diff}")
    print(f"min {min_diff}")
    print(f"avg {avg_diff}")
    print(f"std {std_diff}")


    print(f"s/it {avg_diff/step_size}")

    # filter out steps that contain the time taken for evaluation
    print(f"number of outliers {num_outliers}   number of inliers {num_inliers}")
    print(f"s/it (filtered outliers) {avg_diff_inliers/step_size}")
