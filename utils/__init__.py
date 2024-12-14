def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_min = int(elapsed_time / 60)
    elapsed_sec = int(elapsed_time -  (elapsed_time * 60))

    return elapsed_min, elapsed_sec