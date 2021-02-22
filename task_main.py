from task_functions import *


x_range = 18      # X range for working area in Kinarm display
y_range = 18      # Y range for working area in Kinarm display
vertex = 5        # Between target distance
seq_length = 15   # Length of the sequence
num_seq = 50      # How many sequences to generate
np.random.seed = 777 

SEQ = np.zeros((num_seq, seq_length), dtype=np.int)
JUMP = np.zeros((num_seq, seq_length, 3), dtype=np.float)

# Generate a grid for possible targets
Grid, Grid_dist = GridGen(x_range,y_range, vertex, plot=True)
# Specify the probability of different jump types (0: no Jump, 1: Jump only in +2, 2: Jump in +1 and +2)
jump_type = np.random.choice([0, 1, 2], size = num_seq, p = [0.7, 0.15, 0.15])



for i in range(num_seq):
    # Create a random path for targets, 
    Trajectory, Jump = SeqGen(Grid, Grid_dist, seq_length,vertex, jump=jump_type[i], plot = False)
    # Save the path as a gif file
    SeqAnimate(Grid, Trajectory,Jump, name_idx=i)
    SEQ[i, :] = Trajectory
    JUMP[i, :] = Jump
    
# saving a Kinarm-friendly txt file
TPGen(Grid, SEQ, JUMP)