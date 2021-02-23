from task_functions import *


x_range = 18
y_range = 18
vertex = 5
seq_length = 15
num_seq = 10
np.random.seed = 777
SEQ = np.zeros((num_seq, seq_length), dtype=np.int)
JUMP = np.zeros((num_seq, seq_length, 3), dtype=np.float)

Grid, Grid_dist = GridGen(x_range,y_range, vertex, plot=True)
jump_type = np.random.choice([0, 1, 2], size = num_seq, p = [0, 1, 0])


i = 0
while i < num_seq:
    Trajectory, Jump, status = SeqGen(Grid, Grid_dist, seq_length,vertex, jump=jump_type[i], plot = False)
    if status:
        SeqAnimate(Grid, Trajectory,Jump, name_idx=i)
        SEQ[i, :] = Trajectory
        JUMP[i, :] = Jump
        i += 1
    
TPGen(Grid, SEQ, JUMP)