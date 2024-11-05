import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import tqdm as tqdm


### Initializations

x_dim = 3
y_dim = 3
iterations = 100

# Honest and Rogue Agents

num_agents = 4
num_byz = 1
ind_byz = 2

# Quadratic Objective Functions
# f_i(x) = 0.5 || y_i - M_i*x ||^2

y_arr = [None] * num_agents
M_arr = [None] * num_agents

for i in range(num_agents):
    y_i = np.random.normal(0, 1, (y_dim, 1))
    M_i = np.random.normal(0, 1, (y_dim, x_dim))
    
    y_arr[i] = y_i
    M_arr[i] = M_i



### Finding optimal for all agents

x_cp = cp.Variable((x_dim, 1))
obj_func = 0
for i in range(num_agents):
    obj_i = 0.5 * cp.norm(y_arr[i] - M_arr[i] @ x_cp)
    obj_func += obj_i
prob_all = cp.Problem(cp.Minimize(obj_func))

prob_all.solve()
x_opt_all = np.array(x_cp.value).reshape((-1, 1))
print("\nOptimal x for optimization over all agents: \n\t", x_opt_all.flatten())



### Finding optimal for honest agents

x_cp = cp.Variable((x_dim, 1))
obj_func = 0
for i in range(num_agents):
    if i == ind_byz:
        continue
    obj_i = 0.5 * cp.norm(y_arr[i] - M_arr[i] @ x_cp)
    obj_func += obj_i
prob_honest = cp.Problem(cp.Minimize(obj_func))

prob_honest.solve()
x_opt_honest = np.array(x_cp.value).reshape((-1, 1))
print("\nOptimal x for optimization over honest agents: \n\t", x_opt_honest.flatten())



