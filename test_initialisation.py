#######################################################
# Copyright (c) 2021 Qihao Shan, All Rights Reserved #
#######################################################

import numpy as np
import matplotlib.pyplot as plt
from arena_class import arena
import dm_objects

fig, axis = plt.subplots(2, 2)
hypotheses = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
resample_prob = 0.02
#dm_ob = dm_objects.DM_object_DC(hypotheses, exp_length=1, diss_length=1, resample_prob=resample_prob)
dm_ob = dm_objects.DM_object_ob_sharing(hypotheses=hypotheses, l_cache=10, decay_coeff=0.8, decay_coeff_2=0.4)
#dm_ob = dm_objects.DM_object_individual(hypotheses)
a = arena(0.45, 'Random', 12, 1, hypotheses, dm_ob, axis=axis)

Max_step = 30000

for i in range(Max_step):
    a.random_walk()
    if i % 100 == 0:
        a.dm_object.make_decision(a.robots, a.coo_array)
    a.plot_arena(i)

print(a.dm_object.decision_array)