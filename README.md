This readme will contain the written parts of the answers to the question being answered here:

a) Deriving the corresponding update rule, using the local finite approximation of the derivative:

$$\Delta \phi = f \Rightarrow f = \frac{\phi(x+h,y)+\phi(x-h,y)+\phi(x,y+h)+\phi(x,y-h)-4 \phi(x,y)}{h^2}$$

From this we isolate phi of the point we're trying to calculate:

$$\phi(x,y) = \frac{\phi(x+h,y)+\phi(x-h,y)+\phi(x,y+h)+\phi(x,y-h) - f(x,y)}{4 h^2} $$

In the case of our computer code, we assume h = 1, and as the grid is finite, we get that in each point: 

$$data\[i,j\] = \frac{data\[i+1,j\]+data\[i-1,j\]+data\[i,j+1\]+data[i,j-1\] - \rho\[i,j\]}{4}$$

Which is what we expect.

b) Designing the MPI code: done in the files, the 2D .c file is the one asked by the question itself.

c) The comparison between both codes is explictly given by the log files and visualized in the Time_Comparison.png file, both strategies were tested in a total of 8 cores, we can see in the plot that both strategies are mostly evenly matched, with only a slight difference of performance itched out of the 1D strategy, only in a small grid one can see a sizable difference in favour of the 1D strategy. This can be explained by the fact that the 2D strategy, given its complexity in terms of handling both the compactification and flattening of the grid for transfer to each core, gets to use slightly more time in tasks like calculating the sizes of the smaller grids, flattening the grid to be sent to the cores, deflattening and compiling the results, and others, all which have both a constant complexity component non negligible at smaller grids, and a grid dimension dependant component too, in fact, once the grid size increases, these tasks' constant component become negligible, but the grid size component stays and keeps a very small but constant difference between strategies. These extra loads are mostly non dependant on the communication strategy, and in fact the extra work to accomodate the strategy is the sole culprit, not the strategy itself, in fact both strategies communicate between cores the exact same number of datapoints, so with ideal handling both should be equivalent.
