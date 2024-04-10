
FOR /L %i IN (1,1,100) DO cfx5export -ensight -7 -binary -corrected -include -long -name dp%i -timestep -1 -user 3 D:\P2\ffs_obstacle_DoE_files\dp%i\CFX-10\CFX\FFS_009.res