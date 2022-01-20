### Learning dynamic partitioning  based on deep reinforcement learning

This is the project code of dynamic partition algorithm PPO controller. This is the project code of dynamic partition algorithm PPO controller. The contents and functions of each directory are listed below.

#### Project structure description：

- **baselines:** Some classic dynamic partition algorithms, including applyall, after all, smalldc, autostore, feedback, etc.
- **data:** This directory includes workload generation file and some workload data.
- **db:** Database related module. Provide driver class, transaction class, data types related to table structure and load structure, cost model, etc.
- **environment:** Environment module of RL. Where env4.py is the final environment file of the paper.
- **experiment:** Drawing related files in the paper.
- **par_algorithm:** Partitioner module. Some files related to SCVP algorithm.
- **visualization:** Workload selector module. Workload selection for repartitioning. It also includes some code files related to workload data visualization.
- **other files:**
  - adapter_controller.py: Partition generator module.
  - stable_base_ppo.py: PPO controller module.
  - main.py: The main program entry file, which is used to conduct the comparative experiment. The dataset and dynamic partition algorithms can be flexibly specified.

#### Project dependency package:

Some modules required by the project can be seen in the requirement.txt file.  It is recommended to run the following commands on the console:

`cd PPO-Controller`

`conda create -n dy-ppo python==3.6`

`source activate dy-ppo`

`pip install -r requirements.txt`

