# Reinforcement Learning To Solve Mountain Car Problem
This is the project for the **COMP4125 Designing Intelligent Agents** in 2025.

## Main repo of our group
> from zjy2414's repo: https://github.com/zjy2414/Mountain-Car-Agent

## Structure of the repo

> .<br/>
> ├─humanoid_code # Codes for the humanoid problem.<br/>
> |&nbsp;&nbsp;&nbsp;&nbsp;├─ `standup_sb3_predict_a2c.py` # Code for A2C.<br/>
> |&nbsp;&nbsp;&nbsp;&nbsp;├─ `standup_sb3_predict.py` # Code for SAC, TD3 and PPO.<br/>
> |&nbsp;&nbsp;&nbsp;&nbsp;├─ `standup_sb3_train_a2c.py` # Code for A2C.<br/>
> |&nbsp;&nbsp;&nbsp;&nbsp;└─ `standup_sb3_train.py` # Code for SAC, TD3 and PPO.<br/>
> ├─mountain_car_code # Codes for the mountain car problem without using Q-learning.<br/>
> |&nbsp;&nbsp;&nbsp;&nbsp;├─ `predict.py` # Testing code.<br/>
> |&nbsp;&nbsp;&nbsp;&nbsp;└─ `train.py` # Training code.<br/>
> ├─mountain_car_q_learning # Codes for the mountain car problem using Q-learning.<br/>
> |&nbsp;&nbsp;&nbsp;&nbsp;├─ `exp.py` # Version before applying the Grid Search method to find the best hyperparameter configurations and after integrating for the Q-learning algorithm on the Mountain Car environment.<br/>
> |&nbsp;&nbsp;&nbsp;&nbsp;├─ `grid_search.py` # Version after applying the Grid Search method.<br/>
> |&nbsp;&nbsp;&nbsp;&nbsp;└─ `q_learning.py` # Version before integrating.<br/>
> └─other_unused_code # Codes that are not used eventually.<br/>
>  &nbsp;&nbsp;&nbsp;&nbsp;└─ `humanoid.py` <br/>

## How to use codes?
1. To use the latest version of the *Q-learning* code, please use the `grid_search.py` in `mountain_car_q_learning`. This code is developed containing three evaluation methods, including **descriptive statistics**, **visual analysis** and **inferential statistics**. 

```bash
python grid_search.py
```

2. To use the version before applying the Grid Search method of the *Q-learning* code, please use the `exp.py` in `mountain_car_q_learning`.

```bash
python exp.py
```

3. The code is developed in VSCode with the help of a virtual environment of Anaconda3.
4. Other codes can represent the process of this project.
