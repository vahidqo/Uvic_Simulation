# Uvic_Simulation

 It is recomneded to use open "Gholamzadeh_Code.ipynb" and then run cells in order.
 
 At the final cell, you can simply change settings:
 
 kwargs = {'env_name': 'ma_gym:Switch2-v1',
              'lr': 0.0005,
              'batch_size': 32,
              'gamma': 0.99,
              'buffer_limit': 50000,
              'log_interval': 20,
              'max_episodes': 200,
              'max_epsilon': 0.9,
              'min_epsilon': 0.1,
              'test_episodes': 5,
              'warm_up_steps': 2000,
              'update_iter': 10,
              'monitor': False}
              
And at ther end you can see resuts in the following link:

https://wandb.ai/uvic_simulation/minimal-marl

***********************************************************************************

If you want to run code in your local machine, clone the reposityry first and then open "Gholamzadeh_Code.py"

then run following code:

pip install -r requirements.txt

And finally run the code.
