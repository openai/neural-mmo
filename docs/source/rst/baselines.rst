.. |icon| image:: /resource/icon/icon_pixel.png

.. figure:: /resource/image/splash.png

|icon| Evaluation Protocol
##########################

Neural MMO provides two standard training and evaluation settings along with pretrained and scripted baselines. Training is performed on a pool of N_TRAIN_MAPS procedurally generated maps simulated for TRAIN_HORIZON timesteps per episode. Evaluation is performed on EVAL_MAPS maps never seen during training (GENERALIZE=False) for EVALUATION_HORIZON timesteps. Approaches that modify these parameters, alter the reward function, or substantially modify the environment configuration are not directly comparable to the baselines below.

Publications including new models should clearly indicate training scale, domain knowledge assumptions, and any other significant changes to the default configuration. Ideally, they should also include a discussion of agent behavior and supporting training/evaluation plots, overlay figures, and/or other evidence specific to the associated work.

To reproduce the tables and figures below, run the associated command. Run *train* or *evaluate* instead of *visualize* to verify results from scratch or from the pretrained model.

|icon| Small Maps
#################

This setting includes 256 80x80 maps (60x60 excluding lava) and supports up to 128 agents/32 NPCs. It is intended for training and evaluation horizons of 1000 timesteps, or 10 minutes in real time. Reasonable policies are trainable with as low as 4 CPU cores and a single modern GPU in a few hours. Evaluation takes only a few minutes.

.. image:: /resource/image/small_isometric.png

Neural Baseline
***************

============ ============ ============ ============ ============
Metric       Min          Max          Mean         Std
============ ============ ============ ============ ============
Population          18.00        57.00        45.95         4.09
Lifetime             0.00      1000.00        46.49       110.78
Skilling            10.00        50.50        14.06         5.92
Combat               3.00        28.00         4.64         3.06
Equipment            0.00        18.00         0.22         1.36
Exploration          0.00        73.00         8.23         6.34
============ ============ ============ ============ ============

.. figure:: /resource/image/baselines/SmallMaps/neural_small_maps.png

   **Reproduction:** python Forge.py visualize --config=SmallMaps --MODEL=small-baseline

.. figure:: /resource/image/baselines/SmallMaps/neural_long_train.png

   New skills emerge up to 100k training environments, but performance dynamics appear mostly stable thereafter

Zero-Shot Transfer (LargeMaps Model)
************************************

============ ============ ============ ============ ============
Metric       Min          Max          Mean         Std
============ ============ ============ ============ ============
Population          15.00        79.00        62.63         7.00
Lifetime             0.00       997.00        62.39        96.93
Skilling            10.00        49.50        15.17         6.27
Combat               3.00        17.00         3.82         1.25
Equipment            0.00        10.00         0.01         0.26
Exploration          0.00        91.00        13.20        10.90
============ ============ ============ ============ ============

.. figure:: /resource/image/baselines/SmallMaps/neural_large_maps.png

   **Reproduction:** python Forge.py visualize --config=SmallMaps --MODEL=large-baseline

Scripted Foraging
*****************

============ ============ ============ ============ ============
Metric       Min          Max          Mean         Std
============ ============ ============ ============ ============
Population          40.00       102.00        80.70         9.71
Lifetime             0.00       995.00        81.01       123.84
Skilling            10.00        49.00        16.70         6.74
Combat               3.00         5.00         3.09         0.29
Equipment            0.00         0.00         0.00         0.00
Exploration          0.00       111.00        17.50        13.12
============ ============ ============ ============ ============

.. figure:: /resource/image/baselines/SmallMaps/scripted_forage.png

   **Reproduction:** python Forge.py visualize --config=SmallMaps --MODEL=scripted_forage

Scripted Foraging + Combat
**************************

============ ============ ============ ============ ============
Metric       Min          Max          Mean         Std
============ ============ ============ ============ ============
Population          27.00        62.00        49.50         4.43
Lifetime             0.00       994.00        50.92        74.27
Skilling            10.00        53.00        15.04         5.54
Combat               3.00        33.00         4.35         2.77
Equipment            0.00        26.00         0.10         1.04
Exploration          0.00       101.00        14.94        10.80
============ ============ ============ ============ ============

.. figure:: /resource/image/baselines/SmallMaps/scripted_combat.png

   **Reproduction:** python Forge.py visualize --config=SmallMaps --MODEL=scripted_combat

|icon| Large Maps
#################

This setting includes 256 1024x1024 maps (1004x1004 excluding lava) and supports up to 1024 agents/1024 NPCs. It is intended for training and evaluation horizons of 6000-12000+ timesteps, or 1-2 hours in real time. Reasonable policies are trainable with 64 CPU cores and a single GPU in a few days. Evaluation takes several hours. The bounds of scaling with additional compute are unknown.

.. image:: /resource/image/large_isometric.png

Neural Baseline
***************

============ ============ ============ ============ ============
Metric       Min          Max          Mean         Std
============ ============ ============ ============ ============
Population           73.0        932.0        748.3        184.6
Lifetime              0.0      10000.0        251.4        893.2
Skilling             10.0         79.0         19.7         10.6
Combat                3.0         30.0          4.4          2.6
Equipment             0.0         15.0          0.0          0.2
Exploration           0.0        556.0         34.7         47.9
============ ============ ============ ============ ============

.. figure:: /resource/image/baselines/LargeMaps/neural.png

   **Reproduction:** python Forge.py visualize --config=LargeMaps --MODEL=large-baseline

Scripted Combat
*****************

   ============ ============ ============ ============ ============
   Metric       Min          Max          Mean         Std
   ============ ============ ============ ============ ============
   Population          55.00       648.00       548.28        58.87
   Lifetime             0.00      9996.00       194.07       587.94
   Skilling            10.00        76.00        20.46        10.07
   Combat               3.00        36.00         5.02         2.94
   Equipment            0.00        29.00         0.01         0.40
   Exploration          0.00       532.00        49.50        61.95
   ============ ============ ============ ============ ============

.. figure:: /resource/image/baselines/LargeMaps/scripted_combat.png

   **Reproduction:** python Forge.py visualize --config=LargeMaps --MODEL=scripted_combat
