.. |icon| image:: /resource/icon/icon_pixel.png

.. image:: /resource/image/v1-4_splash.png

|icon| Evaluation Protocol
##########################

Neural MMO provides two standard training and evaluation settings along with pretrained and scripted baselines. Training is performed on a pool of N_TRAIN_MAPS procedurally generated maps simulated for TRAIN_HORIZON timesteps per episode. Evaluation is performed on EVAL_MAPS maps never seen during training (GENERALIZE=False) for EVALUATION_HORIZON timesteps. Approaches that modify these parameters, alter the reward function, or substantially modify the environment configuration are not directly comparable to the baselines below.

Publications including new models should clearly indicate training scale, domain knowledge assumptions, and any other significant changes to the default configuration. Ideally, they should also include a discussion of agent behavior and supporting training/evaluation plots, overlay figures, and/or other evidence specific to the associated work.

To reproduce the tables and figures below, run the associated command. Run *train* or *evaluate* instead of *visualize* to verify results from scratch or from the pretrained model.

|icon| Small Maps
#################

This setting includes 256 80x80 maps (60x60 excluding lava) and supports up to 128 agents/32 NPCs. It is intended for training and evaluation horizons of 1000 timesteps, or 10 minutes in real time. Reasonable policies are trainable with as low as 4 CPU cores and a single modern GPU in a few hours. Evaluation takes only a few minutes. Additional skills may emerge with increased scale.

.. image:: /resource/image/v1-5_small_isometric.png

Neural Baseline
***************

.. image:: /resource/image/baselines/SmallMaps/neural_small_maps.png

============ ============ ============ ============ ============
Metric       Min          Max          Mean         Std
============ ============ ============ ============ ============
Population           21.0         69.0         53.0          5.2
Lifetime              0.0        999.0         52.9        100.3
Skilling             10.0         50.0         14.8          5.9
Combat                3.0         23.0          4.8          2.5
Equipment             0.0         16.0          0.2          1.0
Exploration           0.0         93.0         10.3          7.6
============ ============ ============ ============ ============

**Reproduction:** python Forge.py visualize --config=SmallMaps --MODEL=small-baseline

Scripted Foraging
*****************

.. image:: /resource/image/baselines/SmallMaps/scripted_forage.png

============ ============ ============ ============ ============
Metric       Min          Max          Mean         Std
============ ============ ============ ============ ============
Population           35.0        104.0         83.0         10.4
Lifetime              0.0       1000.0         83.7        131.8
Skilling             10.0         48.5         16.6          6.8
Combat                3.0          4.0          3.1          0.3
Equipment             0.0          0.0          0.0          0.0
Exploration           0.0        111.0         18.1         13.4
============ ============ ============ ============ ============

**Reproduction:** python Forge.py visualize --config=SmallMaps --MODEL=scripted_forage

Scripted Foraging + Combat
**************************

.. image:: /resource/image/baselines/SmallMaps/scripted_combat.png

============ ============ ============ ============ ============
Metric       Min          Max          Mean         Std
============ ============ ============ ============ ============
Population           21.0         67.0         50.7          4.9
Lifetime              0.0        996.0         52.1         77.4
Skilling             10.0         52.5         14.9          5.6
Combat                3.0         33.0          4.4          2.8
Equipment             0.0         23.0          0.1          1.2
Exploration           0.0        105.0         15.1         11.0
============ ============ ============ ============ ============

**Reproduction:** python Forge.py visualize --config=SmallMaps --MODEL=scripted_combat

|icon| Large Maps
#################

This setting includes 256 1024x1024 maps (1004x1004 excluding lava) and supports up to 1024 agents/1024 NPCs. It is intended for training and evaluation horizons of 6000-12000+ timesteps, or 1-2 hours in real time. Reasonable policies are trainable with 64 CPU cores and a single GPU in a few days. Evaluation takes several hours. The bounds of scaling with additional compute are unknown.

.. image:: /resource/image/v1-5_large_isometric.png

Neural Baseline
***************

.. image:: /resource/image/baselines/LargeMaps/neural.png

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

**Reproduction:** python Forge.py visualize --config=LargeMaps --MODEL=large-baseline
