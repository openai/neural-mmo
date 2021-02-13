.. |icon| image:: /resource/icon/icon_pixel.png

.. figure:: /resource/image/environment.png

   Neural MMO provides two standard training and evaluation settings along with pretrained and scripted baselines. You can learn more about them below.

|icon| Evaluation Protocol
##########################

Training is performed on a pool of N_TRAIN_MAPS procedurally generated maps simulated for TRAIN_HORIZON timesteps per episode. Evaluation is performed on EVAL_MAPS maps never seen during training (GENERALIZE=False) for EVALUATION_HORIZON timesteps. Approaches that modify these parameters, alter the reward function, or substantially modify the environment configuration are not directly comparable to the baselines below.

Publications including new models should clearly indicate training scale, domain knowledge assumptions, and any other significant changes to the default configuration. Ideally, they should also include a discussion of agent behavior and supporting training/evaluation plots, overlay figures, and/or other evidence specific to the associated work.

To reproduce the tables and figures below, run the associated command. Run *train* or *evaluate* instead of *visualize* to verify results from scratch or from the pretrained model.

|icon| Small Maps
#################

This setting includes 256 80x80 maps (60x60 excluding lava) and supports up to 128 agents/32 NPCs. It is intended for training and evaluation horizons of 1000 timesteps, or 10 minutes in real time. Evaluation takes only a few minutes.

.. image:: /resource/image/small_isometric.png

Neural Baseline
***************

Our baseline is trained using an RTX 3080 + 64 cores for ~(100k maps/1000 epochs/4 days). The first skills to emerge are survival/foraging/exploration, followed by combat, followed much later by attacking NPCs for equipment. Training for an extra ~500 epochs did not result in any substantial improvements.

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

.. figure:: /resource/image/baselines/SmallMaps/small-map.png

   **Reproduction:** python Forge.py visualize --config=SmallMaps --MODEL=small-map

Untrained
*********

A randomly initialized copy of the trained model architecture

============ ============ ============ ============ ============
Metric       Min          Max          Mean         Std
============ ============ ============ ============ ============
Population           1.00        17.00         8.29         2.37
Lifetime             0.00       480.00         8.86        14.85
Skilling            10.00        36.50        10.41         1.19
Combat               3.00         5.00         3.87         0.34
Equipment            0.00         4.00         0.00         0.07
Exploration          0.00        26.00         2.22         2.72
============ ============ ============ ============ ============

.. figure:: /resource/image/baselines/SmallMaps/None.png

   **Reproduction:** python Forge.py visualize --config=SmallMaps --MODEL=None


Zero-Shot Transfer
******************

Evaluation of the large-map model on the SmallMaps domain

============ ============ ============ ============ ============
Metric       Min          Max          Mean         Std
============ ============ ============ ============ ============
Population           4.00        69.00        50.60         5.90
Lifetime             0.00       999.00        50.90        85.25
Skilling            10.00        48.50        14.43         6.24
Combat               3.00        12.00         3.93         0.99
Equipment            0.00         6.00         0.02         0.24
Exploration          0.00       102.00        11.59        13.54
============ ============ ============ ============ ============

.. figure:: /resource/image/baselines/SmallMaps/large-map.png

   **Reproduction:** python Forge.py visualize --config=SmallMaps --MODEL=large-map

Scripted Foraging
*****************

A scripted model that forages for food and only fights back when attacked by NPCs.

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

.. figure:: /resource/image/baselines/SmallMaps/scripted-forage.png

   **Reproduction:** python Forge.py visualize --config=SmallMaps --MODEL=scripted-forage

Scripted Foraging + Combat
**************************

A scripted model that forages for food and actively fights other scripted agents and NPCs

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

.. figure:: /resource/image/baselines/SmallMaps/scripted-combat.png

   **Reproduction:** python Forge.py visualize --config=SmallMaps --MODEL=scripted-combat

|icon| Large Maps
#################

This setting includes 256 1024x1024 maps (1004x1004 excluding lava) and supports up to 1024 agents/1024 NPCs. It is intended for training and evaluation horizons of 6000-12000+ timesteps, or 1-2 hours in real time. Evaluation takes several hours. Reasonable policies are trainable with 4 CPU cores and a single GPU in a few days. Our baseline is trained using an RTX 3080 + 16 cores for ~5 days. The bounds of scaling with additional compute are unknown -- note that this setting is typically RAM bound.

.. image:: /resource/image/large_isometric.png

Neural Baseline
***************

Our baseline is trained using an RTX 3080 + 16 cores for ~(7.5k maps/100 epochs/1 day).  The first skill to emerge is capable foraging and exploration, followed by combat and some equipment farming. Training for an extra ~740 epochs results in improved combat and equipment farming but greatly decreased lifetime/foraging/exploration.

============ ============ ============ ============ ============
Metric       Min          Max          Mean         Std
============ ============ ============ ============ ============
Population          56.00       904.00       680.67       170.47
Lifetime             0.00     10000.00       232.52       833.39
Skilling            10.00        78.50        19.51        10.30
Combat               3.00        27.00         4.28         2.29
Equipment            0.00        15.00         0.00         0.13
Exploration          0.00       580.00        30.50        44.57
============ ============ ============ ============ ============

.. figure:: /resource/image/baselines/LargeMaps/large-map.png

   **Reproduction:** python Forge.py visualize --config=LargeMaps --MODEL=large-map

Untrained
*********

A randomly initialized copy of the trained model architecture

============ ============ ============ ============ ============
Metric       Min          Max          Mean         Std
============ ============ ============ ============ ============
Population          60.00       103.00        84.49         4.44
Lifetime             0.00       441.00        32.74        11.11
Skilling            10.00        37.00        10.90         1.58
Combat               3.00         7.00         3.02         0.16
Equipment            0.00         5.00         0.00         0.02
Exploration          0.00        42.00         6.25         2.80
============ ============ ============ ============ ============

.. figure:: /resource/image/baselines/LargeMaps/None.png

   **Reproduction:** python Forge.py visualize --config=LargeMaps --MODEL=None

Zero-Shot Transfer
******************

Evaluation of the small-map model on the LargeMaps domain

============ ============ ============ ============ ============
Metric       Min          Max          Mean         Std
============ ============ ============ ============ ============
Population          41.00       260.00       197.00        33.20
Lifetime             0.00     10000.00        75.53       412.76
Skilling            10.00        80.00        13.57         6.37
Combat               3.00        20.00         4.81         1.90
Equipment            0.00        22.00         0.01         0.33
Exploration          0.00       431.00         8.98        10.13
============ ============ ============ ============ ============

.. figure:: /resource/image/baselines/LargeMaps/small-map.png

   **Reproduction:** python Forge.py visualize --config=LargeMaps --MODEL=small-map

Scripted Foraging
*****************

A scripted model that forages for food and only fights back when attacked by NPCs.

============ ============ ============ ============ ============
Metric       Min          Max          Mean         Std
============ ============ ============ ============ ============
Population          95.00      1024.00       989.19       128.00
Lifetime             0.00      9995.00       995.88       994.33
Skilling            10.00        76.00        39.84        14.51
Combat               3.00         3.00         3.00         0.00
Equipment            0.00         0.00         0.00         0.00
Exploration          0.00       570.00       222.63       144.32
============ ============ ============ ============ ============

.. figure:: /resource/image/baselines/LargeMaps/scripted-forage.png

    **Reproduction:** python Forge.py visualize --config=LargeMaps --MODEL=scripted-forage


Scripted Combat
***************

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

.. figure:: /resource/image/baselines/LargeMaps/scripted-combat.png

   **Reproduction:** python Forge.py visualize --config=LargeMaps --MODEL=scripted-combat
