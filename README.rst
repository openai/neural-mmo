
.. |icon| image:: docs/source/resource/icon/icon_pixel.png

.. figure:: /resource/image/splash.png

|icon| Welcome to the Platform!
###############################

`[Demo Video] <https://youtu.be/y_f77u9vlLQ>`_ | `[Discord] <https://discord.gg/BkMmFUC>`_ | `[Twitter] <https://twitter.com/jsuarez5341>`_

Neural MMO is a massively multiagent AI research environment inspired by Massively Multiplayer Online (MMO) role playing games. The project is under active development with major updates every 3-6 months. This README is a stub -- all of our `[Documentation] <https://jsuarez5341.github.io>`_ is hosted by github.io.

Beta Tester Instructions
************************

Build docs locally (requires Make):

```
cd docs
bash make.sh
bash view.sh
```

This build docs and open a local view in a browser. The installation instructions given assume that you want to use the master branch, which will not be the case until v1.5 goes live officially. Use the modified instructions below to install the v1.5 prerelease/dev branches. You will need Anaconda python 3.8 + gcc before starting. If you are on WSL, run the client setup from Windows and the main repo setup from Ubuntu. If you are on a Mac, follow the Ubuntu instructions and let me know if they work. If you do not have CUDA, use --MODEL=scripted-combat in the tutorials.

```
#Server -- if you can't run python Forge.py --help after install, rerun pip install ray[rllib]
git clone https://github.com/jsuarez5341/neural-mmo && cd neural-mmo
git checkout --track origin/v1.5-prerelease
bash scripts/setup.sh --SERVER_ONLY

#Client
git clone https://github.com/jsuarez5341/neural-mmo-client && mv neural-mmo-client forge/embyr && cd forge/embyr
git checkout --track origin/v1.5-cowboy-dev && cd ../..
```
