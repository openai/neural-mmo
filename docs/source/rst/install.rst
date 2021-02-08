|icon| Installation
###################

.. code-block:: python
   :caption: Ubuntu 20.04/18.04 (requires Anaconda Python 3.8.x + gcc)

   #Download Neural MMO and run the pretrained demo model
   git clone --depth=1 https://github.com/jsuarez5341/neural-mmo && cd neural-mmo
   bash scripts/setup.sh
   python Forge.py render

   #Open the client in a separate terminal
   ./client.sh

.. code-block:: python
   :caption: WSl Ubuntu 20.04/18.04 (requires Anaconda Python 3.8.x + gcc) + Windows 10

   #Execute on WSL Ubuntu + Anaconda
   git clone --depth=1 https://github.com/jsuarez5341/neural-mmo && cd neural-mmo
   bash scripts/setup.sh --SERVER_ONLY
   python Forge.py render

   #Execute on Windows
   git clone --depth=1 https://github.com/jsuarez5341/neural-mmo-client
   neural-mmo-client/UnityClient/neural-mmo.exe

**Troubleshooting:**
  - Post installation errors in #support on the `[Discord] <https://discord.gg/BkMmFUC>`_
  - Most compatibility issues with the client and unsupported operating systems can be resolved by opening the project in the Unity Editor
  - If you want full commit history, clone without ``--depth=1`` (including in scripts/setup.sh for the client). This flag is only included to cut down on download time
  - The master branch will always contain the latest stable version. Each previous version release is archived in a separate branch. Dev branches are not nightly builds and may be flammable.
