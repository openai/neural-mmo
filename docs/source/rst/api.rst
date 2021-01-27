.. |env| image:: /resource/image/v1-4_splash.png
.. |icon| image:: /resource/icon/icon_pixel.png

|env|

|icon| User API
###############

Neural MMO's core Env API is a simple multiagent analog to the standard OpenAI Gym API. It also includes a configuration file for customizing everything from terrain generation to spawning behavior to game mechanics

.. toctree::
   :maxdepth: 4

   forge.trinity.env
   forge.blade.core.config

This is all that most environments provide. However, Neural MMO is not a single-task environment: it is a platform built to support a wide diversity of research. The current release includes a full evaluation and visualization suite for creating custom dashboards and in-game overlays. If you wish to make use of these custom features, familiarize yourself with the /projekt demo code and refer to these additional docs:

.. toctree::
   :maxdepth: 4

   forge.trinity.evaluator
   forge.trinity.overlay
   forge.trinity.formatting

|icon| Developer API
####################

The doctree below contains automatically generated documentation for the entire project. This is not intended for typical users but is a useful reference for Neural MMO developers and contributors.

.. toctree::
   :maxdepth: 4

   ../autodoc/projekt
   ../autodoc/forge
