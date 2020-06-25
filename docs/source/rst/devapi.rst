.. |icon| image:: /resource/icon/icon_pixel.png

|icon| Developer API
####################

The doctree below contains automatically generated documentation for the entire project. This is not intended for typical users but is a useful reference for Neural MMO developers and contributors. Neural MMO is fully open source and in no way limited to experienced reinforcment learning researchers. We also need full-stack software engineers and game designers/programmers/technical artists, among others (see the Developer Guide). A few tips and gotchas:

1. Before you implement anything, join the Discord. It is possible that the feature you'd like to build is already in development or that we have a subsystem you can work off of. 

2. Only functions with docstrings are displayed by default; always refer to the source for internal functions. At current scale, we focus on concise, self-documenting code outside of the User API. That said, the self that wrote the code is not always the best at determining what is self-documenting. Let me know if you find anything confusing -- it probably needs reworking.

3. Not all of the files below are in use. In particular, several game systems, such as items+inventory, were prototyped early on in development but have been put on hold in order to prioritize models/infrastructure. That said, if you are looking for environment-side features to work on, these are good candidates.

.. toctree::
   :maxdepth: 4

   projekt
   forge
