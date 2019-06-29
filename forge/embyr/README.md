[ags]: resources/ags.png?raw=true
[fire]: resources/fire_thumbnail.png
[env]: resources/env.jpg

# ![][fire] Neural-MMO-Client
This repository contains the THREE.js based 3D browser client for the main [Neural MMO Project](https://docs.google.com/document/d/1_76rYTPtPysSh2_cFFz3Mfso-9VL3_tF5ziaIZ8qmS8/edit?usp=sharing). It's in Javascript, but it reads like Python. This is both because I am a Python-using researcher and because it allows researchers with under 30 minutes of Javascript experience to begin contributing immediately.

![][env]

## ![][ags] Setup

You don't need to clone this repo manually. Follow the install instructions in the [OpenAI Repo](https://github.com/openai/neural-mmo). This will download THREE.js. You can do this manually if you do not want to download the whole source repo.

## ![][ags] Performance

Around 50-60 FPS with ~3s load on a high-end desktop, 30 FPS with ~10s load on my Razer laptop.

## ![][ags] Details

I personally plan on continuing development on both the main environment and the client. The environment repo is quite clean, but this one could use some restructuring -- I intend to refactor it sometime soon. Environment updates will most likely be released in larger chunks, potentially coupled to future publications. On the other hand, the client is under active and rapid development. You can expect most features, at least in so far as they are applicable to the current environment build, to be released as soon as they are stable. Feel free to contact me with ideas and feature requests.

Please note: this is my personal agenda, and I do not speak for OpenAI.

## ![][ags] Known Limitations

The client has been tested with Firefox on Ubuntu. Don't use Chrome. It should work on other Linux distros and on Macs -- if you run into issues, let me know.

Use Nvidia drivers if your hardware setup allows. The only real requirement is support for more that 16 textures per shader. This is only required for the Counts visualizer -- you'll know your setup is wrong if the terrain map vanishes when switching overlays.

This is because the research overlays are written as raw glsl shaders, which you probably don't want to try to edit. In particular, the counts exploration visualizer hard codes eight textures corresponding to exploration maps. This exceeds the number of allowable textures. I will look into fixing this into future if there is significant demand. If you happen to be a shader wizard with spare time, feel free to submit a PR.

## ![][ags] Authorship

This client is a collaboration between myself (Joseph Suarez) and Clare Zhu. It was originally created as follow-up work for the paper and blog post, but we ended up merging it in. This is also the reason that the project is split into two repositories.

## ![][ags] License

MIT License
