---
name: Enhancement
about: Suggest an improvement to an API or a refactorization of existing code for
  better efficiency or clarity
title: "[Enhancement]"
labels: ''
assignees: ''

---

This feature template is mostly used by the developers to track ongoing tasks, but users are also free to suggest additional enhancements or submit PRs solving existing ones. At the current scale, you should come chat with us on the Discord #development channel before writing one of these.

Try to match one of the templates below. If you can't, use the "other" template for now and we'll add a new template matching your issue afterwards.

**Dead code**: A piece of code is unused and should be deleted. The most common case for a dead code report occurs when we have replaced an older, clunkier routine but have neglected to delete the original. Check to make sure that you are not reporting a util function or paused-development feature before submitting.

**Confusing code**: A piece of code is difficult to parse and should be refactored or at least commented. These are subjective, but we take them seriously. Neural MMO is designed to be hackable -- the internals matter just as much as the user API.

**Poor performance**: A function or subroutine is slow. Describe cases in which this functionality becomes a bottleneck and submit timing data.
