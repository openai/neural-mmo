var raphael = require('raphael'),
	canvas = document.body.appendChild(document.createElement('div')),
	paper = raphael(canvas)

paper.circle(50, 50, 40)
