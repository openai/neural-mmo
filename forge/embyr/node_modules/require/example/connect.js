var connect = require('connect'),
	requireServer = require('require/server')

connect.createServer(
	connect.static(__dirname),
	requireServer.connect(__dirname)
).listen(8080)
