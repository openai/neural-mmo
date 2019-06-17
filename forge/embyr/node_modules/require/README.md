require brings `require` to the browser
=======================================

Node's `require()` is the de facto javascript dependency statement.

`npm` is the de facto javascript module manager.

require brings both of them to the browser.

tl;dr
=====
"Just give me some code that runs"

	mkdir app; cd app
	echo '{ "name":"app" }' > package.json
	sudo npm install require
	sudo npm install raphael
	curl -O https://raw.github.com/gist/975866/little_server.js
	curl -O https://raw.github.com/gist/975868/little_client.js
	node little_server.js

Now go to http://localhost:8080

Install
=======

	sudo npm install -g require

Run
===
Start dev server

	require serve ./example --port 1234 --host localhost

In your HTML:

	<script src="//localhost:1234/require/client"></script>

This is like calling require('client') from inside ./example.
[Read more on node's require path resolution](http://nodejs.org/api/modules.html)

Compile
=======
(You'll want to do this before you deploy to production)

	require compile ./example/client.js > client.min.js

Use programmatically
====================
In node:

	require('require/server').listen(1234)

or mount it on an http server you're already running

	var server = http.createServer(function(req, res) { })
	require('require/server').mount(server)
	server.listen(8080, 'localhost')

or, as connect middleware

	connect.createServer(
		connect.static(__dirname + '/example'),
		require('require/server').connect()
	)

Compile programmatically:

	var compiler = require('require/compiler')
	console.log(compiler.compile('./example/client.js'))
	console.log(compiler.compileCode('require("./example/client")'))

The compiler supports all the options of https://github.com/mishoo/UglifyJS, e.g.

	compiler.compile('./example/client.js', { beautify:true, ascii_only:true })
