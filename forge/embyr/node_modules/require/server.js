var http = require('http')
var fs = require('fs')
var path = require('path')
var extend = require('std/extend')
var isObject = require('std/isObject')
var map = require('std/map')
var each = require('std/each')
var getDependencyLevels = require('./lib/getDependencyLevels')
var getRequireStatements = require('./lib/getRequireStatements')
var getCode = require('./lib/getCode')
var resolve = require('./lib/resolve')

module.exports = {
	listen: listen,
	mount: mount,
	connect: connect,
	isRequireRequest: isRequireRequest,
	handleRequest: handleRequest
}

function listen(portOrOpts) {
	var _opts = (isObject(portOrOpts) ? portOrOpts : { port:portOrOpts || 1234 })
	opts.handleAllRequests = true
	mount(http.createServer(), _opts).listen(opts.port, opts.host)
}

function mount(server, _opts) {
	setOpts(_opts)
	return server.on('request', _checkRequest)
}

function connect(opts) {
	setOpts(opts)
	return _checkRequest
}

function _checkRequest(req, res, next) {
	if (isRequireRequest(req) || opts.handleAllRequests) {
		handleRequest(req, res)
	} else {
		next && next()
	}
}

function isRequireRequest(req) {
	return req.url.substr(1, opts.root.length) == opts.root
}

/* options
 *********/
var opts = {
	path: process.cwd(),
	root: 'require',
	port: null,
	host: null
}
function setOpts(_opts) {
	opts = extend(_opts, opts)
	if (opts.path) {
		resolve._nodePaths.push(opts.path)
	}
}
function getUrlBase() {
	var basePort = (!opts.usePagePort && opts.port)
	if (opts.host && basePort) {
		return '//' + opts.host + ':' + basePort + '/' + opts.root + '/'
	} else {
		return '/' + opts.root + '/'
	}
}

/* request handlers
 ******************/
function handleRequest(req, res) {
	var reqPath = _normalizeURL(req.url).substr(opts.root.length + 2)
	if (reqPath.match(/\.js$/)) {
		_handleModuleRequest(reqPath, res)
	} else {
		_handleMainModuleRequest(reqPath, req, res)
	}

	function _normalizeURL(url) {
		return url.replace(/\?.*/g, '').replace(/\/js$/, '.js')
	}
}

function _handleMainModuleRequest(reqPath, req, res) {
	var mainModulePath = resolve.path('./' + reqPath, opts.path)
	if (!mainModulePath) { return _sendError(res, 'Could not find module "'+reqPath+'" from "'+opts.path+'"') }

	try { var dependencyTree = getDependencyLevels(mainModulePath) }
	catch(err) { return _sendError(res, 'in getDependencyLevels: ' + (err.message || err)) }
	
	var userAgent = req.headers['user-agent']
	var isMobile = userAgent.match('iPad') || userAgent.match('iPod') || userAgent.match('iPhone') || userAgent.match('Android')
	
	var response = isMobile ? _getMobilePayload() : _getNormalPayload()
	
	res.writeHead(200, { 'Cache-Control':'no-cache', 'Expires':'Fri, 31 Dec 1998 12:00:00 GMT', 'Content-Length':response.length, 'Content-Type':'text/javascript' })
	res.end(response)

	function _getMobilePayload() {
		var result = ['__require__={loadNextModule:function(){},onModuleLoaded:function(){}}']
		each(dependencyTree, function(level) {
			each(level, function(dependency) {
				result.push(';(function(){ '+_getModuleCode(res, dependency)+' }());')
			})
		})
		return new Buffer(result.join('\n'))
	}
	
	function _getNormalPayload() {
		var paramsString = map([getUrlBase(), dependencyTree], JSON.stringify).join(',\n\t\t')
		return new Buffer('\t('+clientBootstrapFn.toString()+')(\n\t\t'+paramsString+'\n\t)')	
		
		function clientBootstrapFn(urlBase, levels) {
			// This function gets sent to the client as toString
			__require__ = {
				loadNextLevel: loadNextLevel,
				onModuleLoaded: onModuleLoaded
			}

			var currentLevel = null
			loadNextLevel()

			function loadNextLevel() {
				if (!levels.length) { return } // all done!
				currentLevel = levels.shift()
				var head = document.getElementsByTagName('head')[0]
				for (var i=0; i<currentLevel.length; i++) {
					// var url = location.protocol + '//' + location.host + urlBase + currentLevel[i]
					var url = urlBase + currentLevel[i]
					head.appendChild(document.createElement('script')).src = url
				}
			}

			function onModuleLoaded() {
				currentLevel.pop()
				if (currentLevel.length == 0) {
					loadNextLevel()
				}
			}
		}		
	}
}

function _asString(fn) { return fn.toString() }

function _handleModuleRequest(reqPath, res) {
	try { var code = _getModuleCode(res, reqPath) }
	catch(err) { return _sendError(res, err.stack || err) }

	code += '\n\n'
	
	var buf = new Buffer(code)
	res.writeHead(200, { 'Cache-Control':'no-cache', 'Expires':'Fri, 31 Dec 1998 12:00:00 GMT', 'Content-Length':buf.length, 'Content-Type':'text/javascript' })
	res.end(buf)
}

function _getModuleCode(res, reqPath) {
	var code = getCode(reqPath)
	var requireStatements = getRequireStatements(code)

	try {
		each(requireStatements, function(requireStmnt) {
			var depPath = resolve.requireStatement(requireStmnt, reqPath)
			if (!depPath) { throw 'Could not resolve module' }
			code = code.replace(requireStmnt, '__require__["'+depPath+'"]')
		})
	} catch(e) {
		_sendError(res, e.message || e)
	}

	var _closureStart = ';(function(){'
	var _moduleDef = 'var module={exports:{}},exports=module.exports;/*FILE BEGIN*/ '
	var _closureEnd = '/*FILE END*/__require__["'+reqPath+'"]=module.exports; __require__.onModuleLoaded()\n})()'
	return _closureStart + _moduleDef + code + // all on the first line to make error line number reports correct
		'\n' + _closureEnd
}

function _sendError(res, msg) {
	if (msg) { msg = msg.replace(/\n/g, '\\n').replace(/"/g, '\\"') }
	res.writeHead(200)
	res.end('alert("error: ' + msg + '")')
}
