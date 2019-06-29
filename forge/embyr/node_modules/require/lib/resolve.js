var path = require('path')
var fs = require('fs')
var existsSync = fs.existsSync || path.existsSync

var _nodePaths = (process.env.NODE_PATH ? process.env.NODE_PATH.split(':') : [])
_nodePaths.push(process.cwd())

module.exports = {
	path: resolvePath,
	_nodePaths:_nodePaths,
	requireStatement: resolveRequireStatement
}

function resolvePath(searchPath, pathBase) {
	if (searchPath[0] == '.') {
		// relative path, e.g. require("./foo")
		return _findModuleMain(path.resolve(pathBase, searchPath))
	}
	
	var searchParts = searchPath.split('/')
	var componentName = searchParts[searchParts.length - 1]
	var name = searchParts.shift()
	var rest = searchParts.join('/')
	
	// npm-style path, e.g. require("npm").
	// Climb parent directories in search for "node_modules"
	var modulePath = _findModuleMain(path.resolve(pathBase, 'node_modules', searchPath))
	if (modulePath) { return modulePath }

	if (pathBase != '/') {
		// not yet at the root - keep climbing!
		return resolvePath(searchPath, path.resolve(pathBase, '..'))
	}
	
	return ''
}

var _pathnameGroupingRegex = /require\s*\(['"]([\w\/\.-]*)['"]\)/
function resolveRequireStatement(requireStmnt, currentPath) {
	var rawPath = requireStmnt.match(_pathnameGroupingRegex)[1]
	var resolvedPath = resolvePath(rawPath, path.dirname(currentPath))
	
	if (!resolvedPath && rawPath[0] != '.' && rawPath[0] != '/') {
		for (var i=0; i<_nodePaths.length; i++) {
			resolvedPath = _findModuleMain(path.resolve(_nodePaths[i], rawPath))
			if (resolvedPath) { break }
		}
	}
	
	if (!resolvedPath) { throw 'Could not resolve "'+rawPath+'" in "'+currentPath+'"' }
	return resolvedPath
}

function _findModuleMain(absModulePath, tryFileName) {
	var foundPath = ''
	function attempt(aPath) {
		if (foundPath) { return }
		if (existsSync(aPath)) { foundPath = aPath }
	}
	attempt(absModulePath + '.js')
	try {
		var package = JSON.parse(fs.readFileSync(absModulePath + '/package.json').toString())
		attempt(path.resolve(absModulePath, package.main+'.js'))
		attempt(path.resolve(absModulePath, package.main))
	} catch(e) {}
	attempt(absModulePath + '/index.js')

	if (tryFileName) { attempt(absModulePath + '/' + tryFileName + '.js') }
	return foundPath
}


