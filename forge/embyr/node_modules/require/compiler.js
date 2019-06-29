var fs = require('fs')
var path = require('path')
var extend = require('std/extend')
var each = require('std/each')
var getCode = require('./lib/getCode')
var resolve = require('./lib/resolve')
var getRequireStatements = require('./lib/getRequireStatements')

module.exports = {
	compile: compileFile,
	compileHTML: compileHTMLFile,
	compileCode: compileCode
}

/* api
 *****/
function compileFile(filePath, opts) {
	filePath = path.resolve(filePath)
	opts = extend(opts, { basePath:path.dirname(filePath), toplevel:true })
	var code = getCode(filePath)
	return _compile(code, opts, filePath)
}

function compileCode(code, opts) {
	opts = extend(opts, { basePath:process.cwd(), toplevel:true })
	return _compile(code, opts, '<code passed into compiler.compile()>')
}

function compileHTMLFile(filePath, opts) {
	var html = fs.readFileSync(filePath).toString()
	while (match = html.match(/<script src="\/require\/([\/\w\.]+)"><\/script>/)) {
		var js = compileFile(match[1].toString(), opts)
		
		var BACKREFERENCE_WORKAROUND = '____________backreference_workaround________'
		js = js.replace('\$\&', BACKREFERENCE_WORKAROUND)
		html = html.replace(match[0], '<script>'+js+'</script>')
		html = html.replace(BACKREFERENCE_WORKAROUND, '\$\&')
	}
	return html
}


var _compile = function(code, opts, mainModule) {
	var code = 'var __require__ = {}, require=function(){}\n' + _compileModule(code, opts.basePath, mainModule)
	if (opts.minify === false) { return code } // TODO use uglifyjs' beautifier?

	var UglifyJS = require('uglify-js')
	var result = UglifyJS.minify(code, {
		fromString:true,
		mangle:true,
		output: {
			// http://lisperator.net/uglifyjs/codegen
			indent_start  : 0,     // start indentation on every line (only when `beautify`)
			indent_level  : 4,     // indentation level (only when `beautify`)
			quote_keys    : false, // quote all keys in object literals?
			space_colon   : true,  // add a space after colon signs?
			ascii_only    : false, // output ASCII-safe? (encodes Unicode characters as ASCII)
			inline_script : false, // escape "</script"?
			width         : 80,    // informative maximum line width (for beautified output)
			max_line_len  : 200, // maximum line length (for non-beautified output)
			ie_proof      : true,  // output IE-safe code?
			beautify      : false, // beautify output?
			source_map    : null,  // output a source map
			bracketize    : false, // use brackets every time?
			comments      : false, // output comments?
			semicolons    : true  // use semicolons to separate statements? (otherwise, newlines)
		},
		compress: {
			// http://lisperator.net/uglifyjs/compress
			sequences     : true,  // join consecutive statemets with the “comma operator”
			properties    : true,  // optimize property access: a["foo"] → a.foo
			dead_code     : true,  // discard unreachable code
			drop_debugger : true,  // discard “debugger” statements
			unsafe        : false, // some unsafe optimizations (see below)
			conditionals  : true,  // optimize if-s and conditional expressions
			comparisons   : true,  // optimize comparisons
			evaluate      : true,  // evaluate constant expressions
			booleans      : true,  // optimize boolean expressions
			loops         : true,  // optimize loops
			unused        : true,  // drop unused variables/functions
			hoist_funs    : true,  // hoist function declarations
			hoist_vars    : false, // hoist variable declarations
			if_return     : true,  // optimize if-s followed by return/continue
			join_vars     : true,  // join var declarations
			cascade       : true,  // try to cascade `right` into `left` in sequences
			side_effects  : true,  // drop side-effect-free statements
			warnings      : false, // warn about potentially dangerous optimizations/code
			global_defs   : {}     // global definitions
		}
	})
	// also see result.map
	return result.code
}

/* util
 ******/
var _compileModule = function(code, pathBase, mainModule) {
	var modules = [mainModule]
	_replaceRequireStatements(mainModule, code, modules, pathBase)
	code = _concatModules(modules)
	code = _minifyRequireStatements(code, modules)
	return code
}

var _minifyRequireStatements = function(code, modules) {
	each(modules, function(modulePath, i) {
		var escapedPath = modulePath.replace(/\//g, '\\/').replace('(','\\(').replace(')','\\)')
		var regex = new RegExp('__require__\\["'+ escapedPath +'"\\]', 'g')
		
		code = code.replace(regex, '__require__["_'+ i +'"]')
	})
	return code
}

var _replaceRequireStatements = function(modulePath, code, modules, pathBase) {
	var requireStatements = getRequireStatements(code)

	if (!requireStatements.length) {
		modules[modulePath] = code
		return
	}

	each(requireStatements, function(requireStatement) {
		var subModulePath = resolve.requireStatement(requireStatement, modulePath)

		if (!subModulePath) {
			throw new Error("Require Compiler Error: Cannot find module '"+ rawModulePath +"' (in '"+ modulePath +"')")
		}

		code = code.replace(requireStatement, '__require__["' + subModulePath + '"].exports')
		
		if (!modules[subModulePath]) {
			modules[subModulePath] = true
			var newPathBase = path.dirname(subModulePath)
			var newModuleCode = getCode(subModulePath)
			_replaceRequireStatements(subModulePath, newModuleCode, modules, newPathBase)
			modules.push(subModulePath)
		}
	})

	modules[modulePath] = code
}

var _concatModules = function(modules) {
	var getClosuredModule = function(modulePath) {
		return [
			';(function() {',
			'	// ' + modulePath,
			'	var module = __require__["'+modulePath+'"] = {exports:{}}, exports = module.exports;',
			modules[modulePath],
			'})()'
		].join('\n')
	}

	var moduleDefinitions = []
	for (var i=1, modulePath; modulePath = modules[i]; i++) {
		moduleDefinitions.push(getClosuredModule(modulePath))
	}
	moduleDefinitions.push(getClosuredModule(modules[0])) // __main__

	return moduleDefinitions.join('\n\n')
}
