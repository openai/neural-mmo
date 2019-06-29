var fs = require('fs'),
	compiler = require('../compiler')

var file = __dirname + '/client.js',
	basePath = __dirname,
	code = fs.readFileSync(file).toString()

console.log(compiler.compileCode(code, { basePath:__dirname }))
