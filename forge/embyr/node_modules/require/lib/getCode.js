var fs = require('fs')

module.exports = function readCode(filePath) {
	if (!filePath.match(/\.js$/)) { filePath += '.js' }
	return fs.readFileSync(filePath).toString()
}