var stripRegex = /^\s*(.*?)\s*$/
module.exports = function(str) {
	return str.match(stripRegex)[1]
}

