var isArray = require('./isArray')

module.exports = function(arr) {
	if (!isArray(arr)) { return null }
	return arr[arr.length - 1]
}