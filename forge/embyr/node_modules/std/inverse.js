var each = require('std/each')

module.exports = function inverse(obj) {
	var result = {}
	each(obj, function(val, key) { result[val] = key })
	return result
}
