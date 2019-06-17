module.exports = function options(opts, defaults) {
	if (!opts) { opts = {} }
	var result = {}
	if (!opts) { opts = {} }
	for (var key in defaults) {
		result[key] = typeof opts[key] != 'undefined' ? opts[key] : defaults[key]
	}
	return result
}
