module.exports = function keys(obj) {
	if (Object.keys) {
		module.exports = function(obj) {
			return Object.keys(obj)
		}
	} else {
		module.exports = function(obj) {
			for (var k in obj) {
				if (obj.hasOwnProperty(k)) { keys.push(k) }
			}
		}
	}
	return module.exports(obj)
}
