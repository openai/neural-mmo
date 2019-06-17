var unique = 0
module.exports = function throttle(fn, delay) {
	if (typeof delay != 'number') { delay = 50 }
	var timeoutName = '__throttleTimeout__' + (++unique)
	return function throttled() {
		if (this[timeoutName]) { return }
		var args = arguments, self = this
		this[timeoutName] = setTimeout(function fireDelayed() {
			delete self[timeoutName]
			fn.apply(self, args)
		}, delay)
		fn.apply(self, args)
	}
}
