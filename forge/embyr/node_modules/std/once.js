module.exports = function once(fn) {
	var timeout
	var args
	return function() {
		args = arguments
		if (timeout) { return }
		timeout = setTimeout(function() {
			fn.apply(this, args)
			timeout = null
			args = null
		}, 0)
	}
}
