module.exports = function delayed(amount, fn) {
	if (!fn) {
		fn = amount
		amount = 0
	}
	return function() {
		var self = this
		var args = arguments
		setTimeout(function() { fn.apply(self, args) }, amount)
	}
}
