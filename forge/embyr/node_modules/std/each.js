var isArray = require('./isArray'),
	isArguments = require('./isArguments')

module.exports = function(items, ctx, fn) {
	if (!items) { return }
	if (!fn) {
		fn = ctx
		ctx = this
	}
	if (isArray(items) || isArguments(items)) {
		for (var i=0; i < items.length; i++) {
			fn.call(ctx, items[i], i)
		}
	} else {
		for (var key in items) {
			if (!items.hasOwnProperty(key)) { continue }
			fn.call(ctx, items[key], key)
		}
	}
}
