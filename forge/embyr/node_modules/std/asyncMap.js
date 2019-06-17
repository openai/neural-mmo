var asyncEach = require('std/asyncEach')

module.exports = function asyncMap(items, opts) {
	var result = []
	result.length = items.length
	var includeNullValues = !opts.filterNulls
	var context = opts.context || this

	var originalIterate = asyncEach.makeIterator(context, opts.iterate)
	opts.iterate = function(value, index, next) {
		originalIterate(value, index, function(err, iterationResult) {
			if (err) { return next(err) }
			if (includeNullValues || (iterationResult != null)) {
				result[index] = iterationResult
			}
			next()
		})
	}

	var originalFinish = opts.finish
	opts.finish = function(err) {
		if (err) { return originalFinish.call(context, err) }
		originalFinish.call(context, null, result)
	}

	asyncEach(items, opts)
}
