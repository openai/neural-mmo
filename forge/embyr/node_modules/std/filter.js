/*
	Example usage:
	filter([1,2,0,'',false,null,undefined]) // -> [1,2,0,'',false]
	filter([1,2,3], this, function(val, index) { val == 1 }) // -> [1]
*/
var each = require('./each')
var isArray = require('./isArray')

module.exports = function filter(arr, ctx, fn) {
	if (arguments.length == 2) {
		fn = ctx
		ctx = this
	}
	if (!fn) {
		fn = falseOrTruthy
	}
	
	var result
	if (isArray(arr)) {
		result = []
		each(arr, function(value, index) {
			if (!fn.call(ctx, value, index)) { return }
			result.push(value)
		})
	} else {
		result = {}
		each(arr, function(value, key) {
			if (!fn.call(ctx, value, key)) { return }
			result[key] = value
		})
	}
	return result
}

function falseOrTruthy(arg) {
	return !!arg || arg === false
}

