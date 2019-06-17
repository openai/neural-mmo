/*
	Example usage:

	var obj = {
		setTime: function(ts) { this._time = ts }
	}
	each([obj], call('setTime', 1000))
*/

var slice = require('./slice')
module.exports = function call(methodName /*, curry1, ..., curryN */) {
	var curryArgs = slice(arguments, 1)
	return function futureCall(obj) {
		var fn = obj[methodName],
			args = curryArgs.concat(slice(arguments, 1))
		return fn.apply(obj, args)
	}
}

