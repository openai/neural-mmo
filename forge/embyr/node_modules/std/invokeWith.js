/*
	Example usage:

  var callbacks = [...],
    result = { ... }
	each(callbacks, invokeWith(result))
*/

var slice = require('./slice')
module.exports = function invokeWith(/*, curry1, ..., curryN */) {
	var curryArgs = slice(arguments, 0)
	return function futureInvocation(fn) {
		var args = curryArgs.concat(slice(arguments, 1))
		return fn.apply(this, args)
	}
}

