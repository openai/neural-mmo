var slice = require('./slice')

module.exports = function curry(fn /* arg1, arg2, ... argN */) {
	var curryArgs = slice(arguments, 1)
	return function curried() {
		var invocationArgs = slice(arguments)
		return fn.apply(this, curryArgs.concat(invocationArgs))
	}
}

