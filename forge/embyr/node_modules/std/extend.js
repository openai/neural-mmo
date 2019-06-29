/*
	Example usage:

	var A = Class(function() {
		
		var defaults = {
			foo: 'cat',
			bar: 'dum'
		}

		this.init = function(opts) {
			opts = std.extend(opts, defaults)
			this._foo = opts.foo
			this._bar = opts.bar
		}

		this.getFoo = function() {
			return this._foo
		}

		this.getBar = function() {
			return this._bar
		}
	})

	var a = new A({ bar:'sim' })
	a.getFoo() == 'cat'
	a.getBar() == 'sim'
*/

var copy = require('./copy')

module.exports = function extend(target, extendWith) {
	target = copy(target)
	for (var key in extendWith) {
		if (typeof target[key] != 'undefined') { continue }
		target[key] = extendWith[key]
	}
	return target
}
