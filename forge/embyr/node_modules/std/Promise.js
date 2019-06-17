var Class = require('./Class'),
	invokeWith = require('./invokeWith'),
	slice = require('./slice'),
	each = require('./each'),
	bind = require('./bind')

module.exports = Class(function() {
	this.init = function(callback) {
		this._dependants = []
		this._fulfillment = null
		if (callback) { this.add(callback) }
	}

	this.add = function(callback) {
		if (this._fulfillment) { callback.apply(this, this._fulfillment) }
		else { this._dependants.push(callback) }
		return this
	}

	this.fulfill = function(/* arg1, arg2, ...*/) {
		if (this._fulfillment) { throw new Error('Promise fulfilled twice') }
		this._fulfillment = slice(arguments)
		each(this._dependants, invokeWith.apply(this, this._fulfillment))
		delete this._dependants
		return this
	}
	
	this.nextTickAdd = function(callback) {
		setTimeout(bind(this, this.add, callback), 0)
	}
	
	this.getCallback = function() {
		return this._callback || (this._callback = bind(this, this.handle))
	}
	
	this.handle = function(err, result) {
		if (err) { this.fail(err) }
		else { this.fulfill(result) }
	}
})
