var Class = require('./Class'),
	extend = require('./extend')

module.exports = Class(function() {

	var defaults = {
		duration:250,
		interval:40,
		tween:function linearTween(delta) { return delta }
	}
	
	this.init = function(animationFunction, opts) {
		this._animationFunction = animationFunction
		opts = extend(opts, defaults)
		this._duration = opts.duration
		this._interval = opts.interval
		this._tween = opts.tween
		this._onDone = opts.onDone
	}
	
	this.start = function(reverse) {
		this._playing = true
		this._startT = new Date().getTime()
		this._reverse = reverse
		this._onInterval()
		this._intervalID = setInterval(bind(this, this._onInterval), this._interval)
	}
	
	this.stop = function() {
		this._playing = false
		clearInterval(this._intervalID)
	}
	
	this.isGoing = function() { return this._playing }
	
	this._onInterval = function() {
		var deltaT = new Date().getTime() - this._startT,
			duration = this._duration
		if (deltaT >= duration) {
			this.stop()
			this._animationFunction(this._tween(this._reverse ? 0 : 1))
			if (this._onDone) { this._onDone() }
			return
		}
		var delta = deltaT / duration
		if (this._reverse) { delta = 1 - delta }
		this._animationFunction(this._tween(delta))
	}
})

// Easing equation function for elastic tween: http://code.google.com/p/kawanet/source/browse/lang/as3/KTween/trunk/src/net/kawa/tween/easing/Elastic.as
module.exports.elasticEaseOut = function(delta) {
	var x = 1 - delta,
		elasticity = 0.25
		value = 1 - Math.pow(x, 4) + x * x * Math.sin(delta * delta * Math.PI * elasticity)
	return value 
}
