/*
	Delay the execution of a function.
	If the function gets called multiple times during a delay, the delayed function gets invoced only once,
	with the arguments of the most recent invocation. This is useful for expensive functions that should
	not be called multiple times during a short time interval, e.g. rendering
	
	Example usage:

	Class(UIComponent, function() {
		this.render = delay(function() {
			...
		}, 250) // render at most 4 times per second
	})

	// Bath messages into a single email
	var EmailBatcher = Class(function() {
		this.init = function() {
			this._queue = []
		}

		this.send = function(email) {
			this._queue.push(email)
			this._scheduleDispatch()
		}

		this._scheduleDispatch = delay(function() {
			smtp.send(this._queue.join('\n\n'))
			this._queue = []
		}, 5000) // send emails at most once every 5 seconds
	})
*/
module.exports = function delay(fn, delayBy) {
	if (typeof delayBy != 'number') { delayBy = 50 }
	var timeoutName = '__delayTimeout__' + (++module.exports._unique)
	var delayedFunction = function delayed() {
		if (this[timeoutName]) {
			clearTimeout(this[timeoutName])
		}
		var args = arguments, self = this
		this[timeoutName] = setTimeout(function fireDelayed() {
			clearTimeout(self[timeoutName])
			delete self[timeoutName]
			fn.apply(self, args)
		}, delayBy)
	}
	delayedFunction.cancel = function() {
	  clearTimeout(this[timeoutName])
	}
	return delayedFunction
}
module.exports._unique = 0
