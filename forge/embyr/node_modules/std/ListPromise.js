var proto = require('./proto'),
	bind = require('./bind')

module.exports = proto(null, 
	function(waitForNum, callback) {
		this._waitingFor = waitForNum
		this._callbacks = []
		this._results = []
		this._error = null
		if (callback) { this.add(callback) }
	}, {
		add: function(callback) {
			if (!this._waitingFor) {
				this._notify(callback)
			} else {
				this._callbacks.push(callback)
			}
		},
		getResponder: function() {
			return bind(this, function(err, res) {
				if (err) { this.fail(err) }
				else { this.fulfill(res) }
			})
		},
		fulfill: function(result) {
			if (!this._waitingFor) { throw new Error('ListPromise fulfilled too many times') }
			this._results.push(result)
			if (!--this._waitingFor) {
				this._notifyAll()
			}
		},
		fail: function(error) {
			this._error = error
			this._notifyAll()
		},
		_notifyAll: function() {
			if (!this._callbacks) { return }
			for (var i=0; i<this._callbacks.length; i++) {
				this._notify(this._callbacks[i])
			}
			delete this._callbacks
		},
		_notify: function(callback) {
			callback(this._error, !this._error && this._results)
		}
	}
)
