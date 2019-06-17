var std = require('std')

// TODO Send email on error
module.exports = std.Class(function() {

	var defaults = {
		name: 'Logger'
	}

	this.init = function(opts) {
		if (typeof opts == 'string') { opts = { name:opts } }
		opts = std.extend(opts, defaults)
		this._name = opts.name
		this._emailQueue = []
	}

	this.log = function() {
		this._consoleLog(std.slice(arguments))
	}

	this.error = function(err) {
		var message = this._message(err.stack)
		this._consoleLog(message)
		if (emailAlertSettings) {
			this._emailQueue.push(message)
			this._scheduleEmailDispatch()
		}
	}

	this._message = function(message) {
		return [this._name, new Date().getTime(), message]
	}

	this._consoleLog = function(messageParts) {
		console.log(joinParts(messageParts))
	}

	this._scheduleEmailDispatch = std.delay(function() {
		var mail = require('mail'),
			messages = std.map(this._emailQueue, joinParts).join('\n\n'),
			s = emailAlertSettings,
			from = [s.username, '@', s.domain].join(''),
			to = [s.alert, '+', this._name.replace(/\s/g, '_'), '@', s.domain].join('')

		this._emailQueue = []

		var message = new mail.Message({ from:from, to:to, subject: this._name + ' alert' })
		message.body(messages)
		
		var client = mail.createClient({ host:s.host, username:s.username + '@' + s.domain, password:s.password })
		client.on('error', std.bind(function(err) {
			this._consoleLog(this._message('EMAIL ALERT ERROR ' + err + ' ' + err.stack))
			client.end()
		}))

		var transactionID = new Date().getTime()
		this._consoleLog(this._message('START ALERT EMAIL TRANSACTION ' + transactionID))
		var transaction = client.mail(message.sender(), message.recipients())
		transaction.on('ready', std.bind(this, function() {
			transaction.end(message.toString())
			this._consoleLog(this._message('END ALERT EMAIL TRANSACTION ' + transactionID))
			transaction.on('end', std.bind(this, function() {
				this._consoleLog(this._message('ALERT EMAIL TRANSACTION ' + transactionID + ' COMPLETED'))
				client.quit()
			}))
		}))
	}, 10000) // Send at most one email per 10 seconds
})

function joinParts(arr) { return arr.join(' | ') }

var emailAlertSettings = null
module.exports.setupAlerts = function(s) {
	emailAlertSettings = {
		domain: s.domain,
		host: s.host,
		username: s.username,
		password: s.password,
		alert: s.alert
	}
}
