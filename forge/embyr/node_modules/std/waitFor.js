module.exports = function waitFor(num, callback) {
	var seenError
	return function(err, res) {
		if (num == 0) { return log.warn("waitFor was called more than the expected number of times") }
		if (seenError) { return }
		if (err) {
			seenError = true
			return callback(err)
		}
		num -= 1
		if (num == 0) { callback(null) }
	}
}
