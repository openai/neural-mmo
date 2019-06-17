module.exports.get = function(name) {
	var regex = new RegExp(
		'(^|(; ))' + // beginning of document.cookie, or "; " which signifies the beginning of a new cookie
		name +
		'=([^;]*)') // the value of the cookie, matched up until a ";" or the end of the string
	
	var match = document.cookie.match(regex),
		value = match && match[3]
	return value && decodeURIComponent(value)
}

module.exports.set = function(name, value, duration) {
	if (duration === undefined) { duration = (365 * 24 * 60 * 60 * 1000) } // one year
	var date = (duration instanceof Date ? duration : (duration < 0 ? null : new Date(new Date().getTime() + duration))),
		expires = date ? "expires=" + date.toGMTString() + '; ' : '',
		cookieName = name + '=' + encodeURIComponent(value) + '; ',
		domain = 'domain='+document.domain+'; ',
		path = 'path=/; '

	document.cookie = cookieName + expires + domain + path
}

module.exports.isEnabled = function() {
	var name = '__test__cookie' + new Date().getTime()
	module.exports.set(name, 1)
	var isEnabled = !!module.exports.get(name)
	module.exports.remove(name)
	return isEnabled
}

module.exports.remove = function(name) {
	module.exports.set(name, "", new Date(1))
}
