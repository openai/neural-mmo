var curry = require('./curry'),
	map = require('./map'),
	each = require('./each'),
	json = require('./json')

module.exports = {
	request: request,
	get: curry(request, 'get'),
	post: curry(request, 'post'),
	jsonGet: curry(sendJSON, 'get'),
	jsonPost: curry(sendJSON, 'post')
}

var XHR = window.XMLHttpRequest || function() { return new ActiveXObject("Msxml2.XMLHTTP"); }

var onBeforeUnloadFired = false
function onBeforeUnload() {
	onBeforeUnloadFired = true
	setTimeout(function(){ onBeforeUnloadFired = false }, 100)
}

if (window.addEventListener) { window.addEventListener('beforeunload', onBeforeUnload, false) }
else { window.attachEvent('onbeforeunload', onBeforeUnload) }

function request(method, url, params, callback, headers, opts) {
	var xhr = new XHR()
	method = method.toUpperCase()
	headers = headers || {}
	opts = opts || {}
	xhr.onreadystatechange = function() {
		var err, result
		try {
			if (xhr.readyState != 4) { return }
			if (onBeforeUnloadFired) { return }
			var text = xhr.responseText,
				isJson = xhr.getResponseHeader('Content-Type') == 'application/json'
			if (xhr.status == 200 || xhr.status == 204) {
				result = isJson ? json.parse(text) : text
			} else {
				try { err = isJson ? json.parse(text) : new Error(text) }
				catch (e) { err = new Error(text) }
			}
		} catch(e) {
			err = e
		}
		if (err || typeof result != undefined) {
			_abortXHR(xhr)
			callback(err, result)
		}
	}
	
	var uriEncode = (opts.encode === false
		? function(params) { return map(params, function(val, key) { return key+'='+val }).join('&') }
		: function(params) { return map(params, function(val, key) { return encodeURIComponent(key)+'='+encodeURIComponent(val) }).join('&') })
	
	var data = ''
	if (method == 'GET') {
		var queryParams = uriEncode(params)
		url += (url.indexOf('?') == -1 && queryParams ? '?' : '') + queryParams
	} else if (method == 'POST') {
		var contentType = headers['Content-Type']
		if (!contentType) {
			contentType = headers['Content-Type'] = 'application/x-www-form-urlencoded'
		}
		if (contentType == 'application/x-www-form-urlencoded') {
			data = uriEncode(params)
		} else if (contentType == 'application/json') {
			data = json.stringify(params)
		}
	}
	xhr.open(method, url, true)
	each(headers, function(val, key) { xhr.setRequestHeader(key, val) })
	xhr.send(data)
}

function sendJSON(method, url, params, callback) {
	return request(method, url, params, callback, { 'Content-Type':'application/json' })
}

function _abortXHR(xhr) {
	try {
		if('onload' in xhr) {
			xhr.onload = xhr.onerror = xhr.ontimeout = null;
		} else if('onreadystatechange' in xhr) {
			xhr.onreadystatechange = null;
		}
		if(xhr.abort) { xhr.abort(); }
	} catch(e) {}
}
