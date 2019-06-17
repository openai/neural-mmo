var Class = require('./Class')
var map = require('./map')
var isArray = require('std/isArray')

var URL = Class(function() {

	this._extractionRegex = new RegExp([
		'^', // start at the beginning of the string
		'((\\w+:)?//)?', // match a possible protocol, like http://, ftp://, or // for a relative url
		'(\\w[\\w\\.\\-]+)?', // match a possible domain
		'(:\\d+)?', // match a possible port
		'(\\/[^\\?#]+)?', // match a possible path
		'(\\?[^#]+)?', // match possible GET parameters
		'(#.*)?' // match the rest of the URL as the hash
	].join(''), 'i')

	this.init = function(url) {
		var match = (url || '').toString().match(this._extractionRegex) || []
		this.protocol = match[2] || ''
		this.hostname = match[3] || ''
		this.port = (match[4] || '').substr(1)
		this.host = (this.hostname ? this.hostname : '') + (this.port ? ':' + this.port : '')
		this.pathname = match[5] || ''
		this.search = (match[6]||'').substr(1)
		this.hash = (match[7]||'').substr(1)
	}
	
	this.setProtocol = function(protocol) {
		this.protocol = protocol
		return this
	}

	this.toString = function() {
		return [
			this.protocol || '//',
			this.host,
			this.pathname,
			this.getSearch(),
			this.getHash()
		].join('')
	}

	this.toJSON = this.toString

	this.getTopLevelDomain = function() {
		if (!this.host) { return '' }
		var parts = this.host.split('.')
		return parts.slice(parts.length - 2).join('.')
	}
	
	this.getSearchParams = function() {
		if (this._searchParams) { return this._searchParams }
		return this._searchParams = url.query.parse(this.search) || {}
	}
	
	this.getHashParams = function() {
		if (this._hashParams) { return this._hashParams }
		return this._hashParams = url.query.parse(this.hash) || {}
	}
	
	this.addToSearch = function(key, val) { this.getSearchParams()[key] = val; return this }
	this.addToHash = function(key, val) { this.getHashParams()[key] = val; return this }
	this.removeFromSearch = function(key) { delete this.getSearchParams()[key]; return this }
	this.removeFromHash = function(key) { delete this.getHashParams()[key]; return this }
	
	this.getSearch = function() {
		return (
			this._searchParams ? '?' + url.query.string(this._searchParams)
			: this.search ? '?' + this.search
			: '')
	}
	
	this.getHash = function() {
		return (
			this._hashParams ? '#' + url.query.string(this._hashParams)
			: this.hash ? '?' + this.hash
			: '')
	}

	this.getSearchParam = function(key) { return this.getSearchParams()[key] }
	this.getHashParam = function(key) { return this.getHashParams()[key] }
})

var url = module.exports = function url(url) { return new URL(url) }
url.parse = url

url.query = {
	parse:function(paramString) {
		var parts = paramString.split('&'),
			params = {}
		for (var i=0; i<parts.length; i++) {
			var kvp = parts[i].split('=')
			if (kvp.length != 2) { continue }
			params[decodeURIComponent(kvp[0])] = decodeURIComponent(kvp[1])
		}
		return params
	},
	string:function(params) {
		return map(params, function(val, key) {
			return encodeURIComponent(key) + '=' + url.query.encodeValue(val)
		}).join('&')
	},
	encodeValue:function(val) {
		if (isArray(val)) {
			return map(val, function(v) {
				return encodeURIComponent(v)
			}).join(',')
		} else {
			return encodeURIComponent(val)
		}
	}
}
