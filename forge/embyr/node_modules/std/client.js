var Class = require('./Class')

var mobileRegex = /mobile/i;

var Client = Class(function() {
	
	this.init = function(userAgent) {
		this._userAgent = userAgent
		this._parseBrowser()
		this._parseDevice()
	}
	
	this._parseBrowser = function() {
		(this.isChrome = this._isBrowser('Chrome'))
			|| (this.isFirefox = this._isBrowser('Firefox'))
			|| (this.isIE = this._isBrowser('MSIE'))
			|| (this.isSkyfire = this._isBrowser('Skyfire', 'Skyfire')) // Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_5_7; en-us) AppleWebKit/530.17 (KHTML, like Gecko) Version/4.0 Safari/530.17 Skyfire/2.0
			|| (this.isSafari = this._isBrowser('Safari', 'Version'))
			|| (this.isOpera = this._isBrowser('Opera', 'Version'))
		
		if (this.isOpera) {
			if (this._userAgent.match('Opera Mini')) { this.isOperaMini = true } // Opera mini is a cloud browser - Opera/9.80 (Android 2.3.4; Linux; Opera Mobi/ADR-1110171336; U; en) Presto/2.9.201 Version/11.50
		}
		
		if (this.isIE) {
			this.isChromeFrame = !!this._userAgent.match('chromeframe')
		}
		
		try {
			document.createEvent("TouchEvent")
			this.isTouch = ('ontouchstart' in window)
		} catch (e) {
			this.isTouch = false
		}
	}
	
	this._parseDevice = function() {
		((this.isIPhone = this._is('iPhone'))
			|| (this.isIPad = this._is('iPad'))
			|| (this.isIPod = this._is('iPod')))
		
		this.isAndroid = this._isBrowser('Android', 'Version')
		this.isIOS = (this.isIPhone || this.isIPad || this.isIPod)
		
		if (this.isIOS) {
			var osVersionMatch = this._userAgent.match(/ OS ([\d_]+) /),
				osVersion = osVersionMatch ? osVersionMatch[1] : '',
				parts = osVersion.split('_'),
				version = { major:parseInt(parts[0]), minor:parseInt(parts[1]), patch:parseInt(parts[2]) }
			
			this.os = { version:version }
		}
		
		if (this.isOpera && this._userAgent.match('Opera Mobi')) { this.isMobile = true } // Opera mobile is a proper mobile browser - Opera/9.80 (Android; Opera Mini/6.5.26571/ 26.1069; U; en) Presto/2.8.119 Version/10.54
		if (this.isSkyfire) { this.isMobile = true }
		if (this.isIPhone) { this.isMobile = true }
		if (this.isAndroid) {
			if (this._userAgent.match(mobileRegex)) { this.isMobile = true }
			if (this.isFirefox) { this.isMobile = true } // Firefox Android browsers do not seem to have an indication that it's a phone vs a tablet: Mozilla/5.0 (Android; Linux armv7l; rv:7.0.1) Gecko/20110928 Firefox/7.0.1 Fennec/7.0.1
		}
		
		this.isTablet = this.isIPad
	}
	
	this.isQuirksMode = function(doc) {
		// in IE, if compatMode is undefined (early ie) or explicitly set to BackCompat, we're in quirks
		return this.isIE && (!doc.compatMode || doc.compatMode == 'BackCompat')
	}
	
	this._isBrowser = function(name, versionString) {
		if (!this._is(name)) { return false }
		var agent = this._userAgent,
			index = agent.indexOf(versionString || name)
		this.version = parseFloat(agent.substr(index + (versionString || name).length + 1))
		this.name = name
		return true
	}

	this._is = function(name) {
		return (this._userAgent.indexOf(name) >= 0)
	}
})

if (typeof window != 'undefined') { module.exports = new Client(window.navigator.userAgent) }
else { module.exports = {} }

module.exports.parse = function(userAgent) { return new Client(userAgent) }
