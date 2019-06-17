module.exports = (function() {
	if (Array.isArray && Array.isArray.toString().match('\\[native code\\]')) {
		return function(obj) {
			return Array.isArray(obj)
		}
	} else {
		// thanks @kangax http://perfectionkills.com/instanceof-considered-harmful-or-how-to-write-a-robust-isarray/
		return function(obj) {
			return Object.prototype.toString.call(obj) == '[object Array]'
		}
	}
})();
