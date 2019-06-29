// Thanks goog! http://code.google.com/p/closure-library/source/browse/trunk/closure/goog/string/string.js?r=2
module.exports = function(str) {
	return str ? str.replace(/^[\s\xa0]+|[\s\xa0]+$/g, '') : ''
}
