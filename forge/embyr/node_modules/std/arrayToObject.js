module.exports = function arrayToObject(arr) {
	var obj = {}
	for (var i=0; i<arr.length; i++) { obj[arr[i]] = true }
	return obj
}