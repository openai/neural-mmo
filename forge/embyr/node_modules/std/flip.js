module.exports = function flip(obj) {
	var res = {}
	for (var key in obj) { res[obj[key]] = key }
	return res
}
