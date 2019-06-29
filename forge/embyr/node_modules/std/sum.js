module.exports = function sum(list, fn) {
	if (!list) { return 0 }
	var total = 0
	for (var i=0; i<list.length; i++) {
		total += fn(list[i])
	}
	return total
}
