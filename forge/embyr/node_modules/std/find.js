module.exports = function find(items, fn) {
	if (!items) { return null }
	for (var i=0; i<items.length; i++) {
		if (fn(items[i], i)) { return items[i] }
	}
	return null
}