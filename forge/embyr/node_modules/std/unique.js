var identity = function(item) { return item }
var itemId = function(item) { return item.id }

/*
 * Filters a list to unique items
 */
module.exports = function(list, getId) {
	if (!list || !list.length) { return [] }
	if (!getId) {
		getId = (list[0].id ? itemId : identity)
	}
	var seen = {}
	var result = []
	for (var i=0; i<list.length; i++) {
		var item = list[i]
		var id = getId(item)
		if (seen[id]) { continue }
		seen[id] = true
		result.push(item)
	}
	return result
}
