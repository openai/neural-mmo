module.exports = function merge(objA, objB) {
	var result = {},
		key
	for (key in objA) { result[key] = objA[key] }
	for (key in objB) { result[key] = objB[key] }
	return result
}