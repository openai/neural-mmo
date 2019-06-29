module.exports = function remove(arr, item) {
	if (!arr) { return }
	for (var i=0; i<arr.length; i++) {
		if (arr[i] != item) { continue }
		arr.splice(i, 1)
		return
	}
}
