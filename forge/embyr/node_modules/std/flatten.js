module.exports = function flatten(arr) {
	return Array.prototype.concat.apply([], arr)
}
