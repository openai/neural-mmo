module.exports = function repeat(str, times) {
	if (times < 0) { return '' }
	return new Array(times + 1).join(str)
}
