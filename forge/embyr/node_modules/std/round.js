module.exports = function round(value, decimalPoints) {
	if (!(decimalPoints > 0)) { // (undefined < 0) == false, but (!(undefined > 0)) == true
		return Math.round(value)
	}
	var granularity = Math.pow(10, decimalPoints)
	return Math.round(value * granularity) / granularity
}
