module.exports = function check(a, b) {
	if (a != b) { throw new Error("Not equal " + a + " & " + b) }
}