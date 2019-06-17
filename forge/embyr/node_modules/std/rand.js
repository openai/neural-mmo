module.exports = function rand(floor, ceil) {
	return Math.floor(Math.random() * (ceil - floor + 1)) + floor
}