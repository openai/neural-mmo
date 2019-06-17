/*
	Example usage:

	function log(category, arg1, arg2) { // arg3, arg4, ..., argN
		console.log('log category', category, std.slice(arguments, 1))
	}
*/
module.exports = function args(args, offset, length) {
	if (typeof length == 'undefined') { length = args.length }
	return Array.prototype.slice.call(args, offset || 0, length)
}

