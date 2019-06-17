module.exports = function recall(context, args) {
  var fn = args.callee
	return function() { return fn.apply(context, args); }
}
