var asyncMap = require('std/asyncMap')
var slice = require('std/slice')

module.exports = function parallel(/* fn1, fn2, ..., finishFn */) {
	var parallelFunctions = slice(arguments)
	var finish = parallelFunctions.pop()
	asyncMap(parallelFunctions, {
		parallel:parallelFunctions.length,
		iterate:function(parallelFn, done) {
			parallelFn(done)
		},
		finish:function(err, mapResults) {
			if (err) { return finish(err) }
			finish.apply(this, [null].concat(mapResults))
		}
	})
}
