module.exports = function(object, propertyName, getter) {
	module.exports = object.defineGetter ? _w3cDefineGetter
		: object.__defineGetter__ ? _interimDefineGetter
		: Object.defineProperty ? _ie8DefineGetter
		: function() { throw 'defineGetter not supported' }
	return module.exports(object, propertyName, getter)
}

function defineGetter(object, propertyName, getter) {
	var fn = object.defineGetter ? _w3cDefineGetter
		: object.__defineGetter__ ? _interimDefineGetter
		: Object.defineProperty ? _ie8DefineGetter
		: function() { throw new Error('defineGetter is not supported') }
	
	module.exports.defineGetter = fn
	fn.apply(this, arguments)
}

var _w3cDefineGetter = function(object, propertyName, getter) {
	object.defineGetter(propertyName, getter)
}

var _interimDefineGetter = function(object, propertyName, getter) {
	object.__defineGetter__(propertyName, getter)
}

var _ie8DefineGetter = function(object, propertyName, getter) {
	Object.defineProperty(object, propertyName, { value:getter, enumerable:true, configurable:true })
}
