// Thanks Douglas Crockford! http://javascript.crockford.com/prototypal.html
module.exports = function create(obj, extendWithProperties) {
	function extendObject(result, props) {
		for (var key in props) {
			if (!props.hasOwnProperty(key)) { continue }
			result[key] = props[key]
		}
		return result
	}
	if (typeof Object.create == 'function') {
		module.exports = function nativeCreate(obj, extendWithProperties) {
			return extendObject(Object.create(obj), extendWithProperties)
		}
	} else {
		module.exports = function shimCreate(obj, extendWithProperties) {
			function F() {}
			F.prototype = obj
			return extendObject(new F(), extendWithProperties)
		}
	}
	return module.exports(obj, extendWithProperties)
}
