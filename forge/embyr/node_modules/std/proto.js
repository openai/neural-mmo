/* 
	// Usage: proto(prototypeObj, intantiatingFn, properties)
	
	var base = {
		say:function(arg) { alert(arg) }
	}

	var person = proto(base,
		function(name) {
			this.name = name
		}, {
			greet:function(other) {
				this.say("hello "+other.name+", I'm "+this.name)
			}
		}
	)
	
	var marcus = person("marcus"),
		john = person("john")

	marcus.greet(john)
*/

var create = require('./create')

var proto = module.exports = function proto(prototypeObject, instantiationFunction, propertiesToAdd) {
	// F is the function that is required in order to implement JS prototypical inheritence
	function F(args) {
		// When a new object is created, call the instantiation function
		return instantiationFunction.apply(this, args)
	}
	// The prototype object itself points to the passed-in prototypeObject,
	// but also has all the properties enumerated in propertiesToAdd
	F.prototype = prototypeObject ? create(prototypeObject, propertiesToAdd) : propertiesToAdd
	
	return function() {
		return new F(arguments)
	}
}
