/* Example usage:

	var UIComponent = Class(function() {
		this.init = function() { ... }
		this.create = function() { ... this.createDOM() ... }
	})

	var PublisherMixin = {
		init: function(){ ... },
		publish_: function() { ... }
	}
  
	var Button = Class(UIComponent, PublisherMixin, function(supr) {
		this.init = function(opts) {
			// call UIComponents init method, with the passed in arguments
			supr(this, 'init', arguments) // or, UIComponent.constructor.prototype.init.apply(this, arguments)
			this.color_ = opts && opts.color
		}

		// createDOM overwrites abstract method from parent class UIComponent
		this.createDOM = function() {
			this.getElement().onclick = bind(this, function(e) {
				// this.publish_ is a method added to Button by the Publisher mixin
				this.publish_('Click', e)
			})
		}
	})

*/
module.exports = function Class(/* optParent, optMixin1, optMixin2, ..., proto */) {
	var args = arguments,
		numOptArgs = args.length - 1,
		mixins = []

	// the prototype function is always the last argument
	var proto = args[numOptArgs]

	// if there's more than one argument, then the first argument is the parent class
	if (numOptArgs) {
		var parent = args[0]
		if (parent) { proto.prototype = parent.prototype }
	}

	for (var i=1; i < numOptArgs; i++) { mixins.push(arguments[i]) }

	// cls is the actual class function. Classes may implement this.init = function(){ ... },
	// which gets called upon instantiation
	var cls = function() {
		if(this.init) { this.init.apply(this, arguments) }
		for (var i=0, mixin; mixin = mixins[i]; i++) {
			if (mixin.init) { mixin.init.apply(this) }
		}
	}

	// the proto function gets called with the supr function as an argument. supr climbs the
	// inheritence chain, looking for the named method
	cls.prototype = new proto(function supr(context, method, args) {
		var target = parent
		while(target = target.prototype) {
			if(target[method]) {
				return target[method].apply(context, args || [])
			}
		}
		throw new Error('supr: parent method ' + method + ' does not exist')
	})

	// add all mixins' properties to the class' prototype object
	for (var i=0, mixin; mixin = mixins[i]; i++) {
		for (var propertyName in mixin) {
			if (!mixin.hasOwnProperty(propertyName) || propertyName == 'init') { continue }
			if (cls.prototype.hasOwnProperty(propertyName)) {
				throw new Error('Mixin property "'+propertyName+'" already exists on class')
			}
			cls.prototype[propertyName] = mixin[propertyName]
		}
	}

	cls.prototype.constructor = cls
	return cls
}
