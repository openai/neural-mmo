var dependency = require('./shared/dependency')

var el = document.body.appendChild(document.createElement('div'))
el.innerHTML = 'shared dependency:' + JSON.stringify(dependency)
