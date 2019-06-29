var each = require('std/each')
var getCode = require('./getCode')
var getRequireStatements = require('./getRequireStatements')
var resolve = require('./resolve')

module.exports = getDependencyLevels

function getDependencyLevels(mainModulePath) {
	var leaves = []
	var root = []
	root.isRoot = true
	root.path = mainModulePath
	_buildDependencyTreeOf(root)

	var levels = []
	var seenPaths = {}
	_buildLevel(leaves)

	return levels

	// builds full dependency tree, noting every dependency of every node
	function _buildDependencyTreeOf(node) {
		var requireStatements = getRequireStatements(getCode(node.path))
		if (requireStatements.length == 0) {
			return leaves.push(node)
		}
		each(requireStatements, function(requireStatement) {
			var childNode = []
			childNode.path = resolve.requireStatement(requireStatement, node.path)
			childNode.parent = node
			node.push(childNode)
			_buildDependencyTreeOf(childNode)
		})
		node.waitingForNumChildren = node.length
	}

	// builds a list of dependency levels, where nodes in each level is dependent only on nodes in levels below it
	// the dependency levels allow for parallel loading of every file in any given level
	function _buildLevel(nodes) {
		var level = []
		levels.push(level)
		var parents = []
		each(nodes, function(node) {
			if (!seenPaths[node.path]) {
				seenPaths[node.path] = true
				level.push(node.path)
			}
			if (node.isRoot) { return }

			node.parent.waitingForNumChildren -= 1

			if (node.parent.waitingForNumChildren == 0) {
				parents.push(node.parent)
			}
		})
		if (!parents.length) { return }
		_buildLevel(parents)
	}
}
