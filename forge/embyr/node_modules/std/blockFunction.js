/* 
	Block a function from being called by adding and removing any number of blocks.
	Excellent for waiting on parallel asynchronous operations.
	A blocked function starts out with exactly one block
	
	Example usage:

	http.createServer(function(req, res) {
		var sendResponse = blockFunction(function() {
			res.writeHead(204)
			res.end()
		})
		var queries = parseQueries(req.url)
		for (var i=0; i<queries.length; i++) {
			sendResponse.addBlock()
			handleQuery(queries[i]; function() {
				sendResponse.removeBlock()
			})
		}
	})
*/

module.exports = function blockFunction(fn) {
	var numBlocks = 0
	return {
		addBlock:function() {
			numBlocks++
			return this
		},
		removeBlock:function() {
			if (!fn) { throw new Error("Block removed after function was unblocked") }
			if (!numBlocks) { throw new Error("Tried to remove a block that was never added") }
			if (--numBlocks) { return }
			fn(null)
			delete fn
			return this
		},
		fail:function(error) {
			fn(error)
			delete fn
			return this
		}
	}
}
