module.exports = function isArguments(obj) {
  return Object.prototype.toString.call(obj) == '[object Arguments]'
}