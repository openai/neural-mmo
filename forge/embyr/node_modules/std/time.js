var curry = require('./curry')
var slice = require('./slice')
var each = require('./each')

var time = module.exports = timeWithBase(1) // millisecond

function inMilliseconds() { return timeWithBase(time.milliseconds) }
function inSeconds() { return timeWithBase(time.seconds) }
function inMinutes() { return timeWithBase(time.minutes) }
function inHours() { return timeWithBase(time.hours) }
function inDays() { return timeWithBase(time.days) }
function inWeeks() { return timeWithBase(time.weeks) }

function timeWithBase(base) {
	var time = {
		now: now,
		ago: ago,
		// for creating instances of time in a different base
		inMilliseconds: inMilliseconds,
		inSeconds: inSeconds,
		inMinutes: inMinutes,
		inHours: inHours,
		inDays: inDays,
		inWeeks: inWeeks
	}

	time.millisecond = time.milliseconds = 1 / base
	time.second = time.seconds = 1000 * time.millisecond
	time.minute = time.minutes = 60 * time.second
	time.hour = time.hours = 60 * time.minute
	time.day = time.days = 24 * time.hour
	time.week = time.weeks = 7 * time.day

	function now(_base) { return Math.round(new Date().getTime() / (_base || base)) }

	function ago(ts, yield) { return ago.stepFunction(ts, yield) }
	ago.stepFunction = _stepFunction(
		10 * time.second, 'just now', null,
		time.minute, 'less than a minute ago', null,
		2 * time.minute, 'one minute ago', null,
		time.hour, '%N minutes ago', [time.minute],
		2 * time.hour, 'one hour ago', null,
		time.day, '%N hours ago', [time.hour],
		time.day * 2, 'one day ago', null,
		time.week, '%N days ago', [time.day],
		2 * time.week, '1 week ago', [time.week],
		Infinity, '%N weeks ago', [time.week])

	ago.precise = _stepFunction(
		time.minute, '%N seconds ago', [time.second],
		time.hour, '%N minutes, %N seconds ago', [time.minute, time.second],
		time.day, '%N hours, %N minutes ago', [time.hour, time.minute],
		time.week, '%N days, %N hours ago', [time.day, time.hour],
		Infinity, '%N weeks, %N days ago', [time.week, time.day])

	ago.brief = _stepFunction(
		20 * time.second, 'now', null,
		time.minute, '1 min', null,
		time.hour, '%N min', [time.minute],
		2 * time.hour, '1 hr', null,
		time.day, '%N hrs', [time.hour],
		time.day * 2, '1 day', null,
		time.week, '%N days', [time.day],
		2 * time.week, '1 week', null,
		30 * time.day, '%N weeks', [time.week],
		60 * time.day, '1 month', null,
		Infinity, '%N months', [time.day * 30])

	var MAX_TIMEOUT_VALUE = 2147483647
	function _stepFunction() {
		var steps = arguments
		var stepFn = function(unbasedTimestamp, yield) {
			var millisecondsAgo = (now(time.milliseconds) - unbasedTimestamp)
			for (var i=0; i < steps.length; i+=3) {
				if (millisecondsAgo > steps[i]) { continue }
				var result = _getStepResult(millisecondsAgo, steps, i)
				if (yield) {
					yield(result.payload)
					if (result.smallestGranularity) {
						var timeoutIn = Math.min(result.smallestGranularity - (millisecondsAgo % result.smallestGranularity), MAX_TIMEOUT_VALUE)
						setTimeout(curry(stepFn, unbasedTimestamp, yield), timeoutIn * base)
					}
				}
				return result.payload
			}
			return _getStepResult(millisecondsAgo, steps, i - 3).payload // the last one
		}
		return stepFn
	}

	function _getStepResult(millisecondsAgo, steps, i) {
		var stepSize = steps[i]
		var stepPayload = steps[i+1]
		var stepGranularities = steps[i+2]
		var smallestGranularity = stepSize
		var untakenTime = millisecondsAgo
		each(stepGranularities, function(granularity) {
			var granAmount = Math.floor(untakenTime / granularity)
			untakenTime -= granAmount * granularity
			stepPayload = stepPayload.replace('%N', granAmount)
			if (granularity < smallestGranularity) {
				smallestGranularity = granularity
			}
		})
		return { payload:stepPayload, smallestGranularity:smallestGranularity }
	}

	return time
}
